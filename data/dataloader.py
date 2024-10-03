import torch
import torch.utils.data as dutils
from typing import Any, List, Union
import transformers
from torch.utils.data import Dataset
from PIL import Image
import os, json
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from models.llava.mm_utils import text_to_input_ids
from models.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import DataLoader
from utils.helpers import read_file

transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
])

denormalize = transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225), 
                                   (1/0.229, 1/0.224, 1/0.225))

class BaseDataset(Dataset):
    def __init__(self, args, model):
        self.args = args
        self.data_file = args.data_file 
        self.image_folder = args.image_folder
        self.data_list = np.array(read_file(self.data_file))
        self.task = args.task
        self.transfer = args.transfer
        self.use_descriptors = args.use_descriptors
        self.model = model
        
        self.image_ext = args.image_ext
        self.query_input_ids = None

        if  'classification' in args.task:
            self.label_list = list(set([line['text'] for line in self.data_list]))
            if 'classification_with_context' in args.task:
                assert args.context_file is not None
                with open(args.context_file, 'r') as f:
                    self.contexts = json.load(f)
        elif args.task == 'retrieval_mean':

            self.label_list = [line['text'] for line in self.data_list]
        else:
            self.label_list = list(set([line['text'] for line in self.data_list]))

        self.label_list = [label.lower() for label in self.label_list]

    def __len__(self):
        return len(self.data_list)
    
    def _load_image(self, id: int):
        
        file_path = os.path.join(self.image_folder, self.data_list[id]["image"])
        if self.image_ext == 'pt':
            file_path = os.path.splitext(file_path)[0] + '.pt'
            file = torch.load(file_path)
            if self.transfer:
                file = transforms.ToPILImage()(denormalize(file))
            return file, self.data_list[id]["image"]

        return Image.open(file_path).convert("RGB"), self.data_list[id]["image"]
    
    def _load_label(self, id: int) -> Union[torch.Tensor, str]:
        label_name = self.data_list[id]['text']  #.lower()
        label = self.label_list.index(label_name)
        return label
    
    def __getitem__(self, index) -> Any:
        raise NotImplementedError()

class CLIPDataset(BaseDataset):
    def __init__(self, args, model):
        
        super().__init__(args, model)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image, base_path = self._load_image(idx)
        if self.image_ext != 'pt' or self.transfer:
            image = self.model.processor(images=image, return_tensors='pt')['pixel_values'][0]
        label = self._load_label(idx) 
        # print(label)
        return image, base_path, label

class LLAVA2Dataset(CLIPDataset):
    def __init__(self, args, model):
        
        """
        *** Dataset for LLAVA V1.5.
        """

        super().__init__(args, model)
        self.model = model
        # For classification and retrieval, query is the same across all images
        if self.task == 'classification_with_context':
            if model.args.query.count("{}") == 0:
                self.query_input_ids = {k: text_to_input_ids(v+f'\n{model.args.query}', self.model) for k,v in self.contexts.items()}
            elif model.args.query.count("{}") == 1:
                self.query_input_ids = {k: text_to_input_ids(model.args.query.format(v), self.model) for k,v in self.contexts.items()}
        else:
            self.query_input_ids = None if not hasattr(model.args, 'query') or model.args.query == None or model.args.query == '' else text_to_input_ids(self.model.args.query, self.model).unsqueeze(0)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image, base_path = self._load_image(idx)
        image_tensor = image if self.image_ext == 'pt' and (not self.transfer) else self.model.preprocess_image(image)

        label = self._load_label(idx) 
        if self.task == 'classification':
            
            return image_tensor, base_path, label

        elif self.task == 'classification_with_context':
            input_ids = self.query_input_ids[self.data_list[idx]['text']]
            return image_tensor, base_path, label, input_ids
        
        elif self.task == "classification_with_context_multi_qs":
            correct_class = self.data_list[idx]['text']
            other_classes = [c for c in self.label_list if c != correct_class]
            random_other_classes = list(np.random.choice(other_classes, 19, replace=False))
            random_other_classes.append(correct_class)
            contexts = [self.contexts[c] for c in random_other_classes]
            return image_tensor, base_path, label, random_other_classes, contexts

        elif self.task == 'retrieval_mean':
            input_ids = self.query_input_ids[0]
            return image_tensor, base_path, label, input_ids

        elif self.task == 'llm_retrieval_classification_multi_qs':
            correct_class = self.data_list[idx]['text']
            other_classes = [c for c in self.label_list if c != correct_class]
            random_other_classes = np.random.choice(other_classes, 19, replace=False).tolist()
            random_other_classes.append(correct_class)
            return image_tensor, base_path, label, random_other_classes

class BLIP2CLDataset(BaseDataset):
    def __init__(self, args, model):
        
        """
        *** Dataset for BLIP image/text model
        """

        super().__init__(args, model)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image, base_path = self._load_image(idx)
        image_tensor = image if self.image_ext == 'pt' and (not self.transfer) else self.model.vis_processors['eval'](image)
        label_name = self.data_list[idx]['text'] if 'text' in self.data_list[idx] else self.data_list[idx]['captions']
        label = self.label_list.index(label_name)
        # print(label)
        return image_tensor, base_path, label


class InstructBLIPDataset(BaseDataset):
    def __init__(self, 
                dataset = 'coco',
                task = 'classification',
                image_folder = None,
                context_file = None,
                data_file = None,
                subset = None,
                transfer = False,
                use_descriptors = False,
                image_ext = 'pt',
                model: Any = None):
        
        """
        *** Dataset for InstructBLIP or BLIP2-T5
        """

        super().__init__(dataset=dataset, task=task, 
                         image_folder=image_folder, subset=subset, 
                         context_file=context_file,
                         data_file=data_file,
                         image_ext=image_ext,        
                         transfer=transfer,
                         use_descriptors=use_descriptors)
        self.model = model
        self.query_input_ids = None if not hasattr(model.args, 'query') else model.encode_texts(self.model.args.query).cuda()

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image, base_path = self._load_image(idx)
        image_tensor = image if self.image_ext == 'pt' and (not self.transfer) else self.model.vis_processors['eval'](image)
        label = self._load_label(idx) 
        if self.task == 'classification':
            # print(label)
            return image_tensor, base_path, label
        elif self.task == 'classification_with_context':
            text = f"Context: {self.contexts[self.data_list[idx]['text']]} {self.model.args.query}"
            return image_tensor, base_path, label, text

        elif self.task == 'retrieval_mean':
            return image_tensor, base_path, label, self.model.args.query
            
class MiniGPTDataset(BaseDataset):
    def __init__(self, 
                dataset = 'coco',
                task = 'classification',
                context_file = None,
                image_folder = None,
                data_file = None,
                subset = None,
                transfer = False,
                use_descriptors = False,
                image_ext = 'pt',
                model: Any = None):
        
        """
        *** Dataset for MiniGPT
        """

        super().__init__(dataset=dataset, task=task, 
                         image_folder=image_folder, subset=subset, 
                         context_file=context_file,
                         data_file=data_file,
                         image_ext=image_ext,        
                         transfer=transfer,
                         use_descriptors=use_descriptors)
        self.model = model
        self.query_input_ids = None if not hasattr(model.args, 'query') else model.encode_texts(self.model.args.query).cuda()

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image, base_path = self._load_image(idx)
        image_tensor = image if self.image_ext == 'pt' and (not self.transfer) else self.model.preprocess_image(image)
        label = self._load_label(idx) 
        if self.task == 'classification':
            # print(label)
            return image_tensor, base_path, label
        elif self.task == 'classification_with_context':
            text = f"Context: {self.contexts[self.data_list[idx]['text']]} {self.model.args.query}"
            return image_tensor, base_path, label, text

        elif self.task == 'retrieval_mean':
            return image_tensor, base_path, label, self.model.args.query
            

def collate_fn(batch):
    images, base_paths, label, text, context = zip(*batch)
    images = torch.stack(images)
    label = torch.tensor(label)
    return images, base_paths, label, text, context

def collate_fn_no_context(batch):
    images, base_paths, label, text = zip(*batch)
    images = torch.stack(images)
    label = torch.tensor(label)
    return images, base_paths, label, text

def get_dataloader(args, model):    
    if 'llava-v1.5' in args.model_path:
        dataset_cls = LLAVA2Dataset
    elif 'llava' in args.model_path or 'clip' in args.model_path:
        dataset_cls = CLIPDataset
    elif args.model_path == 'blip2_feature_extractor':
        dataset_cls = BLIP2CLDataset
    elif 'instruct' in args.model_path or 't5' in args.model_path or 'minigpt' in args.model_path:
        dataset_cls = InstructBLIPDataset
    else:
        raise NotImplementedError(f"Dataset for {args.model_path} not implemented")
    
    dataset = dataset_cls(args, model)
    
    if args.task == 'classification_with_context_multi_qs':
        dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, collate_fn=collate_fn)
    elif args.task == 'llm_retrieval_classification_multi_qs':
        dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, collate_fn=collate_fn_no_context)
    else:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    return dataset, dataloader


if __name__ == '__main__':
    dataset = CLIPDataset(dataset='imagenet', use_descriptors=True)
