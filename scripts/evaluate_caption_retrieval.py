import argparse
import torch
import torch.utils.data as dutils
from typing import List
from tqdm import tqdm
import transformers
from torch.utils.data import Dataset
from PIL import Image
import os, json
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPTextModelWithProjection

from models import get_model

clip_text_tokenizer = AutoTokenizer.from_pretrained('openai/clip-vit-large-patch14-336')
clip_text_encoder = CLIPTextModelWithProjection.from_pretrained('openai/clip-vit-large-patch14-336', torch_dtype=torch.float16).cuda().eval()

class CoCoCaption(Dataset):
    def __init__(self, args, model):
        with open(args.data_file) as f:
            self.data_list = json.load(f)
        self.model = model
        self.args = args

    def __len__(self):
        return len(self.data_list)
    
    def _load_image(self, id: int):
        if self.args.image_ext == 'pt':
            file_path = os.path.join(self.args.image_folder, os.path.splitext(self.data_list[id]["image"])[0] + '.pt')
            return torch.load(file_path)
        return Image.open(os.path.join(self.args.image_folder, self.data_list[id]["image"])).convert("RGB")

    def _load_caption(self, id: int) -> List[str]:
        return self.data_list[id]["text"][:5]
    
    def _load_label(self, id: int) -> torch.Tensor:
        return self.data_list[id]["label"]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self._load_image(idx)
        caption = self._load_caption(idx)
        if self.args.image_ext != 'pt':
            image = self.model.preprocess_image(image)
        # caption = output['input_ids']

        label = self._load_label(idx)

        return image, caption, label, self.data_list[idx]['image']


def collate_fn(batch):
    return batch[0][0].unsqueeze(0), batch[0][1], batch[0][2], batch[0][3]


# Encodes all text and images in a dataset
@torch.no_grad()
def encode_both(args, model, dataloader):
    with torch.no_grad():
        image_to_text_map = []
        text_to_image_map = []

        input_encodings = []
        text_label_encodings = []

        text_index = 0
        image_index = 0

        if args.save_response and ('llava' in args.model_path or 'instruct' in args.model_path or 't5' in args.model_path):
            os.makedirs(os.path.dirname(args.save_response), exist_ok=True)
            f = open(args.save_response, 'w')

        for images, text, _, filenames in tqdm(dataloader):
            images = images.cuda().half()

            # text has shape B x 5 x 77
            batch_size, captions_per_image, _ = images.shape[0], 5, 77

            # Update text_to_image_map and image_to_text_map for this batch
            for i in range(batch_size):
                # the next image corresponds to text captions [text_index ... text_index + captions_per_image - 1]
                text_indices = list(range(text_index, text_index + captions_per_image))
                image_to_text_map.append(text_indices)
                text_index += captions_per_image

                # Each of the next captions_per_image text captions correspond to the same image
                text_to_image_map += [image_index] * captions_per_image
                image_index += 1
            
            
            if 'clip' in args.model_path:
                input_encodings.append((model.vision_encoder(images).image_embeds))
                label_input_ids = model.text_tokenizer(text, return_tensors="pt", padding=True).input_ids.cuda()
                label_input_embeds = model.text_encoder(label_input_ids).text_embeds
                text_label_encodings.append(label_input_embeds)
            elif args.model_path == 'blip2_feature_extractor':
                image_feat = model.model.extract_features({"image": images}, mode='image').image_embeds_proj
                input_encodings.append(image_feat)
                text_feat = model.model.extract_features({"text_input": text}, mode='text').text_embeds_proj
                text_label_encodings.append(text_feat[:,0,:])
            else:
                if "llava" in args.model_path:
                    response = model.generate(input_ids=model.query_input_ids, image=images)
                else:
                    response = model.generate(args.query, images)

                if args.save_response and ('llava' in args.model_path or 'instruct' in args.model_path or 't5' in args.model_path):
                    f.write(json.dumps({'image': filenames, 'text': response}))
                    f.write('\n')
                    f.flush()
                
                input_ids = clip_text_tokenizer(response, return_tensors="pt", padding=True, truncation=True).input_ids.cuda()
                input_encoding = clip_text_encoder(input_ids).text_embeds.cuda()
                input_encodings.append(input_encoding)
            
                label_input_ids = clip_text_tokenizer(text, return_tensors="pt", padding=True, truncation=True).input_ids.cuda()
                label_input_embeds = clip_text_encoder(label_input_ids).text_embeds.cuda()
                text_label_encodings.append(label_input_embeds)
        
        if args.save_response and ('llava' in args.model_path or 'instruct' in args.model_path or 't5' in args.model_path):
            f.close()

        input_encodings = torch.cat(input_encodings)
        text_label_encodings = torch.cat(text_label_encodings)

        input_encodings = F.normalize(input_encodings, dim=-1)
        text_label_encodings = F.normalize(text_label_encodings, dim=-1)
        text_to_image_map = torch.LongTensor(text_to_image_map).cuda()
        image_to_text_map = torch.LongTensor(image_to_text_map).cuda()

        return input_encodings, text_label_encodings, text_to_image_map, image_to_text_map

def recall_at_k(image_encodings, text_encodings, text_to_image_map, image_to_text_map, k_vals: List[int]):
     
    num_text = text_encodings.shape[0]
    num_im = image_encodings.shape[0]
    captions_per_image = image_to_text_map.shape[1]

    dist_matrix = image_encodings @ text_encodings.T
    if len(dist_matrix.shape) == 3:
        dist_matrix, _ = dist_matrix.max(dim=1) # BLIP2 return a 3d tensor

    dist_matrix = dist_matrix.cpu()
    inds = torch.argsort(dist_matrix, dim=1, descending=True)
    inds = inds.cuda()

    image_to_text_recall = []

    for k in k_vals:
        topk = inds[:, :k]

        correct = torch.zeros((num_im,), dtype=torch.bool).cuda()
        for i in range(captions_per_image):
            contains_index = torch.eq(topk, image_to_text_map[:, i].unsqueeze(-1)).any(dim=1)
            correct = torch.logical_or(correct, contains_index)

        num_correct = correct.sum().item()
        image_to_text_recall.append(num_correct / num_im)

    dist_matrix = dist_matrix.T
    inds = torch.argsort(dist_matrix, dim=1, descending=True)
    inds = inds.cuda()

    text_to_image_recall = []

    for k in k_vals:
        topk = inds[:, :k]
        correct = torch.eq(topk, text_to_image_map.unsqueeze(-1)).any(dim=1)
        num_correct = correct.sum().item()
        text_to_image_recall.append(num_correct / num_text)

    return text_to_image_recall, image_to_text_recall

def main(args):

    model = get_model(args).cuda().eval()

    dataset = CoCoCaption(args, model)
    dataloader = dutils.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=collate_fn)

    k_vals=[1, 5, 10, 50]

    image_encodings, text_encodings, text_to_image_map, image_to_text_map = encode_both(args, model, dataloader)

    t2i, i2t = recall_at_k(image_encodings, text_encodings, text_to_image_map, image_to_text_map, k_vals=k_vals)

    print("Text-to-image Recall@K")
    for k, x in zip(k_vals, t2i):
        print(f" R@{k}: {100*x:.2f}%")

    print("Image-to-text Recall@K")
    for k, x in zip(k_vals, i2t):
        print(f" R@{k}: {100*x:.2f}%")

    with open("results/coco_caption_results.txt", "a+") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

        f.write("Text-to-image Recall@K\n")
        for k, x in zip(k_vals, t2i):
            f.write(f" R@{k}: {100*x:.2f}%\n")
        
        f.write("Image-to-text Recall@K\n")
        for k, x in zip(k_vals, i2t):
            f.write(f" R@{k}: {100*x:.2f}%\n")

        f.write("==============================================\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-13b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-type", type=str, default=None)
    parser.add_argument("--image_ext", type=str, required=False, default='pt')
    parser.add_argument("--task", type=str, required=False, default='retrieval')
    parser.add_argument("--image-folder", type=str)
    parser.add_argument("--data-file", type=str)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1_1")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=int, default=None)
    parser.add_argument("--query", type=str, default="generate a short caption for this image: ")
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--save_response", type=str, default=None)

    args = parser.parse_args()  
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    main(args)