from transformers import AutoTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection, \
                         Blip2ForConditionalGeneration, AutoProcessor, \
                         AutoProcessor

from models.llava.model import *
import torch
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn
from models.llava.run_llava import run_llava2, run_llava2_multi_qs
from models.llava.modelv2.builder import load_pretrained_model
from models.LAVIS.lavis.models import load_model_and_preprocess
from omegaconf import OmegaConf
from models.LAVIS.lavis.common.registry import registry
from models.llava.mm_utils import tokenizer_image_token
from models.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from models.llava.conversation import conv_templates, SeparatorStyle
from utils.helpers import make_descriptor_sentence

# from models.MiniGPT.minigpt4.common.eval_utils import prepare_texts as minigpt_prepare_texts, init_model as minigpt_init_model, eval_parser
# from models.MiniGPT.minigpt4.conversation.conversation import CONV_VISION_minigptv2
# from models.MiniGPT.minigpt4.common.config import Config

# For encoding labels/captions and the generated response, we use the same CLIP text encoder
clip_text_tokenizer = AutoTokenizer.from_pretrained('openai/clip-vit-large-patch14')
clip_text_encoder = CLIPTextModelWithProjection.from_pretrained('openai/clip-vit-large-patch14', torch_dtype=torch.float16).cuda().eval()
clip_processor = AutoProcessor.from_pretrained('openai/clip-vit-large-patch14')

@torch.no_grad()
def encode_labels_clip(model, labels):

    print("==> Loading text label embeddings...")
    if getattr(model.args, "use_descriptors", False):
        text_label_embeds = []
        for label in labels:
            examples = model.args.descriptors[label]
            sentences = []
            for example in examples:
                sentence = f"{label} {make_descriptor_sentence(example)}"
                sentences.append(sentence)
            text_descriptor = model.processor(text=sentences, padding=True, truncation=True, return_tensors="pt")['input_ids'].cuda()
            text_descriptor_embeds = model.text_encoder(text_descriptor).text_embeds.mean(0) # (N_descriptions, 768)
            text_label_embeds.append(text_descriptor_embeds) # list of (N_descriptions, 768) len = # N_labels
        text_label_embeds = torch.stack(text_label_embeds)
    else:
        if model.args.task == 'classification':
            labels = ["a photo of %s"%v for v in labels]
        
        if type(labels[0]) == list:
            text_label_embeds = []
            for label in labels:
                text_label = model.processor(label, padding=True, truncation=True, return_tensors="pt")['input_ids'].cuda()
                text_label_embed = model.text_encoder(text_label).text_embeds
                text_label_embed = text_label_embed.mean(0)
                text_label_embeds.append(text_label_embed)
            text_label_embeds = torch.stack(text_label_embeds)
        else:
            text_labels = model.processor(labels, padding=True, truncation=True, return_tensors="pt")['input_ids'].cuda()
            text_label_embeds = model.text_encoder(text_labels).text_embeds
        
    print("==> Done.")
    return F.normalize(text_label_embeds, dim=-1)

@torch.no_grad()
def encode_labels_blip(model, labels):

    print("==> Loading text label embeddings...")
    if getattr(model.args, "use_descriptors", False):
        text_label_embeds = []
        for label in labels:
            examples = model.args.descriptors[label]
            sentences = []
            for example in examples:
                sentence = f"{label} {make_descriptor_sentence(example)}"
                sentences.append(sentence)
            text_descriptor_embeds = model.encode_texts(sentences).mean(0) # (N_descriptions, 768)
            text_label_embeds.append(text_descriptor_embeds) # list of (N_descriptions, 768) len = # N_labels
        text_label_embeds = torch.stack(text_label_embeds)
    else:
        if model.args.task == 'classification':
            labels = ["a photo of %s"%v for v in labels]
        if type(labels[0]) == list:
            text_label_embeds = []
            for label in labels:
                text_label_embed = model.encode_texts(label)
                text_label_embed = text_label_embed.mean(0)
                text_label_embeds.append(text_label_embed[0,:])
            text_label_embeds = torch.stack(text_label_embeds)
        else:
            text_label_embeds = model.encode_texts(labels)[:,0,:]
        
    print("==> Done.")
    return text_label_embeds
        
class CLIP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.vision_encoder = CLIPVisionModelWithProjection.from_pretrained(args.model_path, torch_dtype=torch.float16).cuda().eval()
        self.processor = AutoProcessor.from_pretrained(args.model_path)

        self.text_tokenizer = clip_text_tokenizer
        self.text_encoder = clip_text_encoder
        self.attack_mode = False

    def preprocess_image(self, image):
        return self.processor(images=image, return_tensors="pt")['pixel_values'][0]

    def forward(self, images, **kwargs):
        """
        Forward pass to generate logits for image-caption/label retrieval
        """

        def inner():
            image_embeds = F.normalize(self.vision_encoder(images).image_embeds, p=2., dim=-1)
                    ## zero-shot result with clip
            logits = torch.matmul(image_embeds, self.text_label_embeds.t()) # B, n_label
            return logits
        
        if self.attack_mode:
            # self.text_label_embeds.requres_grad = True
            logits = inner()
        else:
            with torch.inference_mode():
                logits = inner()
        return logits

    def set_encoded_labels(self, labels):
        self.text_label_embeds = encode_labels_clip(self, labels)
        return self.text_label_embeds
    
class LLaVA2(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model_path = args.model_path
        self.processor = AutoProcessor.from_pretrained('openai/clip-vit-large-patch14-336')
        self.tokenizer, self.model, self.image_processor, context_len = load_pretrained_model("liuhaotian/llava-v1.5-13b", None, 'llava-v1.5-13b')
        self.vision_encoder = CLIPVisionModelWithProjection.from_pretrained('openai/clip-vit-large-patch14-336', torch_dtype=torch.float16).cuda()

        self.text_tokenizer = clip_text_tokenizer
        self.text_encoder = clip_text_encoder
        
        if args.task == 'classification' or 'retrieval' in args.task and args.task != 'llm_retrieval_classification_multi_qs':
            if self.model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + args.query
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + args.query

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            self.query_input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    
    def preprocess_image(self, image):
        return self.processor(images=image, return_tensors="pt")['pixel_values'][0]
        # return process_images([image], self.image_processor, self.model.config)[0]
    
    def generate(self, input_ids=None, image=None, classes=None, contexts=None, **kwargs):
        if len(image.shape) == 3:
            image = image.unsqueeze(0) # make it a batch
        with torch.inference_mode():
            if self.args.task == 'classification_with_context_multi_qs' or self.args.task == 'llm_retrieval_classification_multi_qs':
                response = run_llava2_multi_qs(self.model, self.tokenizer, image, classes, contexts, formatter=self.args.multi_qs_formatter)
            else:
                response = run_llava2(self.args, model=self.model, 
                                model_path=self.model_path,
                                tokenizer=self.tokenizer, 
                                image_tensor=image, input_ids=input_ids)
        return response

    def forward(self, images, query_input_ids=None, return_response=False, **kwargs):
        """
            Generate responses and logits for image-caption/label retrieval via LLAVA. 
            *** In this case, the input_ids has to be the same for all images. (e.g. a standard query: "describe the image") ***
        """

        if self.args.task != 'classification_with_context_multi_qs' and self.args.task != 'llm_retrieval_classification_multi_qs':
            if query_input_ids is None:
                query_input_ids = self.query_input_ids
                if self.query_input_ids.shape[0] == 1:
                    query_input_ids = query_input_ids.expand(images.shape[0], -1)

        with torch.inference_mode():
            responses, text_response_embeds = [], []
            for i, image in enumerate(images):
                if self.args.task == 'classification_with_context_multi_qs':
                    response = self.generate(classes=kwargs['text'][i], image=image, contexts=kwargs['contexts'][i])
                elif self.args.task == 'llm_retrieval_classification_multi_qs':
                    response = self.generate(classes=kwargs['text'][i], image=image, contexts=None)
                else:
                    response = self.generate(query_input_ids[i].unsqueeze(0), image)

                if 'classification' in self.args.task:
                    response = f"a photo of {response}."
                responses.append(response)
                text_response = self.text_tokenizer(text=response, padding=True, truncation=True, return_tensors="pt")['input_ids'].cuda()
                text_response_embed = F.normalize(self.text_encoder(text_response).text_embeds, p=2., dim=-1)
                text_response_embeds.append(text_response_embed)

            text_response_embeds = torch.stack(text_response_embeds, dim=0) # B, n_sentence, 768
            logits = torch.matmul(text_response_embeds, self.text_label_embeds.t()).mean(dim=1)
        if return_response:
            return logits, responses
        else:
            return logits

    def set_encoded_labels(self, labels):
        self.text_label_embeds = encode_labels_clip(self, labels)
        return self.text_label_embeds


class BLIP2CL(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model, self.vis_processors, self.text_processors = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain", is_eval=True)
        self.args = args
        self.tokenizer = self.model.tokenizer
        self.attack_mode = False
    
    def preprocess_image(self, image):
        return self.vis_processors['eval'](image)
    
    def forward(self, images, texts=None, **kwargs):
        """
        Forward pass to generate logits for image-caption/label retrieval
        if text is not None, then to return contrastive loss
        otherwise, return logits for classification based on itm or itc
        """
        def inner():
            image_feat = self.model.extract_features({"image": images}, mode='image').image_embeds_proj
            if texts:
            # image_embeds_proj and text_embeds_proj are normalized already
                text_feat = self.model.extract_features({"text_input": texts}, mode='text').text_embeds_proj[:,0,:]
            else:
                text_feat = self.text_label_embeds
            return (image_feat @ text_feat.t()).max(1)[0]
                
        if not self.attack_mode:
            with torch.inference_mode():
                return inner()
        else:
            self.text_label_embeds.requres_grad = True
            return inner()

    def encode_images(self, images):
        # B dim
        return self.model.extract_features({"image": images}, mode='image').image_embeds_proj

    def encode_texts(self, texts):
        # B # query_tokens dim
        processed_texts = [self.text_processors['eval'](t) for t in texts]
        return self.model.extract_features({"text_input": processed_texts}, mode='text').text_embeds_proj
    
    def set_encoded_labels(self, labels):

        self.text_label_embeds = encode_labels_blip(self, labels)
        return self.text_label_embeds


class GenerativeBLIP(nn.Module):
    
    def __init__(self, args) -> None:
        """
        *** Only support batch_size=1
        """
        super().__init__()
        self.args = args
        self.model, self.vis_processors, self.text_processors = load_model_and_preprocess(name=args.model_path, model_type=args.model_type, is_eval=True)
        
        self.text_tokenizer = clip_text_tokenizer
        self.text_encoder = clip_text_encoder
        self.processor = clip_processor
    
    def preprocess_image(self, image):
        return self.vis_processors['eval'](image)

    def generate(self, text, image):
        if len(image.shape) == 3:
            image = image.unsqueeze(0) # make it a batch
        
        if type(text) == tuple or type(text) == list or type(text) == torch.tensor:
            text = text[0]
        text = self.text_processors['eval'](text)

        with torch.inference_mode():
            if 'retrieval' in self.args.task:
                return self.model.generate({"image": image, 'text_input': text})[0]
            return self.model.predict_answers({"image": image, 'text_input': text}, inference_method="generate")[0]
        
    def forward(self, images, text, **kwargs):
        """
        Forward pass to generate logits for image-caption/label retrieval
        if text is not None, then to return contrastive loss
        otherwise, return logits for classification based on itm or itc
        """

        with torch.inference_mode():

            response = self.generate(text, images)
            if 'classification' in self.args.task:
                response = f"a photo of {response}."
            text_response = self.text_tokenizer(text=response, padding=True, truncation=True, return_tensors="pt")['input_ids'].cuda()
            text_response_embed = F.normalize(self.text_encoder(text_response).text_embeds, p=2., dim=-1)
            
            logits = torch.matmul(text_response_embed, self.text_label_embeds.t())
        
        return logits

    def encode_texts(self, texts):
        with torch.no_grad():
            text_embeds = self.text_encoder(self.text_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")['input_ids'].cuda()).text_embeds
        
        return F.normalize(text_embeds, dim=-1)
    
    def set_encoded_labels(self, labels):
        self.text_label_embeds = encode_labels_clip(self, labels)
        return self.text_label_embeds

# class MiniGPT(nn.Module):
    
#     def __init__(self, args) -> None:
#         """
#         *** Only support batch_size=1
#         """
#         super().__init__()
#         self.args = args
#         self.model, self.vis_processors = minigpt_init_model(args)
#         conv_temp = CONV_VISION_minigptv2.copy()
#         conv_temp.system = ""
#         self.model.eval()
        
#         self.text_tokenizer = clip_text_tokenizer
#         self.text_encoder = clip_text_encoder
#         self.processor = clip_processor
    
#     def preprocess_image(self, image):
#         return self.vis_processors(image)

#     def generate(self, text, image):
#         if len(image.shape) == 3:
#             image = image.unsqueeze(0) # make it a batch
        
#         if type(text) == tuple or type(text) == list or type(text) == torch.tensor:
#             text = text[0]
#         text = minigpt_prepare_texts(text)

#         with torch.inference_mode():
#             answers = model.generate(image, text, max_new_tokens=20, do_sample=False)
#             answer = answer.lower().replace('<unk>','').strip()
        
#     def forward(self, images, text, **kwargs):
#         """
#         Forward pass to generate logits for image-caption/label retrieval
#         if text is not None, then to return contrastive loss
#         otherwise, return logits for classification based on itm or itc
#         """

#         with torch.inference_mode():

#             response = self.generate(text, images)
#             if 'classification' in self.args.task:
#                 response = f"a photo of {response}."
#             text_response = self.text_tokenizer(text=response, padding=True, truncation=True, return_tensors="pt")['input_ids'].cuda()
#             text_response_embed = F.normalize(self.text_encoder(text_response).text_embeds, p=2., dim=-1)
            
#             logits = torch.matmul(text_response_embed, self.text_label_embeds.t())
        
#         return logits

#     def encode_texts(self, texts):
#         with torch.no_grad():
#             text_embeds = self.text_encoder(self.text_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")['input_ids'].cuda()).text_embeds
        
#         return F.normalize(text_embeds, dim=-1)
    
#     def set_encoded_labels(self, labels):
#         self.text_label_embeds = encode_labels_clip(self, labels)
#         return self.text_label_embeds



def get_model(args):
    if 'llava-v1.5' in args.model_path.lower():
        model = LLaVA2(args)
    elif 'clip' in args.model_path.lower():
        model = CLIP(args)
    elif args.model_path.lower() == 'blip2_feature_extractor':
        model = BLIP2CL(args)
    elif 'instruct' or 'blip2_t5' in args.model_path.lower():
        model = GenerativeBLIP(args)
    # elif 'minigpt' in args.model_path.lower():
    #     model = MiniGPT(args)
    else:
        raise ValueError(f"Model {args.model_path} not supported.")
    
    model.eval()
    return model

if __name__ == '__main__':


    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16).cuda().eval()
    image = "/groups/sernam/datasets/coco/val2014/COCO_val2014_000000000192.jpg"
    image = Image.open(image).convert('RGB')
    prompt = "What is the main object in this image? Answer: "
    inputs = processor(images=image, text=prompt, return_tensors="pt").to('cuda', torch.float16)
    generated_ids = model.generate(**inputs, max_new_tokens=1)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    print(generated_text)

