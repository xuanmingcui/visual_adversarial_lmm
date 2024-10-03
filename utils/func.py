import torch
import re
import numpy as np
from transformers import CLIPVisionModel
from models.llava.conversation import conv_templates, SeparatorStyle
from models.llava.model.utils import KeywordsStoppingCriteria
import sys

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

MEAN = [0.48145466, 0.4578275, 0.40821073]
STD = [0.26862954, 0.26130258, 0.27577711]

criterion = torch.nn.CrossEntropyLoss()

def fgsm_sample_generation(vision_model, image, text_embeds, label_name, label_names, adv_temp=0.05, LR=0.5, steps=50):

    adv_label = get_different_class(label_name, label_names)
    adv_label = label_names.index(adv_label)
    
    for _ in range(steps):
        image.requires_grad = True

        image_embeds = vision_model(image).image_embeds
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)       
        logits_per_image = torch.matmul(image_embeds, text_embeds.t())

        loss = criterion(logits_per_image / adv_temp, torch.Tensor([adv_label]).reshape([1]).cuda().long())

        image.retain_grad()
        loss.retain_grad()
        loss.backward(retain_graph=True)

        image_grad = image.grad.detach().cpu().numpy()
        image_np = image.detach().cpu().numpy()- LR*image_grad
        image = torch.Tensor(image_np).cuda().half()

    return image

def get_different_class(c_true, classes):
    classes_kept = [c for c in classes if c != c_true]
    return np.random.choice(classes_kept)

def run_LLaVA(args, model, tokenizer, image_tensor, emb_type='cls'):
    # Model
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    vision_tower = model.get_model().vision_tower[0]
    if vision_tower.device.type == 'meta':
        vision_tower = CLIPVisionModel.from_pretrained(vision_tower.config._name_or_path, torch_dtype=torch.float16, low_cpu_mem_usage=True).cuda()
        model.get_model().vision_tower[0] = vision_tower

    vision_config = vision_tower.config

    vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
    vision_config.use_im_start_end = mm_use_im_start_end
    if mm_use_im_start_end:
        vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
    image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2

    qs = args.query
    if mm_use_im_start_end:
        qs = qs + '\n' + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN
    else:
        qs = qs + '\n' + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len

    conv_mode = "multimodal"
    
    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    inputs = tokenizer([prompt])

    input_ids = torch.as_tensor(inputs.input_ids).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=args.llava_temp,
            max_new_tokens=1024,
            stopping_criteria=[stopping_criteria])
    
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()

    outputs = re.sub(r"\d+\.\s[^\:]+:\s", "", outputs)

    # Split the cleaned text into a list

    parsed_responses = sent_tokenize(outputs)
    parsed_responses = [
        sent for sent in parsed_responses 
        if len(word_tokenize(sent)) >= 3 and 
        not sent.strip().lower().startswith(('sure', 'yes'))
    ]
    
    if len(parsed_responses) == 0:
        parsed_responses.append(" ")

    if emb_type=="cls":
        image_embed = model.model.image_cls_token
    elif emb_type=="mean":
        image_embed = model.model.image_mean_token
    else:
        image_embed = None

    return parsed_responses, image_embed


def classify(text_response_embeds, image_embeds, text_label_embeds, temp, scale):
    
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    text_response_embeds = text_response_embeds / text_response_embeds.norm(p=2, dim=-1, keepdim=True)

    attention_scores = torch.nn.functional.softmax(torch.mm(image_embeds, text_response_embeds.t()) / temp, dim=1)
    weighted_sum = torch.mm(attention_scores, text_response_embeds)

    combined_embeds = image_embeds * (1.0-scale) + weighted_sum * scale
    combined_embeds = combined_embeds / combined_embeds.norm(p=2, dim=-1, keepdim=True)

    ## zero-shot result with image
    logits_image = torch.matmul(image_embeds, text_label_embeds.t())
    logits_combined = torch.matmul(combined_embeds, text_label_embeds.t())

    return logits_image, logits_combined


def classify_with_descriptors(text_response_embeds, image_embeds, text_label_embeds, temp, scale):
    
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    text_response_embeds = text_response_embeds / text_response_embeds.norm(p=2, dim=-1, keepdim=True)

    attention_scores = torch.nn.functional.softmax(torch.mm(image_embeds, text_response_embeds.t()) / temp, dim=1)
    weighted_sum = torch.mm(attention_scores, text_response_embeds)

    combined_embeds = image_embeds * (1.0-scale) + weighted_sum * scale
    combined_embeds = combined_embeds / combined_embeds.norm(p=2, dim=-1, keepdim=True)

    ## zero-shot result with image

    logits_image = []
        
    for label in text_label_embeds:
        label = label / label.norm(p=2, dim=-1, keepdim=True)
        logit = torch.mm(image_embeds, label.t()).mean(-1) # 1 768 @ 768 5 -> 1,5 -> 1
        logits_image.append(logit.squeeze())

    logits_image = torch.stack(logits_image)

    logits_combined = []

    for label in text_label_embeds:
        label = label / label.norm(p=2, dim=-1, keepdim=True)
        logit = torch.mm(combined_embeds, label.t()).mean(-1)
        logits_combined.append(logit.squeeze())

    logits_combined = torch.stack(logits_combined)

    return logits_image, logits_combined


