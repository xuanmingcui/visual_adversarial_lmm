import torch
import os, re
from transformers import AutoTokenizer, CLIPVisionModel, CLIPImageProcessor
from models.llava.model import *
from models.llava.model.utils import KeywordsStoppingCriteria
from models.llava.conversation import conv_templates, SeparatorStyle

from models.llava.modelv2.builder import load_pretrained_model
from models.llava.utils import disable_torch_init
from models.llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
import os
import json
from models.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from models.llava.conversation import conv_templates, SeparatorStyle

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

# unwanted_words = ['sure', 'okay', 'yes', 'of course', 'yeah', 'no problem']

def run_LLaVA(args, llava_model, llava_tokenizer, image_tensor, text=None, vqa=False):

    llava_model.eval()

    # Model
    mm_use_im_start_end = getattr(llava_model.config, "mm_use_im_start_end", False)
    llava_tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        llava_tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    if 'v1.5' in args.model_path:
        vision_tower = llava_model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to('cuda', dtype=torch.float16)
    else:
        vision_tower = llava_model.get_model().vision_tower[0]
    if vision_tower.device.type == 'meta':
        vision_tower = CLIPVisionModel.from_pretrained(vision_tower.config._name_or_path, torch_dtype=torch.float16, low_cpu_mem_usage=True).cuda()
        llava_model.get_model().vision_tower[0] = vision_tower

    vision_config = vision_tower.config

    vision_config.im_patch_token = llava_tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
    vision_config.use_im_start_end = mm_use_im_start_end
    if mm_use_im_start_end:
        vision_config.im_start_token, vision_config.im_end_token = llava_tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
    image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2

    qs = args.query if not text else text
    if mm_use_im_start_end:
        qs = qs + '\n' + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN
    else:
        qs = qs + '\n' + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len

    if not vqa:
        conv_mode = "multimodal"
        
        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
        else:
            args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    inputs = llava_tokenizer([prompt])

    input_ids = torch.as_tensor(inputs.input_ids).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, llava_tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = llava_model.generate(
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
    outputs = llava_tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    sentences = outputs.split('\n')
    parsed_responses = []

    if len(sentences) == 1:
        return sentences[0], llava_model.model.image_cls_token

    for response in sentences:
        
        if 'Content: ' in sentences[0]:
            match = re.search(r':\s(.*)', response)
        else:
            match = re.search(r'\d\.(.*)', response)
        if match:
            parsed_responses.append(match.group(1))
    return parsed_responses, llava_model.model.image_cls_token


def run_llava2(args, model, model_path, image_tensor, tokenizer, input_ids):
    if 'plain' in model_path and 'finetune' not in model_path.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')
        
    stop_str = conv_templates[args.conv_mode].sep if conv_templates[args.conv_mode].sep_style != SeparatorStyle.TWO else conv_templates[args.conv_mode].sep2
    input_ids = input_ids.to(device='cuda', non_blocking=True)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=128,
            use_cache=True)

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    return outputs

def run_llava2_multi_qs(model, tokenizer, image_tensor, classes, contexts = None, formatter="Is the main object in the image a {}?({})\nAnswer yes or no."):

    # input_ids = []
    output_probability = []
    batch_output_ids = []
    batch_transition_scores = []
    # max_len = -1
    if contexts == None:
        contexts = [None] * len(classes)
    for cls, context in zip(classes, contexts):
        # text = context + '\n' + formatter.format(cls)
        if context:
            text = formatter.format(cls, context)
        else:
            text = formatter.format(cls)
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + text
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + text
        
        conv = conv_templates['vicuna_v1_1'].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        # input_ids.append(prompt)
        # max_len = max(max_len, prompt.shape[-1])

    # for i in range(len(input_ids)):
    #     input_ids[i] = torch.cat([input_ids[i], torch.zeros(max_len - input_ids[i].shape[-1])])

    # input_ids = torch.stack(input_ids).long().cuda()

        with torch.inference_mode():
            results = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                temperature=0,
                max_new_tokens=5,
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=True
            )

        transition_scores = model.compute_transition_scores(results.sequences, results.scores, normalize_logits=True)
        output_ids = results.sequences
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')

        output_ids = output_ids[:, input_token_len:]
        batch_output_ids.append(output_ids.squeeze(0))
        batch_transition_scores.append(transition_scores.squeeze(0))

    
    for token_id, score in zip(batch_output_ids, batch_transition_scores):
        if 694 in token_id:
            sign = -1
            idx = token_id.tolist().index(694)
        elif 1939 in token_id:
            sign = -1
            idx = token_id.tolist().index(1939)
        elif 3869 in token_id:
            sign = 1
            idx = token_id.tolist().index(3869)
        elif 4874 in token_id:
            sign = 1
            idx = token_id.tolist().index(4874)
        else:
            print("Warning: no yes or no found in the output ids")
        prob = torch.exp(score[idx]) * sign

        output_probability.append(prob)

    output_probability = torch.tensor(output_probability, device='cuda')
    max_idx = torch.argmax(output_probability)

    # outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    # outputs = [output.strip() for output in outputs]

    return classes[max_idx]


if __name__ == '__main__':
    model_name = get_model_name_from_path('liuhaotian/llava-v1.5-13b')
    tokenizer, model, image_processor, context_len = load_pretrained_model('liuhaotian/llava-v1.5-13b', None, model_name)
    # image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0].unsqueeze(0).cuda().half()
    filename = 'COCO_val2014_000000491497'
    adv_filename = os.path.join('adv_datasets/coco/retrieval_mean/APGD/clip336_attack_params:default_289152', f'{filename}.pt')
    image_tensor = torch.load(adv_filename)
    images = image_tensor.unsqueeze(0).half().cuda()
    # classes = ['snake', 'dog', 'cat', 'bird', 'fish', 'horse', 'cow', 'sheep', 'elephant']
    # print(run_llava2_multi_qs(model, tokenizer, image_tensor, classes))

    qs = f'Is the following caption describing the image correct?\nWhite ornate seat in nicely decorated room with television.\nAnswer yes or no.'

    if getattr(model.config, 'mm_use_im_start_end', False):
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv = conv_templates['vicuna_v1_1'].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images,
            do_sample=False,
            temperature=0,
            max_new_tokens=1024,
            use_cache=True,
            output_scores = True,
            return_dict_in_generate = True
        )
    
    print(output_ids)




