import argparse
import torch
import os
import json
from tqdm import tqdm
import sys
sys.path.append(os.getcwd())
from models.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from models.llava.conversation import conv_templates, SeparatorStyle
from models.llava.modelv2.builder import load_pretrained_model
from models.llava.utils import disable_torch_init
from models.llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    image_folder = os.path.expanduser(args.image_folder)
    image_folders = sorted(os.listdir(image_folder), key=int)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for dir in tqdm(image_folders):
        for image_name in os.listdir(os.path.join(image_folder, dir)):
            if args.image_ext == 'pt':
                image_name = os.path.splitext(image_name)[0] + '.pt'
                image_tensor = torch.load(os.path.join(image_folder, dir, image_name))
                images = image_tensor.unsqueeze(0).half().cuda()
            else:
                image = Image.open(os.path.join(image_folder, dir, image_name))
                image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                images = image_tensor.unsqueeze(0).half().cuda()
            if getattr(model.config, 'mm_use_im_start_end', False):
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + args.prompt
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + args.prompt

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = [KeywordsStoppingCriteria(keywords, tokenizer, input_ids)] if conv.version == "v0" else None

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    max_new_tokens=1024,
                    use_cache=True,
                    stopping_criteria=stopping_criteria,
                )

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()

            ans_file.write(json.dumps({"image": os.path.join(dir, image_name), "text": outputs}) + "\n")
            ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-13b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="datasets/vqa/scienceqa/test")
    parser.add_argument("--answers-file", type=str, default="datasets/vqa/scienceqa/captions.jsonl")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1_1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--image_ext", type=str, default='jpg')
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--prompt", type=str, default="What is this image about? Answer in one sentence.")

    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    eval_model(args)
