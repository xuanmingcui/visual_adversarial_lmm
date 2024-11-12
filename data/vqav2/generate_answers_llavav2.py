import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
sys.path.append(os.getcwd())
from models.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from models.llava.conversation import conv_templates, SeparatorStyle
from models.llava.modelv2.builder import load_pretrained_model
from models.llava.utils import disable_torch_init
from models.llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader
import numpy as np

from PIL import Image
import math

from m4c_evaluator import EvalAIAnswerProcessor

answer_processor = EvalAIAnswerProcessor()

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, image_ext='pt', query_formatter=None):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.image_ext = image_ext
        self.query_formatter = query_formatter

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.query_formatter:
            qs = self.query_formatter.format(qs)
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        if self.image_ext == 'pt':
            file_path = os.path.splitext(image_file)[0] + '.pt'
            image_tensor = torch.load(os.path.join(self.image_folder, file_path))
        else:
            image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
            image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor

    def __len__(self):
        return len(self.questions)


# DataLoader
def create_data_loader(questions, image_folder, image_ext, query_formatter, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config, image_ext=image_ext, query_formatter=query_formatter)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    paths = args.image_folder.split('/')
    os.makedirs(os.path.dirname(args.result_file), exist_ok=True)
    # ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, args.image_ext, args.query_formatter, tokenizer, image_processor, model.config)
    results = []
    for (input_ids, image_tensor), line in tqdm(zip(data_loader, questions), total=len(questions)):

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

        
        results.append({"question_id": line['question_id'], "answer": answer_processor(outputs)})
        # ans_id = shortuuid.uuid()
        # ans_file.write(json.dumps({"question_id": idx,
        #                            "prompt": cur_prompt,
        #                            "text": outputs,
        #                            "answer_id": ans_id,
        #                            "model_id": model_name,
        #                            "metadata": {}}) + "\n")
        # # ans_file.flush()
    with open(args.result_file, 'w') as f:
        json.dump(results, f)
    # ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-13b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default=None, required=True)
    parser.add_argument("--result-file", type=str, default=None, required=True)
    parser.add_argument("--image_ext", type=str, default='pt')
    parser.add_argument("--query_formatter", type=str, default='{}\nAnswer the question using a single word or phrase.')
    parser.add_argument("--question-file", type=str, default="data/vqav2/coco2014val_questions.jsonl")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1_1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()
    for k,v in vars(args).items():
        print(f"{k}: {v}")
    eval_model(args)
