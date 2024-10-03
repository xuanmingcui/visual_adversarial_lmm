import argparse
from models.llava.modelv2.builder import load_pretrained_model
from models.llava.mm_utils import tokenizer_image_token, get_model_name_from_path
import json
from models.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from models.llava.conversation import conv_templates, SeparatorStyle
import torch
from tqdm import tqdm

def generate_class_descriptions(args):
    with open(args.data_path, 'r') as f:
        labels = json.load(f)

    format_query = "Write a sentence of general description of the visual features of a {}."

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, None, model_name)

    result = {}

    for label in tqdm(labels):
        query = format_query.format(label)
        conv = conv_templates['vicuna_v1_1'].copy()
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=None,
                do_sample=False,
                temperature=0,
                max_new_tokens=1024,
                use_cache=True,
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

        result[label] = outputs

    with open(args.save_path, 'w') as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default='liuhaotian/llava-v1.5-13b')
    parser.add_argument("--data-path", type=str, default="data/coco/coco_labels_val2014.json")
    parser.add_argument("--save-path", type=str)
    args = parser.parse_args()

    generate_class_descriptions(args)