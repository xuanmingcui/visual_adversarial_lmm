import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava import LlavaLlamaForCausalLM
from models.llava.conversation import conv_templates
from models.llava.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria

from PIL import Image
import random
import math

# FOR DATA AUGMENTATION
from models.llava.data.data_util import *

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


def patch_config(config):
    patch_dict = {
        "use_mm_proj": True,
        "mm_vision_tower": "openai/clip-vit-large-patch14",
        "mm_hidden_size": 1024
    }

    cfg = AutoConfig.from_pretrained(config)
    if not hasattr(cfg, "mm_vision_tower"):
        print(f'`mm_vision_tower` not found in `{config}`, applying patch and save to disk.')
        for k, v in patch_dict.items():
            setattr(cfg, k, v)
        cfg.save_pretrained(config)


def eval_model(args):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    if 'lora' in model_name.lower():
        lora_cfg_pretrained = AutoConfig.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if args.mm_projector is None:
        patch_config(model_name)
        if 'lora' in model_name.lower():
            print('Loading LLaVA from base model...')
            llama_state_dict = AutoModelForCausalLM.from_pretrained(args.base_model_path, torch_dtype=torch.float16).state_dict()
            model = LlavaLlamaForCausalLM.from_pretrained(args.base_model_path, config=lora_cfg_pretrained, state_dict=llama_state_dict, torch_dtype=torch.float16, ignore_mismatched_sizes=True)

            print('Loading additional LLaVA weights...')
            if os.path.exists(os.path.join(model_name, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_name, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_name, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.embed_tokens') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            non_lora_trainables = {k: v.to(torch.float16) for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_name)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Moving to CUDA...')
            model = model.cuda()
        else:
            # This is the model being used for experiments
            model = LlavaLlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, use_cache=True).cuda()

        image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower, torch_dtype=torch.float16)

        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

        vision_tower = model.model.vision_tower[0]
        print(f'Vision tower: {vision_tower}')
        vision_tower.to(device='cuda', dtype=torch.float16)
        vision_config = vision_tower.config
        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
        vision_config.use_im_start_end = mm_use_im_start_end
        if mm_use_im_start_end:
            vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
        image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2
    else:
        # in case of using a pretrained model with only a MLP projector weights
        model = LlavaLlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, use_cache=True).cuda()

        vision_tower = CLIPVisionModel.from_pretrained(args.vision_tower, torch_dtype=torch.float16).cuda()
        image_processor = CLIPImageProcessor.from_pretrained(args.vision_tower, torch_dtype=torch.float16)

        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

        vision_config = vision_tower.config
        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
        vision_config.use_im_start_end = mm_use_im_start_end
        if mm_use_im_start_end:
            vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])

        image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2

        mm_projector = torch.nn.Linear(vision_config.hidden_size, model.config.hidden_size)
        mm_projector_weights = torch.load(args.mm_projector, map_location='cpu')
        mm_projector.load_state_dict({k.split('.')[-1]: v for k, v in mm_projector_weights.items()})

        model.model.mm_projector = mm_projector.cuda().half()
        model.model.vision_tower = [vision_tower]

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    # with open(os.path.expanduser(args.question_file), "r") as f:
    #     questions = json.load(f)

    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    all_conversations = {}

    all_answers = []
    num_images_not_found = 0


    for i, line in enumerate(tqdm(questions)):
        print(line)
        idx = line["question_id"]
        image_file = line["image"]
        
        qs = line["text"]
        cur_prompt = qs
        if mm_use_im_start_end:
            qs = qs + '\n' + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN
        else:
            qs = qs + '\n' + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len

        conv = conv_templates[args.conv_mode].copy()

        conv.append_message(conv.roles[0], qs)

        # Add a blank message to the conversation if not using simple mode
        # Commenting this out so I can append the message to the role later,
        # without having the blank message in the conversation history
        if args.conv_mode != 'simple':
            conv.append_message(conv.roles[1], "")

        prompt = conv.get_prompt()

        inputs = tokenizer([prompt])

        # Get the correct image from dataset
        try:
            image = Image.open(os.path.join(args.image_folder, 'COCO_train2014_' + image_file))
        except Exception as e:
            print(f'Failed to open train image. Error: {e}')
            try:
                image = Image.open(os.path.join(args.image_folder, 'COCO_val2014_' + image_file))
            except Exception as e:
                print(f'Failed to open validation image. Error: {e}')


        # try:
        #     image = Image.open(os.path.join(args.image_folder, image_file))
        #     # Continue processing the image...
        # except Exception as e:
        #     print(f"Image file not found: {image_file}", flush=True)
        #     num_images_not_found += 1
        #     print(f"num Images not found: {num_images_not_found}", flush=True)
        #     continue


        print(f'Number of images not found: {num_images_not_found}', flush=True)
        # DATA AUGMENTATION
        # Apply data augmentation to the image if required
        if args.data_augmentation is not None:
            image = apply_data_augmentation(image, args.data_augmentation, image_file)


        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        input_ids = torch.as_tensor(inputs.input_ids).cuda()

        # new stopping implementation
        class KeywordsStoppingCriteria(StoppingCriteria):
            def __init__(self, keywords, tokenizer, input_ids):
                self.keywords = keywords
                self.tokenizer = tokenizer
                self.start_len = None
                self.input_ids = input_ids

            def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                if self.start_len is None:
                    self.start_len = self.input_ids.shape[1]
                else:
                    outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
                    for keyword in self.keywords:
                        # print(f'Keywords: {self.keywords}')
                        if keyword in outputs:
                            return True
                return False

        if args.conv_mode == 'simple':
            keywords = ['###']
        else:
            keywords = [conv.sep2]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        # print(f"Current model information: {model}", flush=True)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=True,
                temperature=0.1,
                max_new_tokens=2048,
                use_cache=True,
                stopping_criteria=[stopping_criteria]
                # , max_length=20
                )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] Sample {i}: {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0].strip()


        if args.conv_mode == 'simple':
            while True:
                cur_len = len(outputs)
                outputs = outputs.strip()
                for pattern in ['###', 'Assistant:', 'Response:']:
                    if outputs.startswith(pattern):
                        outputs = outputs[len(pattern):].strip()
                if len(outputs) == cur_len:
                    break

            try:
                index = outputs.index(conv.sep)
            except ValueError:
                outputs += conv.sep
                index = outputs.index(conv.sep)

            outputs = outputs[:index].strip()
        else:
            outputs = outputs.strip()

        # Strip the image patch tokens from the message history
        if args.conv_mode == 'llava_v1':
            for index, message in enumerate(conv.messages):
                role, text = message
                if role == 'USER':  # Only process the user's messages
                    text = text.split('<im_start>')[0].strip()  # Split on the image tag and keep the question part
                conv.messages[index][1] = text

        # Append the output to the next prompt
        # Not using this anymore
        # if i < len(questions) - 1:  # ensure we're not at the last question
        #     idx_next_question = questions[i+1]["question_id"]
        #     if idx == idx_next_question:
        #         questions[i+1]['text'] = questions[i]['text'] + outputs + ' ' + questions[i+1]['text']


        conv.append_message(conv.roles[1], outputs)

        ans_id = shortuuid.uuid()

        # Commenting this out temporarily to see if I can save the message history to answer file
        # This is how they write to the file in the original code
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                #    "answer_id": ans_id,
                                #    "model_id": model_name,
                                #    "metadata": {}
                                   }
                                   ) + "\n")

        # Create a dictionary with conversation history
        conversation = [{"from": msg[0], "value": msg[1]} for msg in conv.messages]

        # for msg in conversation:
        #     print(f"From: {msg['from']}, Value: {msg['value']}", flush=True)




        # prompt = conv.get_prompt()

    #     if idx not in all_conversations:
    #         all_conversations[idx] = []

    #     # Make sure we don't go out of bounds
    #     if i < len(questions) - 1:
    #         # Get the next question id
    #         idx_next_question = questions[i+1]["question_id"]

    #         # If the index of the current and next question match
    #         # Append the q/a pair to the current question
    #         if idx == idx_next_question:
    #             all_conversations[idx].extend(conversation)
    #         else:
    #             all_conversations[idx].append(conversation[-2])
    #             all_conversations[idx].append(conversation[-1])
    #             final_dict = {"id": idx, "conversation": all_conversations[idx]}

    #             # Append the conversation to the list
    #             all_answers.append(final_dict)

    # json.dump(all_answers, ans_file, indent=4)




    # Don't forget to close the file after writing to it
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--base-model-path", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--mm-projector", type=str, default=None)
    parser.add_argument("--vision-tower", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="llava_v1") # the default is "simple", I changed it to "llava_v1"
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--data-augmentation", type=str, default=None) # Pass in data augmentation arg
    args = parser.parse_args()

    eval_model(args)
