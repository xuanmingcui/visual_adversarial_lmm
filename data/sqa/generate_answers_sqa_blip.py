import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from models.LAVIS.lavis.models import load_model_and_preprocess

from PIL import Image

def eval_model(args):
    # Model
    
    model, vis_processors, _ = load_model_and_preprocess(name=args.model_path, model_type=args.model_type, is_eval=True, device='cuda')
    model_path = os.path.expanduser(args.model_path)

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for i, line in enumerate(tqdm(questions)):
        if 'image' not in line:
            continue
        idx = line["id"]
        question = line['conversations'][0]
        qs = question['value'].replace('<image>', '').strip()

        if args.single_pred_prompt:
            qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
            cur_prompt = question['value'] + '\n' + "Answer with the option's letter from the given choices directly."

        image_file = line["image"]
        if args.image_ext == 'pt':
            image_file = os.path.splitext(image_file)[0] + '.pt'
            image_tensor = torch.load(os.path.join(args.image_folder, image_file))
            image_tensor = image_tensor.unsqueeze(0).half().cuda()
        else:
            image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
            image_tensor = vis_processors['eval'](image).unsqueeze(0).cuda().half()

        with torch.inference_mode():
            outputs = model.predict_answers({"text_input": qs, "image": image_tensor}, inference_method="generate")[0]

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": args.model_path + args.model_type,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="blip2_opt")
    parser.add_argument("--model-type", type=str, default="pretrain_opt2.7b")
    parser.add_argument("--image_ext", type=str, default="pt")
    parser.add_argument("--image-folder", type=str, default='data/vqa/scienceqa/test')
    parser.add_argument("--question-file", type=str, default="data/vqa/scienceqa/llava_test_CQM-A.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--single-pred-prompt", action="store_true")
    args = parser.parse_args()

    for k,v in vars(args).items():
        print(f"{k}: {v}")

    eval_model(args)
