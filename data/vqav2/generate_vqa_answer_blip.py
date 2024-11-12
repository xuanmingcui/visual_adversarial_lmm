import argparse
import torch
import os
import json
from tqdm import tqdm
import numpy as np
from PIL import Image
from models.LAVIS.lavis.models import load_model_and_preprocess
from m4c_evaluator import EvalAIAnswerProcessor
from utils.helpers import get_model_name_from_path

def eval_model(args):

    model, vis_processors, text_processors = load_model_and_preprocess(name=args.model_path, model_type=args.model_type, is_eval=True, device='cuda')
    model_name = get_model_name_from_path(args.model_path)
    with open(args.question_file) as f:
        questions = [json.loads(q) for q in f]
    paths = args.image_folder.split('/')
    os.makedirs(os.path.dirname(args.result_file), exist_ok=True)
    answer_processor = EvalAIAnswerProcessor()
    results = []
    for line in tqdm(questions, total=len(questions)):
        file_path = os.path.join(args.image_folder, line['image'])
        if args.image_ext == 'pt':
            file_path = os.path.splitext(file_path)[0] + '.pt'
            line['image'] = torch.load(file_path).unsqueeze(0).cuda().half()
        else:
            line['image'] = vis_processors['eval'](Image.open(file_path).convert('RGB')).unsqueeze(0).cuda().half()
        
        line['text_input'] = args.query_formatter.format(line['text']) 
        line['text_input'] = text_processors['eval'](line['text_input'])

        with torch.inference_mode():
            result = model.predict_answers(line, inference_method="generate")[0]

        results.append({"question_id": line['question_id'], "answer": answer_processor(result)})

    with open(args.result_file, 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-type", type=str, required=True) 
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--result-file", type=str, required=True)
    parser.add_argument("--image_ext", type=str, default="pt")
    parser.add_argument("--question-file", type=str, default="data/vqav2/coco2014val_questions.jsonl")
    parser.add_argument("--query_formatter", type=str, default="Question: {} Short answer:")
    args = parser.parse_args()
    for k,v in vars(args).items():
        print(f"{k}: {v}")
    eval_model(args)
