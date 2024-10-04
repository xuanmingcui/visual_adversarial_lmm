import os
import json

with open('datasets/textvqa/llava_textvqa_val_v051_ocr.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

with open('datasets/textvqa/blip_textvqa_val_v051_ocr.jsonl', 'w') as f:
    for d in data:
        question, ocr = d['text'].split('\nReference OCR token: ')
        text = 'OCR tokens: {}. Question: {} Short answer: '.format(ocr, question)
        f.write(json.dumps({'question_id': d['question_id'], 'image': d['image'], 'text': text}) + '\n')