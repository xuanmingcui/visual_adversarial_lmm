import os
import json

with open('datasets/vqa/scienceqa/llava_test_CQM-A.json', 'r') as f:
    orig_llava_q = json.load(f)

with open('datasets/vqa/scienceqa/problems.json', 'r') as f:
    orig_scienceqa = json.load(f)

no_context_llava_q = []

for q in orig_llava_q:
    orig_q = orig_scienceqa[q['id']]
    question = orig_q['question']
    choices = ""
    for i, choice in enumerate(orig_q['choices']):
        choices += f"\n{chr(ord('A') + i)}. {choice}"
    if 'image' in q:
        q['conversations'][0]['value'] = '<image>\n'
    else:
        q['conversations'][0]['value'] = ''
    q['conversations'][0]['value'] += question + choices
    no_context_llava_q.append(q)


with open('datasets/vqa/scienceqa/llava_test_CQM-A_no_context.json', 'w') as f:
    json.dump(no_context_llava_q, f, indent=2)