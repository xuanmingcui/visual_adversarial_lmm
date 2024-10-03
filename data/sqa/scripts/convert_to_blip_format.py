import json
import os

with open('datasets/scienceqa/problems.json', 'r') as f:
    data = json.load(f)

with open('datasets/scienceqa/scienceqa_blip_format.jsonl', 'w') as f:
    for k, v in data.items():
        if not v['image'] or v['split'] != 'test':
            continue
        has_context = False
        text = f"Question: {v['question']} "
        if v['hint'] != "":
            text += f"Context: {v['hint']} "
            has_context = True
        choices = ""
        choices_letter = [ord('A') + i for i in range(len(v['choices']))]
        for i, c in zip(choices_letter, v['choices']):
            if i < len(v['choices']) - 1:
                choices += "{}. {} ".format(i, c)
            else:
                choices += "{}. {}".format(i, c)
        text += "Options: {}. Answer: ".format(choices)
        f.write(json.dumps({'id': k, 'image': os.path.join(k, v['image']), 'has_context': has_context, 'text': text}) + '\n')
        
