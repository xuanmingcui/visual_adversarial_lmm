import json
import os
from tqdm import tqdm

with open('datasets/coco/coco_2014val.txt', 'r') as f:
    lines = f.readlines()
    images = []
    for line in tqdm(lines, total=len(lines), position=0, leave=True):
            path, label = line.split('|')
            path = os.path.basename(path)
            images.append(path)
    images_gallery = set(images)
            
def filter_questions():
    with open('datasets/vqa/vqav2/v2_OpenEnded_mscoco_val2014_questions.json', 'r') as f:
        questions = json.load(f)['questions']

    filtered = []
    with open('datasets/vqa/vqav2/coco2014val_questions.json', 'a') as f:
        for q in tqdm(questions):
            image = "COCO_val2014_" + str(q['image_id']).zfill(12) + '.jpg'
            if image in images_gallery:
                # cur_dict = {"question_id": q['question_id'], "image": image, "text": q['question'] + "\nAnswer the question using a single word or phrase."}
                # json.dump(cur_dict, f)
                # f.write('\n')
                filtered.append(q)
        json.dump({"questions": filtered}, f)

def filter_annotations():
    with open('datasets/vqa/vqav2/v2_mscoco_val2014_annotations.json', 'r') as f:
        annos = json.load(f)['annotations']

    filtered = []
    for anno in tqdm(annos):
        image = "COCO_val2014_" + str(anno['image_id']).zfill(12) + '.jpg'
        if image in images_gallery:
            filtered.append(anno)

    result = {"annotations": filtered}
    with open('datasets/vqa/vqav2/coco2014val_annotations.json', 'w') as f:
        json.dump(result, f)


def questions_to_llava_format(path, save_path):
    with open(path, 'r') as f:
        questions = json.load(f)['questions']

    with open(save_path, 'w') as f:
        for q in tqdm(questions):
            image = "COCO_val2014_" + str(q['image_id']).zfill(12) + '.jpg'
            cur = {"question_id": q['question_id'], "image": image, "text": q['question'] + "\nAnswer the question using a single word or phrase."}
            f.write(json.dumps(cur))
            f.write('\n')

if __name__ == '__main__':
    questions_to_llava_format("datasets/vqa/vqav2/coco2014val_questions.json", "datasets/vqa/vqav2/coco2014val_questions_llava.jsonl")
