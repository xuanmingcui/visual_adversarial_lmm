
import torch
from PIL import Image

from lavis.models import load_model_and_preprocess
from lavis.datasets.builders import dataset_zoo
dataset_names = dataset_zoo.get_names()
print(dataset_names)
# raw_image = Image.open("/groups/sernam/datasets/coco/val2014/COCO_val2014_000000000073.jpg").convert("RGB")
# device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
# text='A photo of motocycle'
# model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "pretrain", device=device, is_eval=True)
# img = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
# txt = text_processors["eval"](text)
# itm_output = model({"image": img, "text_input": [txt]*5}, match_head="itc")
# itm_scores = torch.nn.functional.softmax(itm_output, dim=1)

# print(f'The image and text are matched with a probability of {itm_scores[:, 1].item():.3%}')
