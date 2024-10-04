from tqdm import tqdm
import json
import torch
import os
import argparse

from torchvision import transforms
from accelerate import Accelerator
from utils.metric import AverageMeter, accuracy
from utils.helpers import get_model_name_from_path, boolean_string, load_yaml_parameters

from models.model_wrapper import get_model
from data.dataloader import get_dataloader
from attacks import *

denormalize = transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225), 
                                   (1/0.229, 1/0.224, 1/0.225))

SLURM_JOB_ID = os.environ.get('SLURM_JOB_ID', None)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

accelerator = Accelerator()

def generate_adversarials_dataset(model, args):

    model = accelerator.prepare(model)
    dataset, dataloader = get_dataloader(args, model)
    dataloader = accelerator.prepare(dataloader)
    model.eval()
    if args.task != 'vqa':
        model.set_encoded_labels(dataset.label_list)

    if args.attack_name != "None":

        attack = eval(args.attack_name)(model, **args.attack_params)
        attack.set_normalization_used(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        print(attack)

    acc1, acc5 = AverageMeter(), AverageMeter()
    for data in tqdm(dataloader):
        contexts = input_ids = text = None
        if  args.task == 'classification' or 'clip' in args.model_path or 'blip2_feature_extractor' in args.model_path:
            images, base_paths, labels = data
            text=args.query
            labels = labels.long()
        elif args.task == 'classification_with_context':
            if 'llava' in args.model_path:
                images, base_paths, labels, input_ids = data
            else:
                images, base_paths, labels, text = data
        elif args.task == 'classification_with_context_multi_qs':
            if 'llava' in args.model_path:
                images, base_paths, labels, text, contexts = data

        elif args.task == 'llm_retrieval_classification_multi_qs':
            if 'llava' in args.model_path:
                images, base_paths, labels, text = data

        elif 'retrieval' in args.task:
            if 'llava' in args.model_path:
                images, base_paths, labels, input_ids = data
            else:
                images, base_paths, labels, text = data

        if args.attack_name != 'CW':
            # CW overflows with fp16
            images = images.half()
            
        if args.attack_name:
            model.attack_mode = True
            images = attack(images, labels=labels)
            if args.save_image:
                detached_adv_images = images.detach().cpu()
                for i in range(len(base_paths)):
                    save_path = os.path.join(args.save_folder, base_paths[i])
                    os.makedirs(os.path.split(save_path)[0], exist_ok=True)
                    torch.save(detached_adv_images[i].clone(), os.path.splitext(save_path)[0] + '.pt')

        with torch.no_grad():
            model.attack_mode = False

            logits = model(images, text=text, query_input_ids=input_ids, contexts=contexts)
            
        _acc1, _acc5 = accuracy(logits, labels, topk=(1, 5))

        acc1.update(_acc1[0].item())
        acc5.update(_acc5[0].item())

    print("** Acc@1: {} Acc@5: {} **".format(acc1.avg, acc5.avg))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, default="debug")
    parser.add_argument("--model-path", type=str, default='openai/clip-vit-large-patch14')
    parser.add_argument("--image-folder", type=str, required=False, default=None, nargs='?')
    parser.add_argument("--data-file", type=str, required=False, default=None, nargs='?')
    parser.add_argument("--context-file", type=str, required=False, default=None, nargs='?')
    parser.add_argument("--image_ext", type=str, default='pt')
    parser.add_argument("--save-folder", type=str, required=False, default=None)
    parser.add_argument("--task", type=str, default="classification", choices=["classification", 
                                                                               "retrieval", 
                                                                               "retrieval_mean", 
                                                                               "classification_with_context", 
                                                                               "classification_with_context_multi_qs", 
                                                                               "llm_retrieval_classification_multi_qs"])
    parser.add_argument("--dataset", type=str, default="coco")
    parser.add_argument("--save_image", action='store_true', default=False)
    parser.add_argument("--batch_size", type=int, required=False, default=1, nargs='?')
    parser.add_argument("--num_workers", type=int, required=False, default=1, nargs='?')
    parser.add_argument("--model-type", type=str, default=None)

    parser.add_argument("--query", type=str, default=None, required=False)
    parser.add_argument("--multi_qs_formatter", type=str, default=None)
    # params for decoder
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1_1")
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)

    # whether to use descriptors for Imagenet label retrieval
    parser.add_argument("--use_descriptors", action="store_true", default=False)

    ## =============== args for adv attack =================
    parser.add_argument("--attack_name", type=str, default=None)
    parser.add_argument("--attack_params", type=str, default='strong')
    # whether to attack descriptors or attack plain labels and use descriptors to do prediction
    parser.add_argument("--attack_descriptors", action="store_true", default=False)
    parser.add_argument("--transfer", action='store_true')

    args = parser.parse_args()
    
    path_to_config = "data/path_config.yaml"
    args.data_file = args.data_file if args.data_file else load_yaml_parameters(path_to_config, args.dataset)[args.task]['data_file']
    args.image_folder = args.image_folder if args.image_folder else load_yaml_parameters(path_to_config, args.dataset)[args.task]['image_folder']

    model = get_model(args).to(accelerator.device)
    model_name = get_model_name_from_path(args.model_path)

    if args.dataset != 'imagenet' and args.use_descriptors:
        raise ValueError("Only imagenet dataset has descriptors")

    if args.save_image:
        if not args.save_folder:
            args.save_folder = "adv_datasets/{}/{}/{}/{}_attack_params:{}_{}".format(args.dataset, args.task, 
                            args.attack_name, model_name, args.attack_params, args.run_id)
                        
        assert not os.path.exists(args.save_folder), f"Folder {args.save_folder} already exists. It's likely the targeted adversarial dataset is already created."
        os.makedirs(args.save_folder)

    if args.attack_name:
        if args.attack_params in ['strong', 'normal']:
            args.attack_params = load_yaml_parameters('attacks/config.yaml', args.attack_name)[args.attack_params]
        else:
            args.attack_params = eval(args.attack_params)

    print(json.dumps(vars(args), indent=2))

    generate_adversarials_dataset(model, args)




    