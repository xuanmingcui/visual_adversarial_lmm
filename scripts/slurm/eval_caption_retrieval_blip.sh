#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=blipcap


set -e
declare -A model_paths

# Populate the associative array with your variables
model_paths=(
    [blip2]="Salesforce/blip2-opt-2.7b"
    [blip2cl]="blip2_feature_extractor"
    [llava]="ckpts/LLAMA-on-LLaVA"
    [llava2]="liuhaotian/llava-v1.5-13b"
    [clip]="openai/clip-vit-large-patch14"
    [clip336]="openai/clip-vit-large-patch14-336"
    [orig_coco]="datasets/coco/val2014"
    [instructblip]="blip2_vicuna_instruct"
)

module load cuda/cuda-12.1
source ~/.bashrc
conda activate ~/anaconda3/envs/adv_env
rm "./slurm-${SLURM_JOB_ID}.out"

#### Change the following variables ####
batch_size=1
num_workers=4
# params="moderate"
#### End of changeable variables ########
# "blip2_t5,pretrain_flant5xxl" "blip2_vicuna_instruct,vicuna13b" "blip2_feature_extractor,pretrain"

for params in "normal" "strong"; do
    for model in "blip2_feature_extractor,pretrain" "blip2_t5,pretrain_flant5xxl" "blip2_vicuna_instruct,vicuna13b"; do
        IFS="," read -r model_path model_type <<< "$model"
        for attack in "APGD,289128" "CW,289128" "PGD,289128"; do
            IFS="," read -r attack_name dataset_id <<< "$attack"
            # image_folder="datasets/coco/val2014"

            image_folder="adv_datasets/coco/retrieval_mean/${attack_name}/blip2_attack_params:${params}_${dataset_id}"
            export log_file="results/logs/coco/caption_retrieval_mean/${attack_name}/adv${dataset_id}_${params}_${model_path}_${model_type}_ID:${SLURM_JOB_ID}.log"
            exec &> $log_file

            echo "Starting time: $(date)" 

            ### Run python
                python scripts/evaluate_caption_retrieval.py \
                --model-path $model_path \
                --model-type $model_type \
                --data-file datasets/coco/coco_2014val_caption.json \
                --image_ext pt \
                --temperature 0 \
                --image-folder $image_folder \
                --query "Question: what is this image about? Short answer: " \
                --save_response "results/responses/coco_caption/${attack_name}/adv${dataset_id}_${params}_${model_path}_${model_type}_ID:${SLURM_JOB_ID}.jsonl" \
                --num_beams 5
            echo "Ending time: $(date)" 
        done
    done
done
