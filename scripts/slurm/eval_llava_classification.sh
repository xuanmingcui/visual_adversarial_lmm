#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=1-0
#SBATCH --job-name=llavacls

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
export PYTHONPATH=$PYTHONPATH:~/projects/LLaVA
export SLURM_JOB_ID=400000

# rm "./slurm-${SLURM_JOB_ID}.out"


# --query "What is the type of food in the image?\nAnswer in a few word or phrase." \

#### Change the following variables ####
dataset="stanford_cars" # imagenet food101 coco cub
task="classification" # retrieval_mean classification_with_context_multi_qs classification
model_name="llava2"
batch_size=1
num_workers=2
#### End of changeable variables ########

# "What is the type of the food in this image?\nAnswer the question using a single word or phrase."
        # --query "describe this image in a short sentence." \

# params='strong'
        # --query "What is the main object in the image?"

for params in  "strong" ; do
    for attack in "PGD,306089"; do
        IFS="," read -r attack_name dataset_id <<< "$attack"
        mkdir -p results/logs/$dataset/$task
        export log_file="results/logs/${dataset}/${task}/${attack_name}/adv${dataset_id}_${params}_${model_name}_id:${SLURM_JOB_ID}.log"
        exec &> $log_file
        echo "Starting time: $(date)" 
        # Run python
        python scripts/generate_or_evaluate_adversarials.py \
            --model-path ${model_paths[${model_name}]} \
            --dataset "$dataset" \
            --model-type pretrain \
            --image_ext 'pt' \
            --task $task \
            --attack_name 'None' \
            --attack_params 'None' \
            --query "Describe the car in the format: make model body-style year" \
            --image-folder "adv_datasets/${dataset}/classification/${attack_name}/clip336_attack_params:${params}_${dataset_id}" \
            --batch_size $batch_size \
            --num_workers $num_workers 

        echo "Ending time: $(date)" 
    done
done

