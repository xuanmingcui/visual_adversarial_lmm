#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --job-name=llavacap

set -e
declare -A model_paths

# Populate the associative array with your variables
model_paths=(
    [blip2]="Salesforce/blip2-opt-2.7b"
    [blip2itm]="blip2_image_text_matching"
    [llava]="ckpts/LLAMA-on-LLaVA"
    [llava1.5_13b]="liuhaotian/llava-v1.5-13b"
    [clip]="openai/clip-vit-large-patch14"
    [clip336]="openai/clip-vit-large-patch14-336"
    [orig_coco]="datasets/coco/val2014"
    [instructblip]="blip2_vicuna_instruct"
)

module load cuda/cuda-12.1
source ~/.bashrc
conda activate ~/anaconda3/envs/adv_env
export PYTHONPATH=$PYTHONPATH:~/projects/LLaVA
rm "./slurm-${SLURM_JOB_ID}.out"

#### Change the following variables ####
batch_size=1
params='strong'
num_workers=4
#### End of changeable variables ########
orig_coco="datasets/coco/val2014"
model_name="llava1.5_13b"
for attack in "PGD,289152" ; do
    IFS="," read -r attack_name dataset_id <<< "$attack"
    export log_file="results/logs/coco/caption_retrieval_mean/${attack_name}/adv${dataset_id}_${params}_${model_name}_id:${SLURM_JOB_ID}.log"
    exec &> $log_file
    image_folder="adv_datasets/coco/retrieval_mean/${attack_name}/clip336_attack_params:${params}_${dataset_id}"
    echo "Starting time: $(date)" 
    ### Run python
    python scripts/evaluate_caption_retrieval.py \
    --model-path ${model_paths[${model_name}]} \
    --model-type vicuna13b \
    --data-file datasets/coco/coco_2014val_caption.json \
    --image_ext pt \
    --temperature 0 \
    --image-folder $image_folder \
    --query "describe this image in a short sentence." \
    --save_response "results/responses/coco_caption/${attack_name}/adv${dataset_id}_${params}_${model_name}_id:${SLURM_JOB_ID}.jsonl" \
    --num_beams 1
    echo "Ending time: $(date)" 
done
