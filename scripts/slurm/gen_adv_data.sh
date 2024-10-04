#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=1-0
#SBATCH --job-name=gen_adv

set -e
module load cuda/cuda-12.1
source ~/.bashrc
conda activate ~/anaconda3/envs/adv_env
export PYTHONPATH=$PYTHONPATH:~/projects/LLaVA
export SLURM_JOB_ID=$SLURM_JOB_ID

rm "./slurm-${SLURM_JOB_ID}.out"

declare -A model_paths

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

#### Change the following variables ####
dataset="stanford_cars"
task="classification"
# attack_name="APGD"
# attack_params="strong"
model_name="blip2cl"
batch_size=8
num_workers=2
#### End of changeable variables ########

for attack_params in "default" "moderate"; do
    for attack_name in "APGD" "CW" "PGD"; do
        mkdir -p results/logs/$dataset/$task/$attack_name
        export log_file="results/logs/${dataset}/${task}/${attack_name}/gen_adv_${model_name}_param:${attack_params}_id:${SLURM_JOB_ID}.log"
        exec &> $log_file
        echo "Starting time: $(date)" 

        python scripts/generate_or_evaluate_adversarials.py \
            --model-path ${model_paths[${model_name}]} \
            --dataset "$dataset" \
            --model-type pretrain \
            --save_image  \
            --image_ext 'jpg' \
            --task $task \
            --attack_name $attack_name \
            --attack_params $attack_params \
            --batch_size $batch_size \
            --num_workers $num_workers \

        echo "Ending time: $(date)" 
    done
done

