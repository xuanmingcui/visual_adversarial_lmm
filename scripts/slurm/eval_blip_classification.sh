#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=1-0
#SBATCH --job-name=evalblip

set -e
module load cuda/cuda-12.1
source ~/.bashrc
conda activate ~/anaconda3/envs/env_name
export PYTHONPATH=$PYTHONPATH:~/projects/LLaVA
export SLURM_JOB_ID=$SLURM_JOB_ID

rm "./slurm-${SLURM_JOB_ID}.out"

# params='moderate'
#### Change the following variables ####
dataset="stanford_cars"
task="classification"
# model_path="blip2_vicuna_instruct" # blip2_opt, blip2_vicuna_instruct, blip2_t5
# model_type="vicuna13b" # pretrain_opt6.7b, vicuna13b, pretrain_flant5xxl
batch_size=1
num_workers=4
#### End of changeable variables ########

#stanford cars prompts:
# BLIP2 T5: Question: what is th make, model, body-style and year of this car? Answer: 

for params in "normal" "strong"; do
    for model in "blip2_vicuna_instruct,vicuna13b"; do
        IFS="," read -r model_path model_type <<< "$model"
        for attack in "APGD,306090" "PGD,306090" "CW,306090"; do
            IFS="," read -r attack_name dataset_id <<< "$attack"
            mkdir -p adv_llava/results/logs/$dataset/$task
            export log_file="adv_llava/results/logs/${dataset}/${task}/${attack_name}/adv${dataset_id}_${params}_${model_path}_${model_type}_id:${SLURM_JOB_ID}.log"
            # export log_file="results/logs/${dataset}/${task}/orig_${model_path}_${model_type}_id:${SLURM_JOB_ID}.log"
            exec &> $log_file
            echo "Starting time: $(date)" 
            # Run python
            python scripts/generate_or_evaluate_adversarials.py \
                --model-path ${model_path} \
                --dataset "$dataset" \
                --model-type ${model_type} \
                --image_ext 'pt' \
                --task $task \
                --attack_name 'None' \
                --attack_params 'None' \
                --image-folder "adv_datasets/${dataset}/${task}/${attack_name}/blip2_attack_params:${params}_${dataset_id}" \
                --query "Question: what is the make, model, body-style and year of this car? Answer: " \
                --batch_size $batch_size \
                --num_workers $num_workers

            echo "Ending time: $(date)" 
        done
    done
done
