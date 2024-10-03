#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --time=1:00:00
#SBATCH --job-name=blipvqa


module load cuda/cuda-12.1
source activate ~/anaconda3/envs/env_name
export PYTHONPATH=$PYTHONPATH:~/projects/LLaVA
export SLURM_JOB_ID=$SLURM_JOB_ID

rm "./slurm-${SLURM_JOB_ID}.out"

######## params to change #########
attack_name="CW"
dataset_id="289444"
params="strong"
model_path="blip2_vicuna_instruct" # blip2_opt, blip2_vicuna_instruct, blip2_t5
model_type="vicuna13b" # pretrain_opt6.7b, vicuna13b, pretrain_flant5xxl
##### end of params to change #####

result_folder="results/responses/vqav2/orig/${model_path}_${model_type}"
result_file="orig_${model_path}_${model_type}_ID:${SLURM_JOB_ID}.json"
# image_folder="adv_datasets/coco/classification/${attack_name}/blip2_attack_params:${params}_${dataset_id}"
image_folder="datasets/coco/val2014"
export log_file="results/logs/vqav2/vqa/orig_${model_path}_${model_type}_ID:${SLURM_JOB_ID}.log"

exec &> $log_file

echo "Starting time: $(date)" 

# Run python
python data/vqav2/generate_vqa_answer_blip.py \
    --model-path "$model_path" \
    --model-type "$model_type" \
    --image-folder "$image_folder" \
    --result-file "${result_folder}/${result_file}" \
    --image_ext jpg \

sleep 5

echo "==> Evaluation start below:"

python data/vqav2/eval_vqav2_answers.py \
    --resFile "${result_folder}/${result_file}" \
    --resultFolder $result_folder

echo "Ending time: $(date)" 

