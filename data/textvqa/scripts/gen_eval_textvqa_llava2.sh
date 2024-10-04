#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=h100
#SBATCH --time=8:00:00
#SBATCH --job-name=tvqa_llava

attack_name="CW"
dataset_id="289464"
params="default"
image_folder="/groups/sernam/adv_llava/adv_datasets/textvqa/retrieval/${attack_name}/clip336_attack_params:${params}_${dataset_id}"

llavav2="liuhaotian/llava-v1.5-13b"
shared_folder="/groups/sernam"
export log_file="/${shared_folder}/adv_llava/results/logs/textvqa/vqa/${attack_name}/adv${dataset_id}_${params}_llava1.5_13b_response_$SLURM_JOB_ID.log"
# export log_file="/${shared_folder}/adv_llava/results/logs/textvqa/vqa/orig_llava1.5_13b_response_$SLURM_JOB_ID.log"
exec &> $log_file

module load cuda/cuda-12.1
echo "Starting time: $(date)" 
### Activate conda enviroment
source activate ~/anaconda3/envs/adv_env
export PYTHONPATH=$PYTHONPATH:~/projects/LLaVA
export SLURM_JOB_ID=$SLURM_JOB_ID

rm "./slurm-${SLURM_JOB_ID}.out"

# Run python
~/anaconda3/envs/adv_env/bin/python3.9 -m textvqa.generate_answers_llava2 \
    --model-path $llavav2 \
    --question-file /groups/sernam/datasets/vqa/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder $image_folder \
    --answers-file /groups/sernam/adv_llava/results/responses/textvqa/${attack_name}_${params}/datasetID:${dataset_id}_llava-v1.5-13b.jsonl \
    --temperature 0 \
    --image_ext pt \
    --conv-mode vicuna_v1_1

~/anaconda3/envs/adv_env/bin/python3.9 ~/projects/LLaVA/textvqa/eval_textvqa.py \
    --annotation-file /groups/sernam/datasets/vqa/textvqa/TextVQA_0.5.1_val.json \
    --result-file /groups/sernam/adv_llava/results/responses/textvqa/${attack_name}_${params}/datasetID:${dataset_id}_llava-v1.5-13b.jsonl

echo "Ending time: $(date)" 




