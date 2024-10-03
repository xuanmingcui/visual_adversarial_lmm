#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --time=1-0

module load cuda/cuda-12.1
### Activate conda enviroment
source activate ~/anaconda3/envs/env_name
export PYTHONPATH=$PYTHONPATH:~/projects/LLaVA
export SLURM_JOB_ID=$SLURM_JOB_ID

rm "./slurm-${SLURM_JOB_ID}.out"

params="strong"


for attack in "CW,289465"; do
    IFS="," read -r attack_name dataset_id <<< "$attack"
    image_folder="adv_datasets/sqa/retrieval/${attack_name}/clip336_attack_params:${params}_${dataset_id}"
    export log_file="results/logs/sqa/vqa/${attack_name}/adv${dataset_id}_${params}_llava1.5_13b_${SLURM_JOB_ID}.log"
    exec &> $log_file
    echo "Starting time: $(date)" 

    ~/anaconda3/envs/adv_env/bin/python3.9 -m sqa.generate_answers_sqa_llava \
        --model-path liuhaotian/llava-v1.5-13b \
        --question-file datasets/vqa/scienceqa/llava_test_CQM-A.json \
        --image-folder ${image_folder} \
        --answers-file results/responses/sqa/${attack_name}/adv${dataset_id}_${params}_llava-v1.5-13b_id:${SLURM_JOB_ID}.jsonl \
        --image_ext pt \
        --single-pred-prompt \
        --temperature 0 \
        --conv-mode vicuna_v1_1

    ~/anaconda3/envs/adv_env/bin/python3.9 ~/projects/LLaVA/sqa/evaluate_sqa_llava2.py \
        --base-dir datasets/vqa/scienceqa \
        --result-file results/responses/sqa/${attack_name}/adv${dataset_id}_${params}_llava-v1.5-13b_id:${SLURM_JOB_ID}.jsonl \
        --output-file results/responses/sqa/${attack_name}/adv${dataset_id}_${params}_llava-v1.5-13b_id:${SLURM_JOB_ID}_output.jsonl \
        --output-result results/responses/sqa/${attack_name}/adv${dataset_id}_${params}_llava-v1.5-13b_id:${SLURM_JOB_ID}_result.json

    echo "Ending time: $(date)" 
done



