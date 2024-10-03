#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00

module load cuda/cuda-12.1
source activate ~/anaconda3/envs/adv_env
export PYTHONPATH=$PYTHONPATH:~/projects/LLaVA
export SLURM_JOB_ID=$SLURM_JOB_ID

rm "./slurm-${SLURM_JOB_ID}.out"


params="strong"


for attack in "APGD,296842" "CW,296842" "PGD,296842"; do
    IFS="," read -r attack_name dataset_id <<< "$attack"
    export log_file="results/logs/pope/vqa_cls/${attack_name}/${dataset_id}_${params}_llava1.5_13b_with_context_$SLURM_JOB_ID.log"
    exec &> $log_file
    echo "Starting time: $(date)" 
    # image_folder="adv_datasets/pope/retrieval/${attack_name}/clip336_attack_params:${params}_${dataset_id}"
    image_folder="datasets/pope/classification/${attack_name}/clip336_attack_params:${params}_${dataset_id}"
    ~/anaconda3/envs/adv_env/bin/python3.9 -m generate_answers_llava2 \
        --model-path liuhaotian/llava-v1.5-13b \
        --question-file datasets/vqa/pope/pope_test_with_context.jsonl \
        --image-folder ${image_folder} \
        --result-file results/responses/pope/vqa_cls/${attack_name}/${dataset_id}_${params}_llava-v1.5-13b_with_context_${SLURM_JOB_ID}.jsonl \
        --image_ext pt \
        --temperature 0 \
        --conv-mode vicuna_v1_1

    python data/pope/eval_pope.py \
        --annotation-dir datasets/vqa/pope/annotations \
        --question-file datasets/vqa/pope/pope_test_with_context.jsonl \
        --result-file results/responses/pope/vqa_cls/${attack_name}/${dataset_id}_${params}_llava-v1.5-13b_with_context_${SLURM_JOB_ID}.jsonl

    echo "Ending time: $(date)" 
done

