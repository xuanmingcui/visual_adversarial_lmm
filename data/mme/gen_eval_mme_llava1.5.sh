#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=h100
#SBATCH --time=8:00:00
#SBATCH --job-name=mme_llava
#SBATCH --exclude=evc43

module load cuda/cuda-12.1

### change to your env
source activate ~/anaconda3/envs/env_name
export PYTHONPATH=$PYTHONPATH:~/projects/LLaVA
export SLURM_JOB_ID=$SLURM_JOB_ID

rm "./slurm-${SLURM_JOB_ID}.out"

params="strong"

for attack in "APGD,278430"; do
    IFS="," read -r attack_name dataset_id <<< "$attack"
    image_folder="datasets/vqa/mme/MME_Benchmark_release_version"
    result_folder="results/responses/mme/orig_llava1.5_13b_ID:$SLURM_JOB_ID"
    export log_file="results/logs/mme/vqa/orig_llava1.5_13b_ID:$SLURM_JOB_ID.log"
    exec &> $log_file
    echo "Starting time: $(date)" 

    # Run python
    python -m generate_answers_llava2 \
        --model-path liuhaotian/llava-v1.5-13b \
        --question-file datasets/vqa/mme/mme_landmarks_only.jsonl\
        --image-folder $image_folder \
        --result-file ${result_folder}/result.jsonl \
        --image_ext jpg \

    python /home/scui/projects/LLaVA/mme/convert_answer_to_mme.py \
        --result-folder ${result_folder}

    python datasets/vqa/mme/eval_tool/calculation.py \
        --results_dir ${result_folder}

    echo "Ending time: $(date)" 
done
