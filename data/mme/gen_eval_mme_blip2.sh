#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --job-name=mme_blip

module load cuda/cuda-12.1

### change to your env
source activate ~/anaconda3/envs/env_name
export PYTHONPATH=$PYTHONPATH:~/projects/LLaVA
export SLURM_JOB_ID=$SLURM_JOB_ID

rm "./slurm-${SLURM_JOB_ID}.out"

params="strong"

# model_path="blip2_t5" # blip2_opt, blip2_vicuna_instruct, blip2_t5
# model_type="pretrain_flant5xxl" # pretrain_opt6.7b, vicuna13b, pretrain_flant5xxl

for model in "blip2_vicuna_instruct,vicuna13b" "blip2_t5,pretrain_flant5xxl"; do
    IFS="," read -r model_path model_type <<< "$model"
    for attack in "APGD,288382"; do
        IFS="," read -r attack_name dataset_id <<< "$attack"
        result_folder="results/responses/mme/orig_${model_path}_${model_type}_ID:$SLURM_JOB_ID"
        image_folder="datasets/vqa/mme/MME_Benchmark_release_version"
        export log_file="results/logs/mme/vqa/orig_${model_path}_${model_type}_ID:$SLURM_JOB_ID.log"
        exec &> $log_file
        echo "Starting time: $(date)" 

        # Run python
        ~/anaconda3/envs/adv_env/bin/python3.9 -m generate_answers_blip2 \
            --model-path $model_path \
            --model-type $model_type \
            --question-file datasets/vqa/mme/mme_landmarks_only.jsonl\
            --image-folder $image_folder \
            --result-file ${result_folder}/result.jsonl \
            --image_ext jpg

        ~/anaconda3/envs/adv_env/bin/python3.9 data/mme/convert_answer_to_mme.py \
            --result-folder ${result_folder}

        ~/anaconda3/envs/adv_env/bin/python3.9 datasets/vqa/mme/eval_tool/calculation.py \
            --results_dir ${result_folder}

        echo "Ending time: $(date)" 
    done
done