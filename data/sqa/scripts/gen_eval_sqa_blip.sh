#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --time=1-0
#SBATCH --job-name=sqa_blip2

module load cuda/cuda-12.1

### Activate conda enviroment
source activate ~/anaconda3/envs/adv_env
export PYTHONPATH=$PYTHONPATH:~/projects/LLaVA
export SLURM_JOB_ID=$SLURM_JOB_ID

# dataset_id="276587"
# attack_name="CW"
# model_path="blip2_t5" # blip2_opt, blip2_vicuna_instruct, blip2_t5
# model_type="pretrain_flant5xxl" # pretrain_opt6.7b, vicuna13b, pretrain_flant5xxl
# image_folder="datasets/vqa/scienceqa/test"
params="strong"

rm "./slurm-${SLURM_JOB_ID}.out"
 
for model in "blip2_t5,pretrain_flant5xxl" "blip2_vicuna_instruct,vicuna13b"; do
    IFS="," read -r model_path model_type <<< "$model"
    for attack in "CW,290891"; do
        IFS="," read -r attack_name dataset_id <<< "$attack"
        image_folder="adv_datasets/sqa/retrieval/${attack_name}/blip2_attack_params:${params}_${dataset_id}"
        export log_file="results/logs/sqa/vqa/${attack_name}/adv${dataset_id}_${params}_${model_path}_${model_type}_ID:${SLURM_JOB_ID}.log"
        exec &> $log_file
        echo "Starting time: $(date)" 
        python -m data.sqa.generate_answers_sqa_blip \
            --model-path $model_path \
            --model-type $model_type \
            --question-file datasets/vqa/scienceqa/llava_test_CQM-A.json \
            --image-folder ${image_folder} \
            --answers-file results/responses/sqa/${attack_name}/adv${dataset_id}_${params}_${model_path}_${model_type}_ID:${SLURM_JOB_ID}.jsonl \
            --image_ext pt \
            --single-pred-prompt

        python data/sqa/evaluate_sqa_llava2.py \
            --base-dir datasets/vqa/scienceqa \
            --result-file results/responses/sqa/${attack_name}/adv${dataset_id}_${params}_${model_path}_${model_type}_ID:${SLURM_JOB_ID}.jsonl \
            --output-file results/responses/sqa/${attack_name}/adv${dataset_id}_${params}_${model_path}_${model_type}_ID:${SLURM_JOB_ID}_output.jsonl \
            --output-result results/responses/sqa/${attack_name}/adv${dataset_id}_${params}_${model_path}_${model_type}_ID:${SLURM_JOB_ID}_result.json

        echo "Ending time: $(date)" 
    done
done


