#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=h100
#SBATCH --time=24:00:00
#SBATCH --job-name=pope_blip2
#SBATCH --exclude=evc43

module load cuda/cuda-12.1
### Activate conda enviroment
source activate ~/anaconda3/envs/adv_env
export PYTHONPATH=$PYTHONPATH:~/projects/LLaVA
export SLURM_JOB_ID=$SLURM_JOB_ID

rm "./slurm-${SLURM_JOB_ID}.out"

# dataset_id="277241"
# attack_name="PGD"
params="strong"
# image_folder="datasets/coco/val2014"
# model_path="blip2_t5" # blip2_opt, blip2_vicuna_instruct, blip2_t5
# model_type="pretrain_flant5xxl" # pretrain_opt6.7b, vicuna13b, pretrain_flant5xxl

for model in "blip2_t5,pretrain_flant5xxl" "blip2_vicuna_instruct,vicuna13b"; do
    IFS="," read -r model_path model_type <<< "$model"
    for attack in "APGD,296845" "CW,296845" "PGD,296845"; do
        IFS="," read -r attack_name dataset_id <<< "$attack"
        image_folder="datasets/pope/classification/${attack_name}/blip2_attack_params:${params}_${dataset_id}"
        export log_file="results/logs/pope/vqa_cls/${attack_name}/adv${dataset_id}_${params}_${model_path}_${model_type}_with_context_ID:${SLURM_JOB_ID}.log"
        exec &> $log_file
        echo "Starting time: $(date)" 
        ~/anaconda3/envs/adv_env/bin/python3.9 -m generate_answers_blip2 \
            --model-path ${model_path} \
            --model-type ${model_type} \
            --question-file datasets/vqa/pope/pope_test_with_context.jsonl \
            --image-folder ${image_folder} \
            --result-file results/responses/pope/vqa_cls/${attack_name}/adv${dataset_id}_${params}_${model_path}_${model_type}_with_context_ID:${SLURM_JOB_ID}.jsonl \
            --image_ext pt

        python data/pope/eval_pope.py \
            --annotation-dir datasets/vqa/pope/annotations \
            --question-file datasets/vqa/pope/pope_test_with_context.jsonl \
            --result-file results/responses/pope/vqa_cls/${attack_name}/adv${dataset_id}_${params}_${model_path}_${model_type}_with_context_ID:${SLURM_JOB_ID}.jsonl

        echo "Ending time: $(date)" 
    done
done

