#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=h100
#SBATCH --time=24:00:00
#SBATCH --job-name=tvqa_blip
#SBATCH --exclude=evc43

module load cuda/cuda-12.1

### Activate conda enviroment
source activate ~/anaconda3/envs/adv_env
export PYTHONPATH=$PYTHONPATH:~/projects/LLaVA
export SLURM_JOB_ID=$SLURM_JOB_ID

# dataset_id="276587"
# attack_name="CW"
# model_path="blip2_t5" # blip2_opt, blip2_vicuna_instruct, blip2_t5
# model_type="pretrain_flant5xxl" # pretrain_opt6.7b, vicuna13b, pretrain_flant5xxl
params="default"

rm "./slurm-${SLURM_JOB_ID}.out"

for model in  "blip2_vicuna_instruct,vicuna13b"; do
    IFS="," read -r model_path model_type <<< "$model"
    for attack in "CW,289527"  ; do
        IFS="," read -r attack_name dataset_id <<< "$attack"
        image_folder="adv_datasets/textvqa/retrieval/${attack_name}/blip2_attack_params:${params}_${dataset_id}"
        export log_file="results/logs/textvqa/vqa/${attack_name}/adv${dataset_id}_${params}_${model_path}_${model_type}_ID:${SLURM_JOB_ID}.log"
        exec &> $log_file
        echo "Starting time: $(date)" 
        # Run python
        ~/anaconda3/envs/adv_env/bin/python3.9 -m textvqa.generate_answers_blip \
            --model-path $model_path \
            --model-type $model_type \
            --question-file data/textvqa/llava_textvqa_val_v051_ocr.jsonl \
            --image-folder $image_folder \
            --result-file results/responses/textvqa/${attack_name}/adv${dataset_id}_${params}_${model_path}_${model_type}_ID:${SLURM_JOB_ID}.jsonl \
            --image_ext pt \
            --query_formatter "Question: {} Short answer: "

        ~/anaconda3/envs/adv_env/bin/python3.9 ~/projects/LLaVA/textvqa/eval_textvqa.py \
            --annotation-file data/textvqa/TextVQA_0.5.1_val.json \
            --result-file results/responses/textvqa/${attack_name}/adv${dataset_id}_${params}_${model_path}_${model_type}_ID:${SLURM_JOB_ID}.jsonl
        echo "Ending time: $(date)" 
    done
done



