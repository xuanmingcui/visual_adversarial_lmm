#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --time=1-0
#SBATCH --job-name=llava_gen

######## params to change #########
attack_name="CW"
dataset_id="289354"
params="strong"
image_folder="adv_datasets/coco/classification/CW/clip336_attack_params:default_289354"
##### end of params to change #####

result_folder="results/responses/vqav2/${attack_name}/${params}_llava1.5"
result_file="datasetID:${dataset_id}_llava-v1.5-13b.json"
llava="ckpts/LLAMA-on-LLaVA"
llavav2="liuhaotian/llava-v1.5-13b"

export log_file="results/logs/vqav2/vqa/${attack_name}/adv${dataset_id}_${params}_llava1.5_13b_response_$SLURM_JOB_ID.log"
exec &> $log_file

module load cuda/cuda-12.1
echo "Starting time: $(date)" 
### Activate conda enviroment
source activate ~/anaconda3/envs/adv_env
export PYTHONPATH=$PYTHONPATH:~/projects/LLaVA
export SLURM_JOB_ID=$SLURM_JOB_ID

rm "./slurm-${SLURM_JOB_ID}.out"

# Run python
python data/vqav2/generate_answers_llavav2.py \
    --model-path "$llavav2" \
    --image-folder "$image_folder" \
    --result-file "${result_folder}/${result_file}" \
    --question-file "data/vqav2/coco2014val_questions.jsonl" \
    --temperature 0 \
    --image_ext pt \
    --conv-mode vicuna_v1_1

sleep 5

echo "==> Evaluation start below:"

python data/vqav2/eval_vqav2_answers.py \
    --resFile "${result_folder}/${result_file}" \
    --resultFolder $result_folder

echo "Ending time: $(date)" 

