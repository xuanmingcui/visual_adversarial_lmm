#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --time=4:00:00
#SBATCH --job-name=mmellava

dataset="mme"
export log_file="results/logs/${dataset}/gen_caption_llava1.5_$SLURM_JOB_ID.log"

exec &> $log_file

module load cuda/cuda-12.1
echo "Starting time: $(date)" 
### Activate conda enviroment
source activate ~/anaconda3/envs/adv_env
export SLURM_JOB_ID=$SLURM_JOB_ID

rm "./slurm-${SLURM_JOB_ID}.out"

# Run python
python scripts/gen_llava_caption.py \
    --model-path liuhaotian/llava-v1.5-13b \
    --image-folder datasets/mme/MME_Benchmark_release_version \
    --data-file datasets/mme/llava_mme.jsonl \
    --answers-file datasets/${dataset}/llava_captions.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1_1 \
    --prompt "What is this image about?\nAnswer in one sentence."

echo "Ending time: $(date)" 




