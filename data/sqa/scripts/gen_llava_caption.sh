#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --time=4:00:00
#SBATCH --job-name=sqa

module load cuda/cuda-12.1
echo "Starting time: $(date)" 
### Activate conda enviroment
source activate ~/anaconda3/envs/env_name

rm "./slurm-${SLURM_JOB_ID}.out"

# Run python
python3.9 scripts/gen_llava_caption.py \
    --model-path liuhaotian/llava-v1.5-13b \
    --image-folder datasets/vqa/scienceqa \
    --temperature 0 \
    --conv-mode vicuna_v1_1 \
    --prompt "What is this image about?\nAnswer in one sentence."

echo "Ending time: $(date)" 




