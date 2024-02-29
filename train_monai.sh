#!/bin/bash

#SBATCH -o slurm_logs/logs_%A_%a.out
#SBATCH -e slurm_logs/logs_%A_%a.out
#SBATCH --job-name=effunet
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:full:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=76
#SBATCH --time=08:00:00
#SBATCH --reservation=hida
#SBATCH --chdir=/home/hgf_hmgu/hgf_jpp4037/HIDA-Hackathon-Unstable-Unicorns/
#SBATCH --array=1-3

export CUDA_CACHE_DISABLE=1
export OMP_NUM_THREADS=76


# module purge
# module load toolkit/nvidia-hpc-sdk/23.9
source $HOME/.bashrc
conda activate unicorn

LINE_NUM=$((SLURM_ARRAY_TASK_ID + 1))

# Read parameters from CSV file
IFS=',' read -r model augmentation in_channels batch_size  <<< "$(sed -n "${LINE_NUM}p" combinations_train_monai_2.csv)"

# Optional: Echo parameters for debugging
echo "Running job array index: $SLURM_ARRAY_TASK_ID"
echo "Parameters: model=$model, augmentation=$augmentation, in_channels=$in_channels, batch_size = $batch_size"

# Construct and execute the command
COMMAND="python -u train_monai.py --model=$model --in_channels=$in_channels --batch=$batch_size"

if [ "$augmentation" == "True" ]; then
    COMMAND="$COMMAND --augmentation"
fi
echo "Executing command: $COMMAND"
eval $COMMAND