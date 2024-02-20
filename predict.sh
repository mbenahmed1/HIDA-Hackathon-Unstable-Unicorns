#!/bin/bash

#SBATCH --job-name=prediction
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:full:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=76
#SBATCH --time=01:00:00

export CUDA_CACHE_DISABLE=1
export OMP_NUM_THREADS=76

data_workspace=/hkfs/work/workspace_haic/scratch/qx6387-hida-hackathon-data
group_workspace=/hkfs/work/workspace_haic/scratch/qx6387-<groupd_id>

module purge
module load toolkit/nvidia-hpc-sdk/23.9

source ${group_workspace}/hida_venv/bin/activate
srun python ${group_workspace}/HIDA-Hackathon/predict.py ${data_workspace}/train
