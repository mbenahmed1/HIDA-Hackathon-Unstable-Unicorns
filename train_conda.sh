#!/bin/bash


#SBATCH -o train_%j_noRGB_RCNN.txt
#SBATCH -e train_%j_noRGB_RCNN.txt
#SBATCH --job-name=training
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:full:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=76
#SBATCH --time=08:00:00
#SBATCH --reservation=hida
#SBATCH --chdir=/home/hgf_hmgu/hgf_jpp4037/HIDA-Hackathon-Unstable-Unicorns/


export CUDA_CACHE_DISABLE=1
export OMP_NUM_THREADS=76


# module purge
# module load toolkit/nvidia-hpc-sdk/23.9
source $HOME/.bashrc
conda activate unicorn


python train.py
