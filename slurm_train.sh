#!/bin/bash
#SBATCH --job-name=NeurWP
#SBATCH --partition=normal
#SBATCH --account=s83
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCH --time=23:59:00
#SBATCH --output=logs/neurwp.out
#SBATCH --error=logs/neurwp.err

# set -x

# Load necessary modules
source /scratch/e1000/meteoswiss/scratch/sadamov/mambaforge/etc/profile.d/conda.sh
conda activate neural-lam

# Set environment variables for DDP
export MASTER_PORT=12355
# export MASTER_ADDR=$(srun hostname -i | head -n 1)

# Set OMP_NUM_THREADS to a value greater than 1
export OMP_NUM_THREADS=4

# Run the script with torchrun
torchrun --nproc_per_node=4 train_model.py
