#!/bin/bash
#SBATCH --job-name=NeurWP
#SBATCH --partition=normal
#SBATCH --account=s83
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=23:59:00
#SBATCH --output=logs/neurwp_single.out
#SBATCH --error=logs/neurwp_single.err
#SBATCH --verbose

# Load necessary modules
source /scratch/e1000/meteoswiss/scratch/sadamov/mambaforge/etc/profile.d/conda.sh
conda activate neural-lam

# Set OMP_NUM_THREADS to a value greater than 1
export OMP_NUM_THREADS=1

# Run the script with torchrun
torchrun --nproc_per_node=4 train_model.py
