#!/bin/bash
#SBATCH --job-name=pl_ddp_test
#SBATCH --output=logs/pl_ddp_test.out
#SBATCH --error=logs/pl_ddp_test.err
#SBATCH --nodes=2
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:2

# Load necessary modules
source /scratch/e1000/meteoswiss/scratch/sadamov/mambaforge/etc/profile.d/conda.sh
conda activate neural-lam

export MASTER_PORT=12355

export OMP_NUM_THREADS=4

# Run the script
srun torchrun --nproc_per_node=2 test_ddp.py
