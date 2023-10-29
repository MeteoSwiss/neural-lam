#!/bin/bash
#SBATCH --job-name=NeurWP
#SBATCH --partition=normal
#SBATCH --account=s83
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --output=logs/neurwp_eval.out
#SBATCH --error=logs/neurwp_eval.err

# Load necessary modules
source /scratch/e1000/meteoswiss/scratch/sadamov/mambaforge/etc/profile.d/conda.sh
conda activate neural-ddp

# Set OMP_NUM_THREADS to a value greater than 1
export OMP_NUM_THREADS=4

# Run the script with torchrun
srun torchrun train_model.py --load "saved_models/graph_lam-4x64-10_22_03_53_18/min_val_loss.ckpt" --dataset "cosmo" --eval="test" --subset_ds 1 --n_workers 31 --batch_size 4 --model "graph_lam"
