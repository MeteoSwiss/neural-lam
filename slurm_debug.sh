#!/bin/bash -l
#SBATCH --job-name=NeurWPd
#SBATCH --output=lightning_logs/neurwp_debug_out.log
#SBATCH --error=lightning_logs/neurwp_debug_err.log
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=02:00:00
#SBATCH --partition=normal
#SBATCH --account=s83
#SBATCH --mem=490G

# Load necessary modules
conda activate neural-ddp

export OMP_NUM_THREADS=16

# Run the script with torchrun
srun -ul --gpus-per-task=1 python train_model.py \
    --dataset "cosmo" --subset_ds 1 --n_workers 8 --batch_size 3 --model "graph_lam" \
    --epochs 1 --val_interval 1
