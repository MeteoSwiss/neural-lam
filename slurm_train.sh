#!/bin/bash
#SBATCH --job-name=NeurWP
#SBATCH --partition=a100-80gb
#SBATCH --account=s83
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --output=lightning_logs/neurwp.out
#SBATCH --error=lightning_logs/neurwp.err
#SBATCH --exclusive
#SBATCH --mem=490G

# Load necessary modules
source ${SCRATCH}/miniforge3/etc/profile.d/conda.sh
conda activate neural-ddp

# Set OMP_NUM_THREADS to a value greater than 1
export OMP_NUM_THREADS=16

NUM_GPUS=$(echo $SLURM_JOB_GPUS | tr ',' '\n' | wc -l)

# Run the script with torchrun
srun -u torchrun --nnodes=$SLURM_NNODES --nproc_per_node=$NUM_GPUS train_model.py \
    --dataset "cosmo" --subset_ds 0 --val_interval 100 --epochs 200 --n_workers 8 \
    --batch_size 8 --model "graph_lam" \
    --load "saved_models/graph_lam-4x64-11_02_22_55_35/min_val_loss.ckpt"
