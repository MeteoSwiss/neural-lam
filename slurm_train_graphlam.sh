#!/bin/bash -l
#SBATCH --job-name=NeurWP
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --partition=a100-80gb
#SBATCH --account=s83
#SBATCH --output=lightning_logs/neurwp_out.log
#SBATCH --error=lightning_logs/neurwp_err.log
#SBATCH --mem=490G

export PREPROCESS=true

# Load necessary modules
conda activate neural-ddp

export OMP_NUM_THREADS=16

if [ "$PREPROCESS" = true ]; then
    srun -ul -N1 -n1 python create_static_features.py --boundaries 60
    srun -ul -N1 -n1 python create_mesh.py --dataset "cosmo"
    srun -ul -N1 -n1 python create_grid_features.py --dataset "cosmo"
    # This takes multiple hours!
    srun -ul -N1 -n1 python create_parameter_weights.py --dataset "cosmo" --batch_size 12 --n_workers 8 --step_length 1
fi

# Run the script with torchrun
srun -ul --gpus-per-task=1 python train_model.py \
    --dataset "cosmo" --val_interval 20 --epochs 100 --n_workers 8 --batch_size 12 \
    --load wandb/run-20231226_083638-4f5sanqa/files/latest-v1.ckpt \
    --resume_opt_sched 1
    # --resume_run '3gio4mcv'
