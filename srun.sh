srun -N4 --ntasks-per-node=4 --gres=gpu:1 --job-name=NeurWP --time=23:59:00 --partition=normal --account=s83 torchrun --nnodes=4 --nproc_per_node=4 train_model.py

srun -N1 --ntasks-per-node=1 --gres=gpu:1 --job-name=NeurWP --time=23:59:00 --partition=normal --account=s83 torchrun --nnodes=1 --nproc_per_node=1 train_model.py
