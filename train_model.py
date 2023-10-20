import os
import time
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from lightning_fabric.utilities import seed
from pytorch_lightning.utilities import rank_zero_only
from torch.distributed import init_process_group

import wandb
from neural_lam import constants
from neural_lam.models.graph_lam import GraphLAM
from neural_lam.models.hi_lam import HiLAM
from neural_lam.models.hi_lam_parallel import HiLAMParallel
from neural_lam.weather_dataset import WeatherDataset

MODELS = {
    "graph_lam": GraphLAM,
    "hi_lam": HiLAM,
    "hi_lam_parallel": HiLAMParallel,
}


@rank_zero_only
def print_args(args):
    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")


@rank_zero_only
def print_eval(args_eval):
    print(f"Running evaluation on {args_eval}")


@rank_zero_only
def init_wandb(args):
    prefix = "subset-" if args.subset_ds else ""
    if args.eval:
        prefix = prefix + f"eval-{args.eval}-"
    run_name = f"{prefix}{args.model}-{args.processor_layers}x{args.hidden_dim}-"\
        f"{time.strftime('%m_%d_%H_%M_%S')}"
    wandb.init(project=constants.wandb_project, name=run_name, config=args)
    logger = pl.loggers.WandbLogger(project=constants.wandb_project, name=run_name,
                                    config=args)
    return logger, run_name


@rank_zero_only
def init_checkpoint_callback(run_name):
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"saved_models/{run_name}", filename="min_val_loss",
        monitor="val_mean_loss", mode="min", save_last=True)
    return checkpoint_callback


def main():
    if torch.cuda.is_available():
        init_process_group(backend="nccl")
    parser = ArgumentParser(description='Train or evaluate NeurWP models for LAM')

    # General options
    parser.add_argument(
        '--dataset', type=str, default="meps_example",
        help='Dataset, corresponding to name in data directory (default: meps_example)')
    parser.add_argument(
        '--model', type=str, default="graph_lam",
        help='Model architecture to train/evaluate (default: graph_lam)')
    parser.add_argument(
        '--subset_ds', type=int, default=0,
        help='Use only a small subset of the dataset, for debugging (default: 0=false)')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')
    parser.add_argument('--n_workers', type=int, default=4,
                        help='Number of workers in data loader (default: 4)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='upper epoch limit (default: 200)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch size (default: 4)')
    parser.add_argument('--load', type=str,
                        help='Path to load model parameters from (default: None)')
    parser.add_argument(
        '--restore_opt', type=int, default=0,
        help='If optimizer state should be restored with model (default: 0 (false))')
    parser.add_argument(
        '--precision', type=str, default=32,
        help='Numerical precision to use for model (32/16/bf16) (default: 32)')

    # Model architecture
    parser.add_argument(
        '--graph', type=str, default="multiscale",
        help='Graph to load and use in graph-based model (default: multiscale)')
    parser.add_argument(
        '--hidden_dim', type=int, default=64,
        help='Dimensionality of all hidden representations (default: 64)')
    parser.add_argument('--hidden_layers', type=int, default=1,
                        help='Number of hidden layers in all MLPs (default: 1)')
    parser.add_argument('--processor_layers', type=int, default=4,
                        help='Number of GNN layers in processor GNN (default: 4)')
    parser.add_argument(
        '--mesh_aggr', type=str, default="sum",
        help='Aggregation to use for m2m processor GNN layers (sum/mean) (default: sum)')

    # Training options
    parser.add_argument(
        '--ar_steps', type=int, default=1,
        help='Number of steps to unroll prediction for in loss (1-24) (default: 1)')
    parser.add_argument('--loss', type=str, default="mse",
                        help='Loss function to use (default: mse)')
    parser.add_argument(
        '--step_length', type=int, default=1,
        help='Step length in hours to consider single time step 1-3 (default: 1)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (default: 0.001)')
    parser.add_argument(
        '--val_interval', type=int, default=1,
        help='Number of epochs training between each validation run (default: 1)')

    # Evaluation options
    parser.add_argument(
        '--eval', type=str, default=None,
        help='Eval model on given data split (val/test) (default: None (train model))')
    parser.add_argument(
        '--n_example_pred', type=int, default=1,
        help='Number of example predictions to plot during evaluation (default: 1)')
    # TODO Remove this
    args = parser.parse_args(
        args=[
            '--dataset',
            'cosmo',
            '--subset_ds',
            '1',
            '--ar_steps',
            '24',
            '--n_workers',
            '32',
            '--epochs',
            '2',
            '--batch_size',
            '4',
            '--eval',
            'test'])
    print_args(args)

    # Asserts for arguments
    assert args.model in MODELS, f"Unknown model: {args.model}"
    assert args.step_length <= 3, "Too high step length"
    assert args.eval in (None, "val", "test"), f"Unknown eval setting: {args.eval}"

    # Set seed
    seed.seed_everything(args.seed)

    # Create dataset
    train_dataset = WeatherDataset(
        args.dataset,
        split="train",
        subset=bool(
            args.subset_ds))
    val_dataset = WeatherDataset(args.dataset, split="val", subset=bool(args.subset_ds))

    # Create the data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # set to False when using DistributedSampler
        num_workers=args.n_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # set to False when using DistributedSampler
        num_workers=args.n_workers
    )

    # Get the device for the current process
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")  # Allows using Tensor Cores on A100s

    # Load model parameters Use new args for model
    model_class = MODELS[args.model]
    if args.load:
        model = model_class.load_from_checkpoint(args.load, args=args)
        if args.restore_opt:
            # Save for later
            # Unclear if this works for multi-GPU
            model.opt_state = torch.load(args.load)["optimizer_states"][0]
    else:
        model = model_class(args)

    result = init_wandb(args)
    if result is not None:
        logger, run_name = result
    else:
        logger = None
        run_name = None

    checkpoint_callback = init_checkpoint_callback(run_name)

    if args.eval:
        use_distributed_sampler = False
    else:
        use_distributed_sampler = True

    if torch.cuda.is_available():
        accelerator = "cuda"
        devices = torch.cuda.device_count()
        num_nodes = int(os.environ.get('SLURM_JOB_NUM_NODES', 1))
    else:
        accelerator = "cpu"
        devices = 1
        num_nodes = 1

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        deterministic=True,
        logger=logger,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback] if checkpoint_callback is not None else [],
        check_val_every_n_epoch=args.val_interval,
        precision=args.precision,
        use_distributed_sampler=use_distributed_sampler,
        accelerator=accelerator,
        devices=devices,
        num_nodes=num_nodes,
    )
    if args.eval:
        if args.eval == "val":
            eval_loader = val_loader
        else:  # Test
            # Create dataset
            eval_dataset = WeatherDataset(
                args.dataset, split="test", subset=bool(
                    args.subset_ds))

            # Create the data loader
            eval_loader = torch.utils.data.DataLoader(
                eval_dataset,
                batch_size=args.batch_size,
                shuffle=False,  # set to False when using DistributedSampler
                num_workers=args.n_workers
            )

        print_eval(args.eval)

        trainer.test(model=model, dataloaders=eval_loader)
    else:
        # Train model
        trainer.fit(model=model, train_dataloaders=train_loader,
                    val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
