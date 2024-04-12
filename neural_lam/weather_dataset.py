# Standard library
import os
from datetime import datetime, timedelta
from random import randint

# Third-party
import numpy as np
import numpy as np
import pytorch_lightning as pl
import torch
import xarray as xr

# First-party
from neural_lam import constants, utils


class WeatherDataset(torch.utils.data.Dataset):
    """Weather dataset for PyTorch Lightning."""

    def __init__(
        self,
        dataset_name,
        path_verif_file=None,
        split="train",
        standardize=True,
        subset=False,
        batch_size=4,
        control_only=False,
    ):
        super().__init__()

        assert split in ("train", "val", "test"), "Unknown dataset split"
        self.zarr_path = os.path.join(
            "data", dataset_name, "samples", split, "data.zarr"
        )
        self.ds = xr.open_zarr(self.zarr_path, consolidated=True)
        if split == "train":
            self.ds = self.ds.sel(time=slice("2015", "2019"))
        elif split == "verif":
            self.np_files = np.load(path_verif_file)
            self.split = split
        else:
            self.ds = self.ds.sel(time=slice("2020", "2020"))

        new_vars = {
            (
                f"{var_name}_{int(level)}"
                if constants.IS_3D[var_name]
                else var_name
            ): (
                data_array.sel(level=level).drop_vars("level")
                if constants.IS_3D[var_name]
                else data_array
            )
            for var_name, data_array in self.ds.data_vars.items()
            if var_name in constants.PARAM_NAMES_SHORT
            for level in constants.VERTICAL_LEVELS
        }

        self.ds = (
            xr.Dataset(new_vars)
            .to_array()
            .transpose("time", "x", "y", "variable")
        )

        if subset:
            if constants.EVAL_DATETIMES is not None and split == "test":
                eval_datetime_obj = datetime.strptime(
                    constants.EVAL_DATETIMES[0], "%Y%m%d%H"
                )
                self.ds = self.ds.sel(
                    time=slice(
                        eval_datetime_obj,
                        eval_datetime_obj + timedelta(hours=50),
                    )
                )
            else:
                start_idx = randint(0, self.ds.time.size - 50)
                self.ds = self.ds.isel(time=slice(start_idx, start_idx + 50))

        self.standardize = standardize
        if standardize:
            ds_stats = utils.load_dataset_stats(dataset_name, "cpu")
            if constants.GRID_FORCING_DIM > 0:
                (
                    self.data_mean,
                    self.data_std,
                    self.flux_mean,
                    self.flux_std,
                ) = (
                    ds_stats["data_mean"],
                    ds_stats["data_std"],
                    ds_stats["flux_mean"],
                    ds_stats["flux_std"],
                )
            else:
                self.data_mean, self.data_std = (
                    ds_stats["data_mean"],
                    ds_stats["data_std"],
                )
        self.random_subsample = split == "train"
        self.split = split
        self.num_steps = (
            constants.TRAIN_HORIZON
            if self.split == "train"
            else constants.EVAL_HORIZON
        )
        self.batch_size = batch_size
        self.control_only = control_only

    def __len__(self):
        return len(self.ds.time) - self.num_steps

    def __getitem__(self, idx):
        if self.split == "verif":
            return self.np_files
        num_steps = (
            constants.TRAIN_HORIZON
            if self.split == "train"
            else constants.EVAL_HORIZON
        )

        # Calculate which zarr files need to be loaded
        start_file_idx = idx // constants.CHUNK_SIZE
        end_file_idx = (idx + num_steps) // constants.CHUNK_SIZE
        # Index of current slice
        idx_sample = idx % constants.CHUNK_SIZE

        sample_archive = xr.concat(
            self.zarr_datasets[start_file_idx: end_file_idx + 1], dim="time"
        )

        sample_xr = sample_archive.isel(
            time=slice(idx_sample, idx_sample + num_steps)
        )

        # (N_t', N_x, N_y, d_features')
        sample = torch.tensor(sample_xr.values, dtype=torch.float32)
        sample = sample.flatten(1, 2)  # (N_t, N_grid, d_features)

        if self.standardize:
            sample = (sample - self.data_mean) / self.data_std

        init_states = sample[:2]  # (2, N_grid, d_features)
        target_states = sample[2:]  # (sample_length-2, N_grid, d_features)

        batch_time = self.ds.isel(time=idx).time.values
        batch_time = np.datetime_as_string(batch_time, unit="h")
        batch_time = str(batch_time).replace("-", "").replace("T", "")
        return init_states, target_states, batch_time


class WeatherDataModule(pl.LightningDataModule):
    """DataModule for weather data."""

    def __init__(
        self,
        dataset_name,
        split="train",
        path_verif_file=None,
        standardize=True,
        subset=False,
        batch_size=4,
        num_workers=16,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.path_verif_file = path_verif_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.standardize = standardize
        self.subset = subset
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = WeatherDataset(
                dataset_name=self.dataset_name,
                split="train",
                standardize=self.standardize,
                subset=self.subset,
                batch_size=self.batch_size,
            )
            self.val_dataset = WeatherDataset(
                dataset_name=self.dataset_name,
                split="val",
                standardize=self.standardize,
                subset=self.subset,
                batch_size=self.batch_size,
            )

        if stage == "test" or stage is None:
            self.test_dataset = WeatherDataset(
                dataset_name=self.dataset_name,
                split="test",
                standardize=self.standardize,
                subset=self.subset,
                batch_size=self.batch_size,
            )

        if stage == "verif":
            self.verif_dataset = WeatherDataset(
                self.dataset_name,
                self.path_verif_file,
                split="verif",
                standardize=False,
                subset=False,
                batch_size=self.batch_size,
            )

        if stage == "pred" or stage is None:
            self.pred_dataset = WeatherDataset(
                self.dataset_name,
                split="pred",
                standardize=self.standardize,
                subset=False,
                batch_size=1,
            )

    def train_dataloader(self):
        """Load train dataset."""
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,
        )

    def val_dataloader(self):
        """Load validation dataset."""
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,
        )

    def test_dataloader(self):
        """Load test dataset."""
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,
        )

    def pred_dataloader(self):
        """Load prediction dataset."""
        return torch.utils.data.DataLoader(
            self.pred_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,
        )

    def verif_dataloader(self):
        """Load inference output dataset."""
        return torch.utils.data.DataLoader(
            self.verif_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=False,
        )
