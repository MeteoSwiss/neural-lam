import glob
import os

import torch
import xarray as xr

# BUG: Import should work in interactive mode as well
from neural_lam import constants, utils


class WeatherDataset(torch.utils.data.Dataset):
    """
    For our dataset:
    N_t = 1h
    N_x = 582
    N_y = 390
    N_grid = 582*390 = 226980
    d_features = 8(features) * 7(vertical model levels) = 56
    d_forcing = 0 #TODO: extract incoming radiation from KENDA
    """

    def __init__(self, dataset_name, split="train", standardize=True, subset=False):
        super().__init__()

        assert split in ("train", "val", "test"), "Unknown dataset split"
        self.sample_dir_path = os.path.join("data", dataset_name, "samples", split)

        sample_paths = glob.glob(os.path.join(self.sample_dir_path, "laf*.nc"))
        self.sample_names = [path.split("/")[-1][3:-8] for path in sample_paths]
        # Now on form "yyymmddhh_mbrXXX"

        if subset:
            self.sample_names = self.sample_names[:50]  # Limit to 50 samples

        # Set up for standardization
        self.standardize = standardize
        if standardize:
            self.data_mean, self.data_std =\
                utils.load_dataset_stats(dataset_name, "cpu")

        # If subsample index should be sampled (only duing training)
        self.random_subsample = split == "train"

    def __len__(self):
        return len(self.sample_names) - 2

    def __getitem__(self, idx):
        # === Sample ===
        sample = torch.tensor([])
        for i in range(3):
            sample_name = self.sample_names[idx + i]
            sample_path = os.path.join(
                self.sample_dir_path,
                f"laf{sample_name}_extr.nc")
            try:
                ds = xr.load_dataset(sample_path)[constants.param_names_short]
            except ValueError:
                print(f"Failed to load {sample_path}")

            # Select the data for the specified vertical levels
            selected_data = ds.sel(z_1=constants.vertical_levels)

            # Now, you can create separate data variables for each vertical level and
            # each variable
            for var in selected_data.data_vars:
                for level in constants.vertical_levels:
                    new_var_name = f"{var}_z{level}"
                    selected_data[new_var_name] = selected_data[var].sel(z_1=level)

            da = selected_data.drop_dims("z_1").to_array().transpose(
                "time", "x_1", "y_1", "variable").values

            sample = torch.cat((torch.tensor(da, dtype=torch.float32), sample))
            # (N_t', N_x, N_y, d_features')

        # Flatten spatial dim
        sample = sample.flatten(1, 2)  # (N_t, N_grid, d_features)

        if self.standardize:
            # Standardize sample
            sample = (sample - self.data_mean) / self.data_std

        # Split up sample in init. states and target states
        init_states = sample[:2]  # (2, N_grid, d_features)
        target_states = sample[2:]  # (sample_length-2, N_grid, d_features)

        return init_states, target_states
