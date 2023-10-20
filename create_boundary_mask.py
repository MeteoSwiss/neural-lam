import numpy as np
import xarray as xr

# Open the .nc file
ds = xr.open_dataset('data/cosmo/train/laf2015112800_extr.nc')

# Get the dimensions of the dataset
dims = ds.dims

# Create a mask with the same dimensions, initially set to False
mask = np.full((dims['x_1'], dims['y_1']), False)

# Set the 30 grid-cells closest to each boundary to True
mask[:30, :] = True  # top boundary
mask[-30:, :] = True  # bottom boundary
mask[:, :30] = True  # left boundary
mask[:, -30:] = True  # right boundary

# Save the numpy array to a .npy file
np.save('data/cosmo/static/border_mask', mask)
