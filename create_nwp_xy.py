import xarray as xr
import numpy as np

# Load the netCDF file
ds = xr.open_dataset('data/cosmo/train/laf2015112800_extr.nc')

# Get the dimensions for x and y
x_dim, y_dim = ds.dims['x_1'], ds.dims['y_1']

# Create a 2D meshgrid for x and y indices
x_grid, y_grid = np.indices((x_dim, y_dim))

# Invert the order of x_grid
x_grid = np.transpose(x_grid)
y_grid = np.transpose(y_grid)

# Stack the 2D arrays into a 3D array with x and y as the first dimension
grid_xy = np.stack((x_grid, y_grid))

np.save('data/cosmo/static/nwp_xy.npy', grid_xy)