import xarray as xr
import numpy as np

# Open the .nc file
ds = xr.open_dataset('data/cosmo/train/laf2015112800_extr.nc')

# Extract the 'HSURF' data variable
hsurf = ds['hsurf']

# Convert the DataArray to numpy array
hsurf_np = hsurf.values.transpose()

# Save the numpy array to a .npy file
np.save('data/cosmo/static/surface_geopotential.npy', hsurf_np)
