import cartopy.crs as ccrs

wandb_project = "neural-lam"

# TODO: fix for leap years
# Assuming no leap years in dataset (2024 is next)
seconds_in_year = 365 * 24 * 60 * 60

# Full names
param_names = [
    'Temperature',
    'Zonal wind component',
    'Meridional wind component',
    'Relative humidity',
]
# Short names
param_names_short = [
    'T',
    'U',
    'V',
    'RELHUM',
]

# Units
param_units = [
    'K',
    'm/s',
    'm/s',
    'g/g',
]

vertical_levels = [
    1,
    5,
    13,
    22,
    38,
    41,
    60
]

# Projection and grid
grid_shape = (582, 390)  # (x, y)

grid_limits = [  # In projection
    -0.6049805,  # min x
    17.48751,  # max x
    42.1798,  # min y
    50.35996,  # max y
]

cosmo_proj = ccrs.PlateCarree()
