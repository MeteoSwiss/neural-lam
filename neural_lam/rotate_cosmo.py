"""unrotate COSMO rotated coordinates to geographical lat/lon"""

import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def unrot_lon(rotlon, rotlat, pollon, pollat):
    """Transform rotated longitude to longitude.

    Parameters
    ----------
    rotlon : np.ndarray(i,j)
        rotated longitude (deg)
    rotlat : np.ndarray(i,j)
        rotated latitude (deg)
    pollon : float
        rotated pole longitude (deg)
    pollat : float
        rotated pole latitude (deg)

    Returns
    -------
    lon : np.ndarray(i,j)
        geographical longitude

    """

    # to radians
    rlo = np.radians(rotlon)
    rla = np.radians(rotlat)

    # sin and cos of pole position
    s1 = np.sin(np.radians(pollat))
    c1 = np.cos(np.radians(pollat))
    s2 = np.sin(np.radians(pollon))
    c2 = np.cos(np.radians(pollon))

    # subresults
    tmp1 = s2 * (-s1 * np.cos(rlo) * np.cos(rla) + c1 *
                 np.sin(rla)) - c2 * np.sin(rlo) * np.cos(rla)
    tmp2 = c2 * (-s1 * np.cos(rlo) * np.cos(rla) + c1 *
                 np.sin(rla)) + s2 * np.sin(rlo) * np.cos(rla)

    return np.degrees(np.arctan(tmp1 / tmp2))


def unrot_lat(rotlat, rotlon, pollon, pollat):
    """Transform rotated latitude to latitude.

    Parameters
    ----------
    rotlat : np.ndarray(i,j)
        rotated latitude (deg)
    rotlon : np.ndarray(i,j)
        rotated longitude (deg)
    pollon : float
        rotated pole longitude (deg)
    pollat : float
        rotated pole latitude (deg)

    Returns
    -------
    lat : np.ndarray(i,j)
        geographical latitude

    """

    # to radians
    rlo = np.radians(rotlon)
    rla = np.radians(rotlat)

    # sin and cos of pole position
    s1 = np.sin(np.radians(pollat))
    c1 = np.cos(np.radians(pollat))

    # subresults
    tmp1 = s1 * np.sin(rla) + c1 * np.cos(rla) * np.cos(rlo)

    return np.degrees(np.arcsin(tmp1))

def unrotate_latlon(data):
    # COSMO-1E rotated pole position
    pollon = -170.0
    pollat = 43.0
    
    xx, yy = np.meshgrid(data.x_1.values, data.y_1.values)
    # unrotate lon/lat
    lon = unrot_lon(xx, yy, pollon, pollat)
    lat = unrot_lat(yy, xx, pollon, pollat)

    return lon, lat


if __name__ == '__main__':

    # get test data
    data = xr.open_dataset(
        "/scratch/e1000/meteoswiss/scratch/cosuna/KENDA/ml_v1/ANA16/20160101/laf2016010100_extr.nc")
    var = data.T.values[0, 59, :, :]

    lon, lat = unrotate_latlon(data)

    # plot
    fig = plt.figure()

    ax1 = fig.add_subplot(projection=ccrs.PlateCarree())
    ax1.contourf(lon, lat, var, transform=ccrs.PlateCarree())
    ax1.add_feature(cf.BORDERS, linestyle='-', edgecolor='white')
    ax1.add_feature(cf.COASTLINE, linestyle='-', edgecolor='white')
    ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0)
    plt.title("Sones hübsches Chärtli *.*")

    plt.savefig('test.png')
