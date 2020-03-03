
import matplotlib.pyplot as plt
from matplotlib import rc
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import sg.grids
import sg.plot

import matplotlib.pyplot as plt
import sg.plot
import sg.figure_axes

import numpy as np

from tqdm import tqdm


def draw_grid(sf, tlat, tlon, frame_number):
    grid = sg.grids.StretchedGrid(24, sf, tlat, tlon)

    # rc('text', usetex=True)
    rc('font', **{'family': 'sans-serif', 'size': 20, 'weight': 'bold'})

    plt.figure(figsize=(12, 8))
    projection=ccrs.Orthographic(tlon, tlat)
    ax = plt.axes(projection=projection)
    ax.set_global()
    ax.outline_patch.set_linewidth(0.3)

    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.LAND, edgecolor='black', linewidth=0.3)
    ax.add_feature(cartopy.feature.LAKES, edgecolor='black', linewidth=0.3)
    ax.add_feature(cartopy.feature.STATES, edgecolor='black', linewidth=0.4)
    ax.add_feature(cartopy.feature.BORDERS, edgecolor='black', linewidth=0.4)

    figax = sg.figure_axes.FigureAxes(ax, projection=projection)
    for i in range(6):
        sg.plot.draw_minor_grid_boxes(figax, *figax.transform_xy(grid.xe(i), grid.ye(i)), color='black', linewidth=0.3)
        sg.plot.draw_major_grid_boxes(figax, *figax.transform_xy(grid.xe(i), grid.ye(i)), color='black', linewidth=1.6)
        plt.tight_layout()
    plt.scatter(106.8456, -6.2088, transform=ccrs.PlateCarree())
    plt.scatter(107.6191, -6.9175, transform=ccrs.PlateCarree())
    plt.text(0.95, 0.8, f'C24:\nSF:\nTLat:\nTLon:', transform=ax.transAxes)
    plt.text(1.08, 0.8, f'\n{sf:4.1f}x\n{tlat:4.1f}° N\n{-tlon:4.1f}° W', transform=ax.transAxes)
    # plt.savefig(f'/extra-space/frames-sg-demo/{frame_number:02d}.jpeg', dpi=100, quality=80, optimize=True)
    # plt.close()


def logspace(distance, base, num):
    v = np.logspace(1, np.log((distance+1)+(base-1)) / np.log(base), num=num, base=base)-(base-1)
    return v-1


if __name__ == '__main__':
    #stretch = [*np.linspace(1, 2, 12), *np.linspace(2, 15, 24)]

    # base = 1.1
    # stretch = np.logspace(1, np.log(10+(base-1)) / np.log(base), num=24, base=base)-(base-1)
    #
    #
    # # lat = np.linspace(0, 33.75, 24)
    # # lon = np.linspace(-100, -84.39, 24)
    #
    # lat = abs(logspace(33.75, 1.5, 24)[::-1]-33.75)
    # lon = -logspace(15.61, 1.5, 24)[::-1]-84.39
    #
    # for frame_number, (sf, tlat, tlon) in tqdm(enumerate(zip(stretch, lat, lon))):
    #     draw_grid(sf, tlat, tlon, frame_number)

    draw_grid(sf=15, tlat=-7, tlon=108, frame_number=-1)
    plt.show()
