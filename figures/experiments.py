import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shapereader
import shapely.geometry
import shapely.errors
import pyproj
from matplotlib.lines import Line2D

import sg.grids
import sg.plot

import tqdm

def draw_target_face_outline(ax: plt.Axes, sf, tlat, tlon, color):
    grid = sg.grids.StretchedGrid(24, sf, tlat, tlon)
    xx = grid.xe(5)
    yy = grid.ye(5)
    xx_majors = [xx[0, :], xx[-1, :], xx[:, 0], xx[:, -1]]
    yy_majors = [yy[0, :], yy[-1, :], yy[:, 0], yy[:, -1]]
    for xm, ym in zip(xx_majors, yy_majors):
        ax.plot(xm, ym, transform=ccrs.PlateCarree(), color=color, linewidth=0.8, linestyle='-')


if __name__ == '__main__':

    cmap = plt.get_cmap('Dark2')
    dtfal = 4*np.pi/180

    grids = [
        {'sf': 2, 'tlat': 35, 'tlon': 264, 'color': cmap(0)},
        {'sf': 3.6, 'tlat': 38, 'tlon': 252, 'color': cmap(0)},
        {'sf': 6.8, 'tlat': 37, 'tlon': 244, 'color': cmap(0)},
        {'sf': 12.5, 'tlat': 36, 'tlon': 241, 'color': cmap(0)},

        {'sf': 2, 'tlat': 48, 'tlon': 14, 'color': cmap(1)},
        {'sf': 3.4, 'tlat': 47, 'tlon': 5, 'color': cmap(1)},
        {'sf': 6.8, 'tlat': 42.5, 'tlon': 12.5, 'color': cmap(1)},
        {'sf': 15, 'tlat': 45, 'tlon': 10.5, 'color': cmap(1)},

        {'sf': 2.8, 'tlat': 21.5, 'tlon': 79, 'color': cmap(2)},
        {'sf': 6, 'tlat': 25, 'tlon': 81, 'color': cmap(2)},
        {'sf': 14, 'tlat': 27.5, 'tlon': 78.5, 'color': cmap(2)},

        {'sf': 2.7, 'tlat': 5, 'tlon': 111, 'color': cmap(3)},
        {'sf': 4, 'tlat': 0, 'tlon': 108, 'color': cmap(3)},
        {'sf': 7, 'tlat': -5, 'tlon': 110, 'color': cmap(3)},
        {'sf': 15, 'tlat': -7, 'tlon': 108, 'color': cmap(3)},
    ]


    plt.figure(figsize=(3.26772,1.6))
    ax = plt.axes(projection=ccrs.EqualEarth())
    ax.set_global()

    countries = shapereader.natural_earth(resolution='110m',
                                          category='cultural',
                                          name='admin_0_countries')

    exclude_US = [geom.geometry for geom in shapereader.Reader(countries).records() if geom.attributes['SOV_A3'] != 'US1']



    ax.add_feature(cfeature.OCEAN, linewidth=0)
    ax.add_feature(cfeature.LAND, facecolor='none', linewidth=0)
    ax.add_feature(cfeature.LAKES, linewidth=0)
    #ax.add_geometries(exclude_US, linewidth=0.2, edgecolor='gray', facecolor='none', crs=ccrs.Geodetic())
    ax.add_feature(cfeature.BORDERS, linewidth=0.1, edgecolor='gray')
    ax.add_feature(cfeature.STATES, linewidth=0.05, edgecolor='gray')
    #ax.outline_patch.set_edgecolor('#151515')
    ax.outline_patch.set_linewidth(0.2)
    ax.outline_patch.set_edgecolor('gray')


    custom_lines = [Line2D([0], [0], color=cmap(0), lw=1.5),
                    Line2D([0], [0], color=cmap(1), lw=1.5),
                    Line2D([0], [0], color=cmap(2), lw=1.5),
                    Line2D([0], [0], color=cmap(3), lw=1.5)]

    for g in tqdm.tqdm(grids):
        draw_target_face_outline(ax, **g)

    legend = ax.legend(
        custom_lines, ['NA1--4', 'EU1--4', 'IN1--3', 'SE1--4'],
        loc='lower center', mode='expand', ncol=4,
        handlelength=0.4, handletextpad=0.2, framealpha=1, prop={'size': 'small'}, columnspacing=4,
        bbox_to_anchor=(0, 0.1, 1, 0.1),
        borderpad=0.3, borderaxespad=0
    )
    legend.get_frame().set_linewidth(0.2)
    legend.get_frame().set_edgecolor('gray')


    plt.tight_layout()
    # plt.show()

    plt.savefig('/home/liam/Copernicus_LaTeX_Package/figures/sg-experiments.png')



