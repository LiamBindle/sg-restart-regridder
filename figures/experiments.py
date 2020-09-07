import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shapereader
import shapely.geometry
import shapely.errors
import pyproj
from matplotlib.lines import Line2D

import figures

import sg.grids
import sg.plot

import tqdm
import cartopy


import pandas as pd
df = pd.DataFrame()
n = 1
def draw_target_face_outline(ax: plt.Axes, sf, tlat, tlon, color, face=5, is_tough=False, line=None):
    global n, df
    grid = sg.grids.StretchedGrid(12, sf, tlat, tlon)
    xx = grid.xe(face)
    yy = grid.ye(face)
    xx_majors = [xx[0, :], xx[-1, :], xx[:, 0], xx[:, -1]]
    yy_majors = [yy[0, :], yy[-1, :], yy[:, 0], yy[:, -1]]
    if line is not None:
        xx_majors = [xx_majors[line]]
        yy_majors = [yy_majors[line]]
    for xm, ym in zip(xx_majors, yy_majors):
        if is_tough:
            xm = xm % 360
        line = shapely.geometry.LineString(np.moveaxis([xm, ym], 0, -1))
        ax.add_feature(
            cartopy.feature.ShapelyFeature([line], crs=ccrs.PlateCarree()),
            facecolor='none',
            edgecolor=color,
            linewidth=0.8,
        )
        df[f'x:{n}'] = xm
        df[f'y:{n}'] = ym
        print(f'{list(xm)},')
        print(f'{list(ym)},')
        n = n + 1


if __name__ == '__main__':

    cmap = plt.get_cmap('Dark2')
    dtfal = 4*np.pi/180

    grids = [
        # {'sf': 2, 'tlat': 35, 'tlon': 264, 'color': cmap(0)},
        # {'sf': 3.6, 'tlat': 38, 'tlon': 252, 'color': cmap(0)},
        # {'sf': 6.8, 'tlat': 37, 'tlon': 244, 'color': cmap(0)},
        # {'sf': 12.5, 'tlat': 36, 'tlon': 241, 'color': cmap(0)},
        #
        # {'sf': 2, 'tlat': 48, 'tlon': 14, 'color': cmap(1)},
        # {'sf': 3.4, 'tlat': 47, 'tlon': 5, 'color': cmap(1)},
        # {'sf': 6.8, 'tlat': 42.5, 'tlon': 12.5, 'color': cmap(1)},
        # {'sf': 15, 'tlat': 45, 'tlon': 10.5, 'color': cmap(1)},
        #
        # {'sf': 2.8, 'tlat': 21.5, 'tlon': 79, 'color': cmap(2)},
        # {'sf': 6, 'tlat': 25, 'tlon': 81, 'color': cmap(2)},
        # {'sf': 14, 'tlat': 27.5, 'tlon': 78.5, 'color': cmap(2)},
        #
        # {'sf': 2.7, 'tlat': 5, 'tlon': 111, 'color': cmap(3)},
        # {'sf': 4, 'tlat': 0, 'tlon': 108, 'color': cmap(3)},
        # {'sf': 7, 'tlat': -5, 'tlon': 110, 'color': cmap(3)},
        # {'sf': 15, 'tlat': -7, 'tlon': 108, 'color': cmap(3)},

        # {'sf': 5, 'tlat': 40, 'tlon': 248, 'color': cmap(3)},
        # {'sf': 1, 'tlat': -90, 'tlon': 170, 'color': cmap(1)},
        {'sf': 3, 'tlat': 36, 'tlon': 261, 'color': cmap(1)},
        #
        {'sf': 1.0, 'tlat': -90, 'tlon': 170, 'color': cmap(0)},
        #
        # # {'sf': 8, 'tlat': 37, 'tlon': 242, 'color': cmap(3)},
        #
        {'sf': 10, 'tlat': 37.2, 'tlon': 240.5, 'color': cmap(2)},
    ]


    plt.figure(figsize=figures.one_col_figsize(1.6))
    ax = plt.axes(projection=ccrs.EqualEarth())
    ax.set_global()

    countries = shapereader.natural_earth(resolution='110m',
                                          category='cultural',
                                          name='admin_0_countries')

    exclude_US = [geom.geometry for geom in shapereader.Reader(countries).records() if geom.attributes['SOV_A3'] != 'US1']



    ax.add_feature(cfeature.OCEAN, linewidth=0, color='#f0f0f0')
    ax.add_feature(cfeature.LAND, facecolor='none', linewidth=0, color='#bdbdbd')
    ax.add_feature(cfeature.LAKES, linewidth=0, color='#f0f0f0')
    #ax.add_geometries(exclude_US, linewidth=0.2, edgecolor='gray', facecolor='none', crs=ccrs.Geodetic())
    # ax.add_feature(cfeature.BORDERS, linewidth=0.1, edgecolor='gray')
    # ax.add_feature(cfeature.STATES, linewidth=0.05, edgecolor='gray')
    #ax.outline_patch.set_edgecolor('#151515')
    ax.outline_patch.set_linewidth(0.2)
    ax.outline_patch.set_edgecolor('gray')


    custom_lines = [Line2D([0], [0], color=cmap(0), lw=1.5),
                    Line2D([0], [0], color=cmap(1), lw=1.5),
                    Line2D([0], [0], color=cmap(2), lw=1.5),]


    draw_target_face_outline(ax, **grids[0], face=0)
    draw_target_face_outline(ax, **grids[0], face=3, line=0, is_tough=True)
    draw_target_face_outline(ax, **grids[0], face=3, line=1)
    draw_target_face_outline(ax, **grids[0], face=3, line=2)
    draw_target_face_outline(ax, **grids[0], face=3, line=3)
    draw_target_face_outline(ax, **grids[0], face=4)
    draw_target_face_outline(ax, **grids[0], face=1, line=0)
    draw_target_face_outline(ax, **grids[0], face=1, line=1, is_tough=True)
    # draw_target_face_outline(ax, **grids[0], face=1, line=0)
    # draw_target_face_outline(ax, **grids[0], face=1, line=1)
    # draw_target_face_outline(ax, **grids[0], face=1, line=3)
    # draw_target_face_outline(ax, **grids[0], face=3, line=2, is_tough=True)
    # draw_target_face_outline(ax, **grids[0], face=3, line=3, is_tough=True)
    # draw_target_face_outline(ax, **grids[0], face=4, line=2)
    # draw_target_face_outline(ax, **grids[0], face=4, line=3)
    # draw_target_face_outline(ax, **grids[0], face=4, line=0)

    # draw_target_face_outline(ax, **grids[0], face=1, is_tough=False)
    # draw_target_face_outline(ax, **grids[0], face=0)
    # draw_target_face_outline(ax, **grids[0], face=4)
    # draw_target_face_outline(ax, **grids[0], face=2, line=3)
    #
    draw_target_face_outline(ax, **grids[1], face=1, is_tough=True)
    draw_target_face_outline(ax, **grids[1], face=3, is_tough=True)
    draw_target_face_outline(ax, **grids[1], face=4)
    draw_target_face_outline(ax, **grids[1], face=0)
    #
    draw_target_face_outline(ax, **grids[2])
    draw_target_face_outline(ax, **grids[2], face=1)
    draw_target_face_outline(ax, **grids[2], face=0)
    draw_target_face_outline(ax, **grids[2], face=4)
    draw_target_face_outline(ax, **grids[2], face=2)


    legend = ax.legend(
        custom_lines, ['Cubed-sphere grids', 'C180e-US', 'C900e-CA'],
        loc='upper center', mode='expand', ncol=3,
        handlelength=1, handletextpad=0.3, framealpha=1, prop={'size': 'small'}, columnspacing=3,
        bbox_to_anchor=(0, -0.15, 1, 0.1),
        borderpad=0.6, borderaxespad=0
    )
    legend.get_frame().set_linewidth(0.2)
    legend.get_frame().set_edgecolor('gray')

    # plt.scatter([-84.3880], [33.7490], transform=ccrs.PlateCarree(), color='k', s=0.5)


    plt.tight_layout()
    # plt.show()

    # figures.display_figure_instead=True
    figures.savefig(plt.gcf(), 'sg-experiments.png', pad_inches=0.01)
    # plt.savefig('/home/liam/Copernicus_LaTeX_Package/figures/sg-experiments.png')


df.T.to_csv('foo.csv')


