from typing import Tuple

import pyproj
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

from sg.transform import make_grid_SCS

class StretchedGrid:
    def __init__(self, cs, sf, target_lat, target_lon):
        _, self._csgrid_list = make_grid_SCS(cs, sf, target_lat, target_lon)

    def xe(self, face) -> np.ndarray:
        return self._csgrid_list[face]['lon_b']

    def ye(self, face) -> np.ndarray:
        return self._csgrid_list[face]['lat_b']


class FigureAxes:
    def __init__(self, ax: plt.Axes, projection: ccrs.Projection):
        self._ax = ax
        self._crs = pyproj.Proj(projection.proj4_init)

    @property
    def ax(self) -> plt.Axes:
        return self._ax

    @property
    def crs(self) -> plt.Axes:
        return self._crs

    def transform_xy(self, xx, yy, init_crs: pyproj.Proj=pyproj.Proj(init='epsg:4326')) -> Tuple[np.ndarray, np.ndarray]:
        return pyproj.transform(init_crs, self.crs, xx, yy)


def draw_facenumber(figax: FigureAxes, grid: StretchedGrid, face, **kwargs):
    xx, yy = figax.transform_xy(grid.xe(face), grid.ye(face))
    middle = np.array(xx.shape, dtype=np.int) // 2
    figax.ax.text(xx[middle[0], middle[1]], yy[middle[0], middle[1]], f'{face}', **kwargs)


def draw_gridboxes(figax: FigureAxes, grid: StretchedGrid, major={}, minor={}, x_split=None):
    major.setdefault('linewidth', 1.5)
    major.setdefault('color', 'black')
    minor.setdefault('linewidth', 0.3)
    minor.setdefault('color', 'black')
    ax = figax.ax
    for i in range(6):
        xe, ye = grid.xe(i), grid.ye(i)
        xx, yy = figax.transform_xy(xe, ye)
        for x, y in zip(xx, yy):
            if x_split:
                idx = np.argwhere(np.diff(np.sign(x % 360 - x_split))).flatten()
                x360 = x % 360
                idx = idx[(x360[idx] > 10) & (x360[idx] < 350)]
                start = [0, *(idx+1)]
                end = [*(idx+1), len(x)]
            else:
                start = [0]
                end = [len(x)]
            for s, e in zip(start, end):
                ax.plot(x[s:e], y[s:e], **minor)
        for x, y in zip(xx.transpose(), yy.transpose()):
            if x_split:
                idx = np.argwhere(np.diff(np.sign(x % 360 - x_split))).flatten()
                x360 = x % 360
                idx = idx[(x360[idx] > 10) & (x360[idx] < 350)]
                start = [0, *(idx + 1)]
                end = [*(idx + 1), len(x)]
            else:
                start = [0]
                end = [len(x)]
            for s, e in zip(start, end):
                ax.plot(x[s:e], y[s:e], **minor)
        ax.plot(xx[0,:], yy[0,:], **major)
        ax.plot(xx[-1, :], yy[-1, :], **major)
        ax.plot(xx[:, 0], yy[:, 0], **major)
        ax.plot(xx[:, -1], yy[:, -1], **major)


if __name__ == '__main__':
    f = plt.figure()

    grid = StretchedGrid(48, 15, 36,  360 - 78)

    proj = ccrs.PlateCarree()
    #proj = ccrs.NearsidePerspective(360 - 78, 36, 740e3)

    ax = plt.subplot(1, 1, 1, projection=proj)
    figax = FigureAxes(ax, proj)

    ax.set_global()
    ax.coastlines(linewidth=0.8)

    draw_gridboxes(figax, grid, x_split=180)

    for i in range(6):
        draw_facenumber(figax, grid, i)

    plt.tight_layout()
    plt.show()