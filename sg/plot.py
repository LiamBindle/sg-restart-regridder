from typing import Tuple, List

import pyproj
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

from sg.transform import make_grid_SCS
from gcpy.grid.horiz import make_grid_CS


class CSDataBase:
    def xe(self, face) -> np.ndarray:
        return self._csgrid_list[face]['lon_b']

    def ye(self, face) -> np.ndarray:
        return self._csgrid_list[face]['lat_b']

    def data(self, face) -> xr.DataArray:
        return  self._data[face]


class CSData(CSDataBase):
    def __init__(self, cs, data: List[xr.DataArray]):
        _, self._csgrid_list = make_grid_CS(cs)
        self._data = data


class StretchedGrid(CSDataBase):
    def __init__(self, cs, sf, target_lat, target_lon, data: List[xr.DataArray]):
        _, self._csgrid_list = make_grid_SCS(cs, sf, target_lat, target_lon)
        self._data = data


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


def draw_facenumber(figax: FigureAxes, grid: CSDataBase, face, **kwargs):
    xx, yy = figax.transform_xy(grid.xe(face), grid.ye(face))
    middle = np.array(xx.shape, dtype=np.int) // 2
    figax.ax.text(xx[middle[0], middle[1]], yy[middle[0], middle[1]], f'{face}', **kwargs)


def draw_gridboxes(figax: FigureAxes, grid: CSDataBase, major={}, minor={}, x_split=None):
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

        xx_majors = [xx[0,:], xx[-1, :], xx[:, 0], xx[:, -1]]
        yy_majors = [yy[0,:], yy[-1, :], yy[:, 0], yy[:, -1]]
        for x, y in zip(xx_majors, yy_majors):
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
                ax.plot(x[s:e], y[s:e], **major)


def plot_columnar_data(figax: FigureAxes, grid: CSDataBase, face, x_split=180):
    xx, yy = figax.transform_xy(grid.xe(face), grid.ye(face))

    mask = np.zeros(grid.data(face).shape, dtype=bool)

    for i, x in enumerate(xx[:-1,:-1]):
        if x_split:
            idx = np.argwhere(np.diff(np.sign(x % 360 - x_split))).flatten()
            x360 = x % 360
            idx = idx[(x360[idx] > 10) & (x360[idx] < 350)]
            mask[i, idx] = True
    for i, x in enumerate(xx[:-1,:-1].transpose()):
        if x_split:
            idx = np.argwhere(np.diff(np.sign(x % 360 - x_split))).flatten()
            x360 = x % 360
            idx = idx[(x360[idx] > 10) & (x360[idx] < 350)]
            mask[idx, i] = True

    data = np.ma.masked_where(mask, grid.data(face).values)
    pc = figax.ax.pcolormesh(xx, yy, data)


if __name__ == '__main__':
    f = plt.figure()


    d = xr.open_dataset('/home/liam/stretched_grid/GCHP.SpeciesConc.20160201_1630z-control.nc4')

    data = [d.SpeciesConc_Rn222.isel(lev=slice(0, 29)).mean(dim='lev')[0, face, ::] for face in range(6)]

    #grid = StretchedGrid(48, 15, 36,  360 - 78, data)
    grid = CSData(48, data)

    proj = ccrs.PlateCarree()
    #proj = ccrs.NearsidePerspective(360 - 78, 36, 740e3)


    ax = plt.subplot(1, 1, 1, projection=proj)
    figax = FigureAxes(ax, proj)

    ax.set_global()
    ax.coastlines(linewidth=0.8)


    draw_gridboxes(figax, grid, x_split=180)

    plot_columnar_data(figax, grid, 0)
    plot_columnar_data(figax, grid, 1)
    plot_columnar_data(figax, grid, 2)
    plot_columnar_data(figax, grid, 3)
    plot_columnar_data(figax, grid, 4)
    plot_columnar_data(figax, grid, 5)

    for i in range(6):
       draw_facenumber(figax, grid, i)

    plt.show()