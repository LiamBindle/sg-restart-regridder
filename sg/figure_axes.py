from typing import Tuple

import pyproj
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


class FigureAxes:
    def __init__(self, ax: plt.Axes, projection: ccrs.Projection):
        self._ax = ax
        self._crs = pyproj.Proj(projection.proj4_init)
        self._x_split = None
        if isinstance(projection, ccrs._RectangularProjection):
            self._x_split=180

    @property
    def ax(self) -> plt.Axes:
        return self._ax

    @property
    def crs(self) -> plt.Axes:
        return self._crs

    @property
    def x_split(self) -> float:
        return self._x_split

    def transform_xy(self, xx, yy, init_crs: pyproj.Proj=pyproj.Proj(init='epsg:4326')) -> Tuple[np.ndarray, np.ndarray]:
        return pyproj.transform(init_crs, self.crs, xx, yy)