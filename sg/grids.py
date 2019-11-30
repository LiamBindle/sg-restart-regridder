from typing import List

import numpy as np
import xarray as xr

from sg.transform import make_grid_SCS
from gcpy.grid.horiz import make_grid_CS

class CSDataBase:
    def xc(self, face) -> np.ndarray:
        return self._csgrid_list[face]['lon']

    def yc(self, face) -> np.ndarray:
        return self._csgrid_list[face]['lat']

    def xe(self, face) -> np.ndarray:
        return self._csgrid_list[face]['lon_b']

    def ye(self, face) -> np.ndarray:
        return self._csgrid_list[face]['lat_b']

class CubeSphere(CSDataBase):
    def __init__(self, cs):
        _, self._csgrid_list = make_grid_CS(cs)


class StretchedGrid(CSDataBase):
    def __init__(self, cs, sf, target_lat, target_lon):
        _, self._csgrid_list = make_grid_SCS(cs, sf, target_lat, target_lon)
        self.cs = cs
        self.sf = sf
        self.target_lat = target_lat
        self.target_lon = target_lon