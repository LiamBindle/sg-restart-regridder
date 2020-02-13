from typing import List

import numpy as np
import xarray as xr
import pyproj
import cartopy.crs as ccrs
import shapely.geometry


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

    def _make_rings(self):
        self._face_polygon = 6 * [None]
        self._gnomonic_proj = 6 * [None]
        for i in range(6):
            xe = self.xe(i)
            ye = self.ye(i)
            center = ((np.array(xe.shape) + 0.5) / 2).astype(int)
            xc = xe[center[0], center[1]]
            yc = ye[center[0], center[1]]
            self._gnomonic_proj[i] = pyproj.Proj(ccrs.Gnomonic(central_longitude=xc, central_latitude=yc).proj4_init)
            ring_x, ring_y = self._gnomonic_proj[i](
                np.array([xe[0, 0], xe[0, -1], xe[-1, -1], xe[-1, 0]]),
                np.array([ye[0, 0], ye[0, -1], ye[-1, -1], ye[-1, 0]])
            )
            self._face_polygon[i] = shapely.geometry.Polygon([*zip(ring_x[::-1], ring_y[::-1])])

    def get_face(self, xx, yy):
        def find_face(lon, lat):
            for i in range(6):
                pt = shapely.geometry.Point(self._gnomonic_proj[i](lon, lat))
                if self._face_polygon[i].contains(pt):
                    return i
            raise ValueError(f'Point(lon={lon}, lat={lat}) did not fall within the bounds of a face!')
        face_index = np.zeros_like(xx, dtype=int)
        v_find_faces = np.vectorize(find_face)
        face_index = v_find_faces(xx, yy)
        return  face_index

class CubeSphere(CSDataBase):
    def __init__(self, cs):
        _, self._csgrid_list = make_grid_CS(cs)
        #self._make_rings()


class StretchedGrid(CSDataBase):
    def __init__(self, cs, sf, target_lat, target_lon):
        _, self._csgrid_list = make_grid_SCS(cs, sf, target_lat, target_lon)
        self.cs = cs
        self.sf = sf
        self.target_lat = target_lat
        self.target_lon = target_lon
        #self._make_rings()


if __name__ == '__main__':
    grid = CubeSphere(48)

    lon, lat = np.meshgrid(np.linspace(0, 360, 200), np.linspace(-90, 90, 200))
    faces = grid.get_face(lon, lat)
    #print(faces)

    import matplotlib.pyplot as plt
    import sg.plot
    import sg.figure_axes
    plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    ax.coastlines()
    figax = sg.figure_axes.FigureAxes(ax, projection=ccrs.PlateCarree())
    for i in range(6):
        sg.plot.draw_major_grid_boxes(figax, *figax.transform_xy(grid.xe(i), grid.ye(i)), color='black')
        data = xr.DataArray(faces)
        sg.plot.plot_pcolomesh(figax, *figax.transform_xy(lon, lat), data)
    plt.show()