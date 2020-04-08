import numpy as np
import shapely.geometry
import shapely.wkb
import cartopy.crs as ccrs
import scipy.spatial
import pyproj
import shapely.strtree

import sg.grids


def get_minor_xy(xe, ye):
    p0 = slice(0, -1)
    p1 = slice(1, None)
    boxes_x = np.moveaxis(np.array([xe[p0, p0], xe[p1, p0], xe[p1, p1], xe[p0, p1], xe[p0, p0]]), 0, -1)
    boxes_y = np.moveaxis(np.array([ye[p0, p0], ye[p1, p0], ye[p1, p1], ye[p0, p1], ye[p0, p0]]), 0, -1)
    return np.moveaxis(np.array([boxes_x, boxes_y]), 0, -1)


def get_major_xy(xe, ye):
    boxes_x = np.moveaxis(np.array([*xe[:-1, 0], *xe[-1, :-1], *xe[1:, -1][::-1], *xe[0, :][::-1]]), 0, -1)
    boxes_y = np.moveaxis(np.array([*ye[:-1, 0], *ye[-1, :-1], *ye[1:, -1][::-1], *ye[0, :][::-1]]), 0, -1)
    return np.moveaxis(np.array([boxes_x, boxes_y]), 0, -1)


def transform_xy(xy_in, in_proj: pyproj.Proj, out_proj: pyproj.Proj):
    xy_out = pyproj.transform(in_proj, out_proj, xy_in[..., 0], xy_in[..., 1])
    xy_out = np.moveaxis(xy_out, 0, -1)
    return xy_out


def longest_edge(xy):
    return np.max(np.diff(xy, axis=-2))


def polygon_areas(polygons):
    areas = np.ones_like(polygons, dtype=np.float) * np.nan
    for i, polygon in enumerate(polygons):
        areas[i] = polygon.area
    return areas


def xy_to_polygons(xy, transform=None, error_on_bad_polygon=True):
    if len(xy.shape) == 2:
        xy = np.expand_dims(xy, 0)

    output_shape = xy.shape[:-2]
    stacked = np.product(xy.shape[:-2])
    xy = np.reshape(xy, (stacked, *xy.shape[-2:]))
    polygons = np.ndarray((stacked,), dtype=object)
    bad = []
    zero_area = []
    for i, polygon_xy in enumerate(xy):
        polygons[i] = shapely.geometry.Polygon(polygon_xy)
        if not polygons[i].is_valid:
            bad.append(i)
        if polygons[i].area <= 0:
            zero_area.append(i)

    if error_on_bad_polygon and (len(bad) > 0 or len(zero_area) > 0):
        if transform is not None:
            ax = quick_map(projection=transform)
            draw_polygons(ax, xy[bad], transform, color='red')
            plt.show()
        raise RuntimeError('A bad polygon was detected')
    elif len(bad) > 0 or len(zero_area) > 0:
        for bad_index in [*bad, *zero_area]:
            polygons[bad_index] = shapely.geometry.Polygon([(0,0), (0,0), (0,0)]) # zero area
    return np.reshape(polygons, output_shape)


def p1_intersects_in_p2_extent(p1: np.ndarray, p2: np.ndarray, return_slices=True):
    p1_indexes = np.array([ar.flatten() for ar in np.meshgrid(*[np.arange(s) for s in p1.shape], indexing='ij')])
    p1 = p1.flatten().tolist()
    index_by_id = dict((id(poly), i) for i, poly in enumerate(p1))
    rtree = shapely.strtree.STRtree(p1)

    def query_one_polygon(polygon):
        return [index_by_id[id(poly)] for poly in rtree.query(polygon)]

    p1_intersects = np.ndarray(shape=p2.shape, dtype=object).flatten()

    for i, poly in enumerate(p2.flatten()):
        indexes = [[dim[i] for dim in p1_indexes] for i in query_one_polygon(poly)]
        if return_slices:
            indexes = np.moveaxis(indexes, 0, -1)
            indexes = tuple(indexes)
        p1_intersects[i] = indexes

    return p1_intersects.reshape(p2.shape)

import scipy.sparse
from tqdm import tqdm

def ciwam(grid_in: sg.grids.CSDataBase, grid_out: sg.grids.CSDataBase):
    # rows correspond to boxes in grid_out, columns correspond to boxes in grid_in
    latlon = pyproj.Proj('+init=epsg:4326')

    M_data = []
    M_i = []
    M_j = []

    flat_index = lambda grid, f, i, j: f*(grid.csres**2) + i*grid.csres + j

    for face_out in tqdm(range(6), desc='Output face', unit='face'):
        minor_out_ll = get_minor_xy(grid_out.xe(face_out) % 360, grid_out.ye(face_out))

        center_x = (grid_out.xe(face_out) % 360)[grid_out.csres//2, grid_out.csres//2]
        center_y = grid_out.ye(face_out)[grid_out.csres // 2, grid_out.csres // 2]
        laea = pyproj.Proj(
            f'+proj=laea +lat_0={center_y} +lon_0={center_x}  +x_0=0 +y_0=0 +a=6370997 +b=6370997 +units=m +no_defs'
        )

        minor_out_ea = transform_xy(minor_out_ll, latlon, laea)
        minor_out = xy_to_polygons(minor_out_ea)

        for face_in in tqdm(range(6), desc='Input face', unit='face'):
            minor_in_ll = get_minor_xy(grid_in.xe(face_in) % 360, grid_in.ye(face_in))
            minor_in_ea = transform_xy(minor_in_ll, latlon, laea)
            minor_in = xy_to_polygons(minor_in_ea, error_on_bad_polygon=False) #transform=ccrs.LambertAzimuthalEqualArea(center_x, center_y))

            minor_in_indexes = p1_intersects_in_p2_extent(minor_in, minor_out, return_slices=False)

            for i in range(grid_out.csres):
                for j in range(grid_out.csres):
                    row_index = flat_index(grid_out, face_out, i, j)
                    for indexes in minor_in_indexes[i, j]:
                        col_index = flat_index(grid_in, face_in, indexes[0], indexes[1])

                        gridbox_out = minor_out[i, j]
                        gridbox_in = minor_in[indexes[0], indexes[1]]

                        weight = gridbox_out.intersection(gridbox_in).area / gridbox_out.area

                        if weight > 0:
                            M_data.append(weight)
                            M_i.append(row_index)
                            M_j.append(col_index)

    M = scipy.sparse.coo_matrix((M_data, (M_i, M_j)), shape=(6 * grid_out.csres ** 2, 6 * grid_in.csres ** 2))

    # QA
    weight_sum = M.sum(axis=1)

    return M










def quick_map(projection=ccrs.PlateCarree(), set_global=True, coastlines=True):
    plt.figure()
    ax = plt.axes(projection=projection)
    if set_global:
        ax.set_global()
    if coastlines:
        ax.coastlines(linewidth=0.5)
    return ax


def draw_polygons(ax, polygons_xy, transform, **kwargs):
    if len(polygons_xy.shape) == 2:
        polygons_xy = np.expand_dims(polygons_xy, 0)

    stacked = np.product(polygons_xy.shape[:-2])
    polygons_xy = np.reshape(polygons_xy, (stacked, *polygons_xy.shape[-2:]))
    for polygon_xy in polygons_xy:
        ax.plot(polygon_xy[:, 0], polygon_xy[:, 1], transform=transform, **kwargs)


if __name__=='__main__':
    import matplotlib.pyplot as plt
    import sys



    sf=12
    target_lat=35
    target_lon=-55
    # target_lat = 30
    # target_lon = -75
    dist_tol_abs=20e3
    area_tol_rel=0.4

    control = sg.grids.CubeSphere(24)
    experiment = sg.grids.StretchedGrid(12, sf, target_lat, target_lon)

    ciwam(experiment, control)

    minor = get_minor_xy(experiment.xe(5) % 360, experiment.ye(5))

    minor_exp_xy = minor
    minor_exp = xy_to_polygons(minor_exp_xy)
    minor_ctl_xy = get_minor_xy(control.xe(4) % 360, control.ye(4))
    minor_ctl = xy_to_polygons(minor_ctl_xy)

    p1_intersects = p1_intersects_in_p2_extent(minor_exp, minor_ctl)


    latlon = pyproj.Proj('+init=epsg:4326')
    laea = pyproj.Proj('+proj=laea +lat_0=-90 +lon_0=0  +x_0=0 +y_0=0 +a=6370997 +b=6370997 +units=m +no_defs')

    #foo = longest_dxdy(ctl_minor)

    gno_ccrs = ccrs.Gnomonic(30, -70)

    gno = pyproj.Proj(gno_ccrs.proj4_init)

    minor_gno = transform_xy(minor, latlon, gno)

    ax = quick_map(projection=ccrs.PlateCarree())
    draw_polygons(ax, minor_ctl_xy[22,0], ccrs.PlateCarree(), color='k')
    draw_polygons(ax, minor_exp_xy[p1_intersects[22,0]], ccrs.PlateCarree(), color='red')
    plt.show()
