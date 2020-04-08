import numpy as np
import shapely.geometry
import shapely.wkb
import cartopy.crs as ccrs
import scipy.spatial
import pyproj

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


def transform_xy(in_proj: pyproj.Proj, out_proj: pyproj.Proj, xy_in):
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


def xy_to_polygons(xy, transform=ccrs.PlateCarree()):
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

    if len(bad) > 0 or len(zero_area) > 0:
        raise RuntimeError('A bad polygon was detected')
    return np.reshape(polygons, output_shape)

import shapely.strtree

def p1_intersects_in_p2(p1: np.ndarray, p2: np.ndarray):
    p1_indexes = np.array([ar.flatten() for ar in np.meshgrid(*[np.arange(s) for s in p1.shape], indexing='ij')])
    p1 = p1.flatten().tolist()
    index_by_id = dict((id(poly), i) for i, poly in enumerate(p1))
    rtree = shapely.strtree.STRtree(p1)

    def query_one_polygon(polygon):
        return [index_by_id[id(poly)] for poly in rtree.query(polygon)]

    p1_intersects = np.ndarray(shape=p2.shape, dtype=object).flatten()

    for i, poly in enumerate(p2.flatten()):
        indexes = [[dim[i] for dim in p1_indexes] for i in query_one_polygon(poly)]
        indexes = np.moveaxis(indexes, 0, -1)
        indexes = tuple(indexes)
        p1_intersects[i] = indexes

    return p1_intersects.reshape(p2.shape)






def get_am_and_pm_masks_and_polygons_outline(xe, ye, far_from_pm=80):
    if np.any(xe >= 180):
        raise ValueError('xe must be in [-180, 180)')
    # xe must be [-180 to 180]
    p0 = slice(0, -1)
    p1 = slice(1, None)

    # Mask where bounding box crosses the prime meridian or antimeridian
    cross_pm_or_am_line1 = np.not_equal(np.sign(xe[p0, p0]), np.sign(xe[p1, p0]))
    cross_pm_or_am_line2 = np.not_equal(np.sign(xe[p1, p0]), np.sign(xe[p1, p1]))
    cross_pm_or_am_line3 = np.not_equal(np.sign(xe[p1, p1]), np.sign(xe[p0, p1]))
    cross_pm_or_am_line4 = np.not_equal(np.sign(xe[p0, p1]), np.sign(xe[p0, p0]))
    cross_pm_or_am = cross_pm_or_am_line1 | cross_pm_or_am_line2 | cross_pm_or_am_line3 | cross_pm_or_am_line4

    # Make xy polygons for each gridbox
    boxes_x = np.moveaxis(np.array([xe[p0, p0], xe[p1, p0], xe[p1, p1], xe[p0, p1]]), 0, -1)
    boxes_y = np.moveaxis(np.array([ye[p0, p0], ye[p1, p0], ye[p1, p1], ye[p0, p1]]), 0, -1)
    polygon_outlines = np.moveaxis(np.array([boxes_x, boxes_y]), 0, -1)

    pm = np.ones((xe.shape[0]-1, xe.shape[1]-1), dtype=bool)
    am = np.ones((xe.shape[0]-1, xe.shape[1]-1), dtype=bool)

    # Figure out which polygon_outlines cross the prime meridian and antimeridian
    crossing_indexes = np.argwhere(cross_pm_or_am)
    for idx in crossing_indexes:
        box = shapely.geometry.LinearRing(polygon_outlines[tuple(idx)])
        far_from_the_prime_meridian = shapely.geometry.LineString([(far_from_pm, -90), (80, far_from_pm)])
        if box.crosses(far_from_the_prime_meridian):
            am[tuple(idx)] = False
        else:
            pm[tuple(idx)] = False

    return am, pm, polygon_outlines


def colocated_centers(xc1, yc1, xc2, yc2, dist_tol_abs):
    indexes1 = np.array([ar.flatten() for ar in np.meshgrid(*[np.arange(s) for s in xc1.shape], indexing='ij')])
    indexes2 = np.array([ar.flatten() for ar in np.meshgrid(*[np.arange(s) for s in xc2.shape], indexing='ij')])
    data1 = np.moveaxis([xc1.flatten(), yc1.flatten()], 0, -1)
    data2 = np.moveaxis([xc2.flatten(), yc2.flatten()], 0, -1)
    kdtree1 = scipy.spatial.KDTree(data1)
    kdtree2 = scipy.spatial.KDTree(data2)
    nearby = kdtree1.query_ball_tree(kdtree2, dist_tol_abs)

    # nearby is list of indexes in 2 for each in 1
    co1 = []
    co2 = []
    for i, neighbors in enumerate(nearby):
        for neighbor in neighbors:
            co1.append(indexes1[:,i])
            co2.append(indexes2[:,neighbor])

    # Put into format that can index original arrays
    co1 = tuple([idx_list for idx_list in np.moveaxis(np.array(co1), 0, -1)])
    co2 = tuple([idx_list for idx_list in np.moveaxis(np.array(co2), 0, -1)])
    return co1, co2

import shapely.errors

def comparable_gridboxes(control_grid, exp_grid, dist_tol_abs, area_tol_rel=None, intersect_tol_rel=None, target_lat=None, target_lon=None, inside=None):
    # Get center coordinates
    control_xc = np.array([control_grid.xc(i) for i in range(6)])
    control_yc = np.array([control_grid.yc(i) for i in range(6)])
    experiment_xc = np.array([exp_grid.xc(i) for i in range(6)])
    experiment_yc = np.array([exp_grid.yc(i) for i in range(6)])

    # Convert coordinates to equidistant projection with units of meters
    equidistant_meters = pyproj.Proj(init='epsg:4087')
    control_xc_m, control_yc_m = equidistant_meters(control_xc, control_yc)
    experiment_xc_m, experiment_yc_m = equidistant_meters(experiment_xc, experiment_yc)

    # Find colocated centers with a tolerance of 100km
    co1, co2 = colocated_centers(control_xc_m, control_yc_m, experiment_xc_m, experiment_yc_m,
                                 dist_tol_abs=dist_tol_abs)

    # Get polygons
    control_boxes_xy = np.ndarray((6, *control_xc.shape[1:], 4, 2))  # [face, i, i, pts, xy]
    experiment_boxes_xy = np.ndarray((6, *experiment_xc.shape[1:], 4, 2))  # [face, i, i, pts, xy]
    for i in range(6):
        xx = control_grid.xe(i)
        xx[xx > 180] -= 360
        _, _, control_boxes_xy[i, ...] = get_am_and_pm_masks_and_polygons_outline(xx, control_grid.ye(i))
        xx = exp_grid.xe(i)
        xx[xx > 180] -= 360
        _, _, experiment_boxes_xy[i, ...] = get_am_and_pm_masks_and_polygons_outline(xx, exp_grid.ye(i))

    # Colocated boxes
    co1_boxes_xy = control_boxes_xy[co1]
    co2_boxes_xy = experiment_boxes_xy[co2]

    # Project to equal area coordinate system with units of meters
    equal_area_meters = pyproj.Proj(f'+proj=laea +lon_0={target_lon} +lat_0={target_lat} +units=m')
    co1_boxes_x_m, co1_boxes_y_m = equal_area_meters(co1_boxes_xy[..., 0], co1_boxes_xy[..., 1])
    co1_boxes_xy_m = np.moveaxis([co1_boxes_x_m, co1_boxes_y_m], 0, -1)
    co2_boxes_x_m, co2_boxes_y_m = equal_area_meters(co2_boxes_xy[..., 0], co2_boxes_xy[..., 1])
    co2_boxes_xy_m = np.moveaxis([co2_boxes_x_m, co2_boxes_y_m], 0, -1)

    # Convert to polygons
    co1_boxes = np.array([shapely.geometry.Polygon(outline) for outline in co1_boxes_xy_m])
    co2_boxes = np.array([shapely.geometry.Polygon(outline) for outline in co2_boxes_xy_m])
    co1_box_areas = np.array([box.area for box in co1_boxes])
    co2_box_areas = np.array([box.area for box in co2_boxes])
    if area_tol_rel is not None:
        similar_areas = np.abs(co2_box_areas - co1_box_areas) / co1_box_areas < area_tol_rel
    elif intersect_tol_rel is not None:
        intersections = []
        for b1, b2 in zip(co1_boxes, co2_boxes):
            try:
                if b1.is_valid and b2.is_valid:
                    intersect_area = b1.intersection(b2).area
                else:
                    intersect_area = 0
            except shapely.errors.ShapelyError:
                intersect_area = 0
            intersections.append(intersect_area)
        intersections = np.array(intersections)
        co1_similar = np.abs(intersections - co1_box_areas) / co1_box_areas < intersect_tol_rel
        co2_similar = np.abs(intersections - co2_box_areas) / co2_box_areas < intersect_tol_rel
        similar_areas = co1_similar & co2_similar

    # Find boxes inside a polygon
    if inside:
        with open(inside, 'rb') as f:
            polygon = shapely.wkb.load(f).buffer(0.5)

        co2_ll_boxes = np.array([shapely.geometry.Polygon(outline) for outline in co2_boxes_xy])
        contained = np.array([polygon.contains(box) for box in co2_ll_boxes])
        similar_areas &= contained

    # Find boxes with similar areas
    co1_final = tuple([indexes[similar_areas] for indexes in co1])
    co2_final = tuple([indexes[similar_areas] for indexes in co2])

    return co1_final, co2_final


def quick_map(projection=ccrs.PlateCarree(), set_global=True, coastlines=True):
    plt.figure()
    ax = plt.axes(projection=projection)
    if set_global:
        ax.set_global()
    if coastlines:
        ax.coastlines(linewidth=0.5)
    return ax


def plot_polygon(ax, polygon_xy, transform, **kwargs):
    assert len(polygon_xy.shape) == 2
    ax.plot(polygon_xy[:, 0], polygon_xy[:, 1], transform=transform, **kwargs)


def plot_polygons(ax, polygons_xy, transform, **kwargs):
    if len(polygons_xy.shape) == 2:
        polygons_xy = np.expand_dims(polygons_xy, 0)

    stacked = np.product(polygons_xy.shape[:-2])
    polygons_xy = np.reshape(polygons_xy, (stacked, *polygons_xy.shape[-2:]))
    for polygon_xy in polygons_xy:
        plot_polygon(ax, polygon_xy, transform, **kwargs)


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

    minor = get_minor_xy(experiment.xe(5) % 360, experiment.ye(5))
    major = get_major_xy(experiment.xe(5) % 360, experiment.ye(5))

    major_ctl = get_major_xy(control.xe(4) % 360, control.ye(4))


    p1 = xy_to_polygons(minor)
    p2 = xy_to_polygons(major_ctl)


    p1_intersects = p1_intersects_in_p2(p1, p2)


    latlon = pyproj.Proj('+init=epsg:4326')
    laea = pyproj.Proj('+proj=laea +lat_0=-90 +lon_0=0  +x_0=0 +y_0=0 +a=6370997 +b=6370997 +units=m +no_defs')

    #foo = longest_dxdy(ctl_minor)

    gno_ccrs = ccrs.Gnomonic(30, -70)

    gno = pyproj.Proj(gno_ccrs.proj4_init)

    minor_gno = transform_xy(latlon, gno, minor)

    ax = quick_map(projection=ccrs.PlateCarree())
    plot_polygons(ax, minor[p1_intersects[0]], ccrs.PlateCarree(), color='k')
    plot_polygons(ax, major_ctl, ccrs.PlateCarree(), color='red')
    plt.show()

    # co1, co2, intersect = many_comparable_gridboxes(control, experiment)
    #
    #
    # ax = plt.axes(projection=ccrs.PlateCarree())
    # ax.set_global()
    # ax.coastlines(linewidth=0.8)
    # control_xc = np.array([control.xc(i) for i in range(6)])
    # control_yc = np.array([control.yc(i) for i in range(6)])
    # experiment_xc = experiment.xc(5)
    # experiment_yc = experiment.yc(5)
    # # experiment_xc = np.array([experiment.xc(i) for i in range(6)])
    # # experiment_yc = np.array([experiment.yc(i) for i in range(6)])
    # [draw_minor_grid_boxes(ax, control.xe(i), control.ye(i), color='blue') for i in range(6)]
    # draw_minor_grid_boxes(ax, experiment.xe(5), experiment.ye(5), linewidth=1)
    #
    # for i in co1:
    #     plt.scatter(control_xc[i], control_yc[i], color='red', s=50)
    # for i in co2:
    #     plt.scatter(experiment_xc[i], experiment_yc[i], color='pink', s=50)
    #
    # print(len(co2[0]), np.count_nonzero(co2[0]==5))
    # plt.show()

