import numpy as np
import shapely.geometry
import cartopy.crs as ccrs
import scipy.spatial


def draw_minor_grid_boxes(ax, xx, yy, **kwargs):
    kwargs.setdefault('linewidth', 0.5)
    kwargs.setdefault('color', '#151515')
    for x, y in zip(xx, yy):
        idx = np.argwhere(np.diff(np.sign(x % 360 - 180))).flatten()
        x360 = x % 360
        idx = idx[(x360[idx] > 10) & (x360[idx] < 350)]
        start = [0, *(idx + 1)]
        end = [*(idx + 1), len(x)]
        for s, e in zip(start, end):
            ax.plot(x[s:e], y[s:e], transform=ccrs.PlateCarree(), **kwargs)
    for x, y in zip(xx.transpose(), yy.transpose()):
        idx = np.argwhere(np.diff(np.sign(x % 360 - 180))).flatten()
        x360 = x % 360
        idx = idx[(x360[idx] > 10) & (x360[idx] < 350)]
        start = [0, *(idx + 1)]
        end = [*(idx + 1), len(x)]
        for s, e in zip(start, end):
            ax.plot(x[s:e], y[s:e], transform=ccrs.PlateCarree(), **kwargs)

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

if __name__=='__main__':
    import sg.grids
    import matplotlib.pyplot as plt
    import pyproj

    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    ax.coastlines(linewidth=0.8)

    dist_tol_abs=50e3
    area_tol_rel=0.4

    control = sg.grids.CubeSphere(48)
    target_lat=33.7
    target_lon=275.6
    experiment = sg.grids.StretchedGrid(24, 2, target_lat, target_lon)

    # Get center coordinates
    control_xc = np.array([control.xc(i) for i in range(6)])
    control_yc = np.array([control.yc(i) for i in range(6)])
    experiment_xc = np.array([experiment.xc(i) for i in range(6)])
    experiment_yc = np.array([experiment.yc(i) for i in range(6)])

    # Convert coordinates to equidistant projection with units of meters
    equidistant_meters = pyproj.Proj(init='epsg:4087')
    control_xc_m, control_yc_m = equidistant_meters(control_xc, control_yc)
    experiment_xc_m, experiment_yc_m = equidistant_meters(experiment_xc, experiment_yc)

    # Find colocated centers with a tolerance of 100km
    co1, co2 = colocated_centers(control_xc_m, control_yc_m, experiment_xc_m, experiment_yc_m, dist_tol_abs=dist_tol_abs)

    # Get polygons
    control_boxes_xy = np.ndarray((6, *control_xc.shape[1:], 4, 2))          # [face, i, i, pts, xy]
    experiment_boxes_xy = np.ndarray((6, *experiment_xc.shape[1:], 4, 2))    # [face, i, i, pts, xy]
    for i in range(6):
        xx = control.xe(i)
        xx[xx > 180] -= 360
        _, _, control_boxes_xy[i, ...] = get_am_and_pm_masks_and_polygons_outline(xx, control.ye(i))
        xx = experiment.xe(i)
        xx[xx > 180] -= 360
        _, _, experiment_boxes_xy[i, ...] = get_am_and_pm_masks_and_polygons_outline(xx, experiment.ye(i))

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
    co1_box_areas = np.array([shapely.geometry.Polygon(outline).area for outline in co1_boxes_xy_m])
    co2_box_areas = np.array([shapely.geometry.Polygon(outline).area for outline in co2_boxes_xy_m])

    # Find boxes with similar areas
    similar_areas = np.abs(co2_box_areas - co1_box_areas) / co1_box_areas < area_tol_rel

    [draw_minor_grid_boxes(ax, control.xe(i), control.ye(i), color='blue') for i in range(6)]
    draw_minor_grid_boxes(ax, experiment.xe(5), experiment.ye(5))



    plt.scatter(control_xc[co1][similar_areas], control_yc[co1][similar_areas], color='red', s=50)
    plt.scatter(experiment_xc[co2][similar_areas], experiment_yc[co2][similar_areas], color='pink', s=50)

    print(co1, co2)
    plt.show()

