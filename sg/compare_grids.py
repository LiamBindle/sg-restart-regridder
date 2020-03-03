import numpy as np
import shapely.geometry
import shapely.wkb
import cartopy.crs as ccrs
import scipy.spatial
import pyproj

import sg.grids


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

def many_comparable_gridboxes(control_grid, exp_grid: sg.grids.StretchedGrid):
    # Get center coordinates
    control_xc = np.array([control_grid.xc(i) for i in range(6)])
    control_yc = np.array([control_grid.yc(i) for i in range(6)])
    experiment_xc = exp_grid.xc(5)
    experiment_yc = exp_grid.yc(5)

    control_xc[control_xc < 0] += 360
    experiment_xc[experiment_xc < 0] += 360

    res = control_xc[0].shape[0]
    co1, co2 = colocated_centers(control_xc, control_yc, experiment_xc, experiment_yc,
                                 dist_tol_abs=360/(4*res)/2*np.sqrt(2.3)*2
                                 )

    # Get polygons
    control_boxes_xy = np.ndarray((6, *control_xc.shape[1:], 4, 2))  # [face, i, i, pts, xy]
    for i in range(6):
        xx = control_grid.xe(i)
        xx[xx > 180] -= 360
        _, _, control_boxes_xy[i, ...] = get_am_and_pm_masks_and_polygons_outline(xx, control_grid.ye(i))
    xx = exp_grid.xe(5)
    xx[xx > 180] -= 360
    _, _, experiment_boxes_xy = get_am_and_pm_masks_and_polygons_outline(xx, exp_grid.ye(5))

    # Colocated boxes
    co1_boxes_xy = control_boxes_xy[co1]
    co2_boxes_xy = experiment_boxes_xy[co2]


    # Get weighting (intersection)

    # Project to equal area coordinate system with units of meters
    equal_area_meters = pyproj.Proj(f'+proj=laea +lon_0={exp_grid.target_lon} +lat_0={exp_grid.target_lat} +units=m')
    co1_boxes_xy_m = np.moveaxis(equal_area_meters(co1_boxes_xy[..., 0], co1_boxes_xy[..., 1]), 0, -1)
    co2_boxes_xy_m = np.moveaxis(equal_area_meters(co2_boxes_xy[..., 0], co2_boxes_xy[..., 1]), 0, -1)

    co1_boxes = np.array([shapely.geometry.Polygon(outline) for outline in co1_boxes_xy_m])
    co2_boxes = np.array([shapely.geometry.Polygon(outline) for outline in co2_boxes_xy_m])
    intersect_rel = []
    for b1, b2 in zip(co1_boxes, co2_boxes):
        try:
            if b1.is_valid and b2.is_valid:
                intersect = b1.intersection(b2).area / b1.area
            else:
                intersect = 0
        except shapely.errors.ShapelyError:
            intersect = 0
        intersect_rel.append(intersect)

    keep = np.array(intersect_rel) > 0

    # Find boxes with similar areas
    co1 = tuple([indexes[keep] for indexes in co1])
    co2 = tuple([indexes[keep] for indexes in co2])
    intersect = np.array(intersect_rel)[keep]

    prev = None
    co1_unique = []
    co2_match = []
    weight = []
    for idx1, idx2, overlap in zip(zip(*co1), zip(*co2), intersect):
        if idx1 != prev:
            co1_unique.append(idx1)
            co2_match.append([])
            weight.append([])
        weight[-1].append(overlap)
        co2_match[-1].append(idx2)
        prev = idx1

    # only keep boxes with total weight equal to approximately 1
    total_weight = np.array([np.sum(weights) for weights in weight])

    keep = total_weight > 0.95
    co1 = [c for c, k in zip(co1_unique, keep) if k]
    co2 = [tuple(np.array(c).transpose()) for c, k in zip(co2_match, keep) if k]
    weights = [np.array(w) for w, k in zip(weight, keep) if k]

    return co1, co2, weights


cs_bank = {}

def minimize_objective(sf, target_lat, target_lon, cs_res=48, sf_res=24, dist_tol_abs=40e3, intersect_tol_rel=0.35):
    sf = np.asscalar(np.array(sf))
    target_lat = np.asscalar(np.array(target_lat))
    target_lon = np.asscalar(np.array(target_lon))
    if cs_res in cs_bank:
        cs = cs_bank[cs_res]
    else:
        cs = sg.grids.CubeSphere(cs_res)
        cs_bank[cs_res] = cs
    _, comparable = comparable_gridboxes(
        sg.grids.CubeSphere(cs_res),
        sg.grids.StretchedGrid(sf_res, sf, target_lat, target_lon),
        dist_tol_abs=dist_tol_abs, intersect_tol_rel=intersect_tol_rel,
        target_lat=target_lat,
        target_lon=target_lon
    )
    print([sf_res, target_lat, target_lon], -np.count_nonzero(comparable[0] == 5))
    sys.stdout.flush()
    return -np.count_nonzero(comparable[0] == 5)


if __name__=='__main__':
    import matplotlib.pyplot as plt
    import sys


    import scipy.optimize

    # # Global settings
    # cs_res = 360
    # sf = float(sys.argv[1])
    # max_aspect_ratio = 1.3      # center box area / edge box area
    # ll_pm = (360/(cs_res*4))/2  # lat-lon plus/minus around initial guess, half the representative grid-box width
    # dist_tol = 50e3
    # intersect_tol = 0.4
    #
    # # Initial guesses
    # target_lat = float(sys.argv[2])
    # target_lon = float(sys.argv[3])
    #
    # print(f'optimizing: sf={sf}, tlat={target_lat}, tlon={target_lon}')
    #
    # # Optimize the stretch factor for matching box areas
    # dist_tol_abs=2*dist_tol         # moderate distance tolerance
    # intersect_tol=intersect_tol     # small area tolerance
    #
    # center = cs_res/sf
    # sg_res_range = np.arange(int(center/sf**0.5 + 0.5)//2*2, int(center*sf**0.5 + 0.51)//2*2, step=2, dtype=int)
    #
    # count = np.zeros_like(sg_res_range)
    # for i, sg_res in enumerate(sg_res_range):
    #     count[i] = minimize_objective(sf, target_lat, target_lon, cs_res, sg_res, dist_tol_abs=dist_tol_abs, intersect_tol_rel=intersect_tol)
    # sg_res_opt = sg_res_range[np.argmin(count)]
    #
    # # Optimize the position factor
    # dist_tol_abs=dist_tol           # smaller distance tolerance
    # intersect_tol=intersect_tol     # small intersect tolerance
    # lat_range=(target_lat-ll_pm, target_lat+ll_pm)
    # lon_range=(target_lon-ll_pm, target_lon+ll_pm)
    # lat_opt, lon_opt = scipy.optimize.brute(
    #     lambda x: minimize_objective(
    #         target_lat=x[0],
    #         target_lon=x[1],
    #         sf=sf,
    #         cs_res=cs_res,
    #         sf_res=int(sg_res_opt),
    #         dist_tol_abs=dist_tol_abs,
    #         intersect_tol_rel=intersect_tol
    #     ),
    #     ranges=[lat_range, lon_range],
    #     Ns=11,
    #     finish=None
    # )
    #
    # print(f'sf={sf}, sg_res={sg_res_opt}, lat={lat_opt}, lon={lon_opt}')


    sf=8
    target_lat=33.7
    target_lon=275.5
    dist_tol_abs=20e3
    area_tol_rel=0.4

    control = sg.grids.CubeSphere(24)
    experiment = sg.grids.StretchedGrid(12, sf, target_lat, target_lon)
    co1, co2, intersect = many_comparable_gridboxes(control, experiment)


    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    ax.coastlines(linewidth=0.8)
    control_xc = np.array([control.xc(i) for i in range(6)])
    control_yc = np.array([control.yc(i) for i in range(6)])
    experiment_xc = experiment.xc(5)
    experiment_yc = experiment.yc(5)
    # experiment_xc = np.array([experiment.xc(i) for i in range(6)])
    # experiment_yc = np.array([experiment.yc(i) for i in range(6)])
    [draw_minor_grid_boxes(ax, control.xe(i), control.ye(i), color='blue') for i in range(6)]
    draw_minor_grid_boxes(ax, experiment.xe(5), experiment.ye(5), linewidth=1)

    for i in co1:
        plt.scatter(control_xc[i], control_yc[i], color='red', s=50)
    for i in co2:
        plt.scatter(experiment_xc[i], experiment_yc[i], color='pink', s=50)

    print(len(co2[0]), np.count_nonzero(co2[0]==5))
    plt.show()

