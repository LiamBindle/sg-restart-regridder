import numpy as np
import shapely.geometry
import shapely.wkb
import cartopy.crs as ccrs
import scipy.spatial
import pyproj
import shapely.strtree
import scipy.sparse
import shapely.errors
from tqdm import tqdm
import matplotlib.pyplot as plt

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


def edge_length(xy, use_central_angle=False):
    if use_central_angle:
        diff = central_angle(xy[..., :-1, 0], xy[..., :-1, 1], xy[..., 1:, 0], xy[..., 1:, 1])
    else:
        diff = np.diff(xy, axis=-2)
    return diff


def longest_edge(xy, use_central_angle=False):
    if use_central_angle:
        diff = central_angle(xy[..., :-1, 0], xy[..., :-1, 1], xy[..., 1:, 0], xy[..., 1:, 1])
    else:
        diff = np.diff(xy, axis=-2)
    return np.max(diff)


def polygon_areas(polygons):
    areas = np.ones_like(polygons, dtype=np.float) * np.nan
    for i, polygon in enumerate(polygons):
        areas[i] = polygon.area
    return areas


def xy_to_polygons(xy, transform=None, error_on_bad_polygon=True, only_valid=False):
    if len(xy.shape) == 2:
        xy = np.expand_dims(xy, 0)

    output_shape = xy.shape[:-2]

    indexes = np.moveaxis(np.meshgrid(*[range(i) for i in xy.shape[:-2]], indexing='ij'), 0, -1)
    stacked = np.product(xy.shape[:-2])
    xy = np.reshape(xy, (stacked, *xy.shape[-2:]))
    indexes = np.reshape(indexes, (stacked, len(xy.shape[-2:])))
    polygons = []
    bad = []
    zero_area = []
    index_lut = {}
    for i, (polygon_xy, index) in enumerate(zip(xy, indexes)):
        polygon = shapely.geometry.Polygon(polygon_xy)

        is_bad = False
        if np.count_nonzero(np.isnan(polygon_xy)) > 0:
            bad.append(i)
            is_bad = True
        elif not polygon.is_valid:
            bad.append(i)
            is_bad = True
        elif polygon.area <= 0:
            zero_area.append(i)
            is_bad = True

        if not only_valid:
            polygons.append(polygon)
            index_lut[id(polygons[-1])] = tuple(index)
        elif only_valid and not is_bad:
            polygons.append(polygon)
            index_lut[id(polygons[-1])] = tuple(index)

    if error_on_bad_polygon and (len(bad) > 0 or len(zero_area) > 0):
        if transform is not None:
            ax = quick_map(projection=transform)
            draw_polygons(ax, xy[bad], transform, color='red')
            plt.show()
        raise RuntimeError('A bad polygon was detected')
    elif not only_valid and (len(bad) > 0 or len(zero_area) > 0):
        for bad_index in [*bad, *zero_area]:
            polygons[bad_index] = shapely.geometry.Polygon([(0,0), (0,0), (0,0)]) # zero area

    if only_valid:
        return index_lut, polygons
    else:
        return np.reshape(polygons, output_shape)


def central_angle(x0, y0, x1, y1):
    RAD2DEG = 180 / np.pi
    DEG2RAD = np.pi / 180

    x0 = x0 * DEG2RAD
    x1 = x1 * DEG2RAD
    y0 = y0 * DEG2RAD
    y1 = y1 * DEG2RAD

    return np.arccos(np.sin(y0) * np.sin(y1) + np.cos(y0) * np.cos(y1) * np.cos(np.abs(x0-x1))) * RAD2DEG


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


def map_gridbox_intersects(xy_in, xy_in_matches, xy_out, xy_out_ij, transform):
    ax = quick_map(transform)
    draw_polygons(ax, xy_out[xy_out_ij[0], xy_out_ij[1]], transform=transform, color='k')
    for match in xy_in_matches:
        draw_polygons(ax, xy_in[match[0], match[1]], transform=transform, color='red')
    plt.title(xy_out_ij)
    plt.show()


def determine_blocksize(xy, xc, yc):
    latlon = pyproj.Proj('+init=epsg:4326')
    N = xc.shape[0]
    factors = [i for i in range(1, N+1) if N % i == 0]

    for f in factors:
        blocksize = N // f
        nblocks = f

        blocked_shape = (nblocks, nblocks, blocksize, blocksize)

        try:
            for bi in range(nblocks):
                for bj in range(nblocks):

                    block = np.ix_(range(bi*blocksize, (bi+1)*blocksize), range(bj*blocksize, (bj+1)*blocksize))

                    xc_block = xc[block]
                    yc_block = yc[block]
                    xy_block = xy[block]

                    center_x = xc_block[blocksize//2, blocksize//2]
                    center_y = yc_block[blocksize//2, blocksize//2]

                    local_gno = pyproj.Proj(ccrs.Gnomonic(center_y, center_x).proj4_init)

                    block_xy_gno = transform_xy(xy_block, latlon, local_gno)
                    _ = xy_to_polygons(block_xy_gno, error_on_bad_polygon=True)
            return blocksize
        except RuntimeError:
            pass
    raise RuntimeError('Failed to determine the appropriate blocksize')



def ciwam2(grid_in: sg.grids.CSDataBase, grid_out: sg.grids.CSDataBase):
    # rows correspond to boxes in grid_out, columns correspond to boxes in grid_in
    latlon = pyproj.Proj('+init=epsg:4326')

    M_data = []
    M_i = []
    M_j = []

    for face_in in tqdm(range(6), desc='Input face', unit='face'):
        minor_in_ll = get_minor_xy(grid_in.xe(face_in) % 360, grid_in.ye(face_in))
        xc_in = grid_in.xc(face_in) % 360
        yc_in = grid_in.yc(face_in)
        blocksize = determine_blocksize(minor_in_ll, xc_in, yc_in)

        blocks = []
        for bi in range(grid_in.csres // blocksize):
            for bj in range(grid_in.csres // blocksize):
                blocks.append(np.ix_(
                    range(bi * blocksize, (bi + 1) * blocksize),
                    range(bj * blocksize, (bj + 1) * blocksize))
                )

        for block in tqdm(blocks, desc='Input face block', unit='block'):
            for face_out in range(6):
                minor_out_ll = get_minor_xy(grid_out.xe(face_out) % 360, grid_out.ye(face_out))

                block_x = xc_in[block][blocksize // 2, blocksize // 2]
                block_y = yc_in[block][blocksize // 2, blocksize // 2]

                gno = pyproj.Proj(ccrs.Gnomonic(block_y, block_x).proj4_init)
                laea = pyproj.Proj(
                    f'+proj=laea +lat_0={block_y} +lon_0={block_x}  +x_0=0 +y_0=0 +a=6370997 +b=6370997 +units=m +no_defs'
                )

                block_in_gno_xy = transform_xy(minor_in_ll[block], latlon, gno)
                block_in_laea_xy = transform_xy(minor_in_ll[block], latlon, laea)
                block_in_gno = xy_to_polygons(block_in_gno_xy, error_on_bad_polygon=True)
                block_in_laea = xy_to_polygons(block_in_laea_xy, error_on_bad_polygon=False)

                block_out_gno_xy = transform_xy(minor_out_ll, latlon, gno)
                block_out_laea_xy = transform_xy(minor_out_ll, latlon, laea)
                index_lut, block_out_gno = xy_to_polygons(block_out_gno_xy, error_on_bad_polygon=False, only_valid=True)
                block_out_laea = xy_to_polygons(block_out_laea_xy, error_on_bad_polygon=False)

                # search block_out_gno, in block_in_gno; whatever is queried is searched for in its extent;
                rtree = shapely.strtree.STRtree(block_out_gno)

                def draw_map():
                    crs = ccrs.Gnomonic(block_y, block_x)
                    quick_map(crs)
                    draw_polygons(block_in_gno_xy, crs, color='red', linewidth=0.8)
                    draw_polygons(block_out_gno_xy, crs, color='k', linewidth=0.5)

                for i in range(blocksize):
                    for j in range(blocksize):
                        matches = rtree.query(block_in_gno[i, j])
                        matching_indexes = [index_lut[id(matching_poly)] for matching_poly in matches if id(matching_poly) in index_lut]

                        for out_index in matching_indexes:
                            box_in = block_in_laea[i, j]
                            box_out = block_out_laea[out_index[0], out_index[1]]

                            if not box_in.is_valid or box_in.area <= 0 or not box_in.is_simple:
                                raise RuntimeError("Box in is bad and it shouldn't be")

                            if not box_out.is_valid or box_out.area <= 0 or not box_out.is_simple:
                                raise RuntimeError("Box in is bad and it shouldn't be")

                            weight = box_out.intersection(box_in).area / box_out.area

                            def draw_boxes():
                                crs = ccrs.Gnomonic(block_y, block_x)
                                draw_polygons(block_in_gno_xy[i, j, ...], crs, color='blue')
                                draw_polygons(block_out_gno_xy[out_index[0], out_index[1], ...], crs, color='g')

                            if weight > 0:
                                M_data.append(weight)

                                row_index = np.ravel_multi_index(
                                    (face_out, out_index[0], out_index[1]),
                                    (6, grid_out.csres, grid_out.csres)
                                )
                                col_index = np.ravel_multi_index(
                                    (face_in, block[0][i, 0], block[1][0, j]),
                                    (6, grid_in.csres, grid_in.csres)
                                )

                                M_i.append(row_index)
                                M_j.append(col_index)

    M = scipy.sparse.coo_matrix((M_data, (M_i, M_j)), shape=(6 * grid_out.csres ** 2, 6 * grid_in.csres ** 2))
    return M


def enhance_xy(xy, spacing=1.0):
    latlon = pyproj.Proj('+init=epsg:4326')
    edge_length = longest_edge(xy, use_central_angle=True)
    nsegs = int(edge_length / spacing)
    nsegs = max(nsegs, 2)

    gno = pyproj.Proj(ccrs.Gnomonic(xy[0, 1], xy[0, 0]).proj4_init)

    xy_gno = transform_xy(xy, latlon, gno)

    x = xy_gno[:, 0]
    y = xy_gno[:, 1]

    new_xs = []
    new_ys = []
    for x1, x2, y1, y2 in zip(x[:-1], x[1:], y[:-1], y[1:]):
        if abs(x2-x1) < 1e-6:
            new_x = np.linspace(x1, x2, nsegs)
            new_y = np.linspace(y1, y2, nsegs)
        else:
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m*x1
            new_x = np.linspace(x1, x2, nsegs)
            new_y = m*new_x + b
        new_xs.extend(new_x[:-1])
        new_ys.extend(new_y[:-1])

    new_xs.append(new_x[-1])
    new_ys.append(new_y[-1])

    xy_new = transform_xy(np.moveaxis([new_xs, new_ys], 0, -1), gno, latlon)
    return xy_new

def quick_map_polygons(transform, red_boxes, gray_boxes):
    quick_map(transform)
    for p in red_boxes:
        draw_polygons(p, transform, color='red', linewidth=0.5)
    for p in gray_boxes:
        draw_polygons(p, transform, color='gray', linewidth=0.5)

def revist(M, grid_in, grid_out, tol, start=0):
    latlon = pyproj.Proj('+init=epsg:4326')
    total_weights = M.sum(axis=-1)
    bad = np.where(total_weights < tol)

    for out_row in tqdm(bad[0][start:], desc='Revisiting rows', unit='row'):

        of, oi, oj = unravel_grid_index(grid_out, out_row)
        obox_ll = get_grid_xy(grid_out, of, oi, oj)
        xc = obox_ll[0, 0, 0]
        yc = obox_ll[0, 0, 1]

        f, i, j = intersecting_boxes(M, out_row, grid_in)
        boxes_in_ll = get_grid_xy(grid_in, f, i, j)
        enhanced_boxes_in_ll = [enhance_xy(xy) for xy in boxes_in_ll]
        laea = pyproj.Proj(
            f'+proj=laea +lat_0={yc} +lon_0={xc}  +x_0=0 +y_0=0 +a=6370997 +b=6370997 +units=m +no_defs'
        )
        iboxes_ea = [transform_xy(iebox_ll, latlon, laea) for iebox_ll in enhanced_boxes_in_ll]
        iboxes = [shapely.geometry.Polygon(xy) for xy in iboxes_ea]
        obox_ea = transform_xy(obox_ll, latlon, laea)
        obox = shapely.geometry.Polygon(obox_ea[0])

        col = [ravel_grid_index(grid_in, fi, ii, ji) for fi, ii, ji in zip(f, i, j)]
        row = np.ones_like(col, dtype=int) * out_row

        indexes = [np.argwhere((M.row == r) & (M.col == c)).item() for r, c in zip(row, col)]

        updated_intersects = [obox.intersection(ibox).area / obox.area for ibox in iboxes]

        for index, update in zip(indexes, updated_intersects):
            M.data[index] = update
    return M

def try_fix_zeros(M, grid_in, grid_out, start=0, search_dist=2):
    latlon = pyproj.Proj('+init=epsg:4326')
    total_weights = M.sum(axis=-1)
    bad = np.where(total_weights == 0.0)

    ibox_cache = {}

    M_data = M.data.tolist()
    M_i = M.row.tolist()
    M_j = M.col.tolist()
    for out_row in tqdm(bad[0][start:], desc='Revisiting rows', unit='row'):
        # This obox doesn't have any intersects
        of, oi, oj = unravel_grid_index(grid_out, out_row)
        obox_ll = get_grid_xy(grid_out, of, oi, oj)
        xc = obox_ll[0, 0, 0]
        yc = obox_ll[0, 0, 1]

        laea = pyproj.Proj(
            f'+proj=laea +lat_0={yc} +lon_0={xc}  +x_0=0 +y_0=0 +a=6370997 +b=6370997 +units=m +no_defs'
        )

        obox_ea = transform_xy(obox_ll, latlon, laea)
        obox = shapely.geometry.Polygon(obox_ea[0])

        oblock = np.meshgrid(*np.ix_(
            range(max(0, oi - search_dist), min(grid_out.csres, oi + search_dist)),
            range(max(0, oj - search_dist), min(grid_out.csres, oj + search_dist)),
        ))

        oblock = [np.ones_like(oblock[0].flatten(), dtype=int)*of, oblock[0].flatten(), oblock[1].flatten()]
        nearby_rows = ravel_grid_index(grid_out, *oblock)

        nearby_iboxes = set()
        for nearby_row in nearby_rows:
            nearby_intersects = [(f, i, j) for f, i, j in zip(*intersecting_boxes(M, nearby_row, grid_in))]
            for ibox_idx in nearby_intersects:
                nearby_iboxes.add(ibox_idx)

        for ibox_idx in nearby_iboxes:
            if ibox_idx in ibox_cache:
                continue
            else:
                ibox_ll = get_grid_xy(grid_in, *ibox_idx)
                enhanced_ibox_ll = enhance_xy(ibox_ll[0])
                ibox_cache[ibox_idx] = enhanced_ibox_ll

        for ibox_idx in nearby_iboxes:
            ibox_ll = ibox_cache[ibox_idx]
            ibox_ea = transform_xy(ibox_ll, latlon, laea)
            ibox = shapely.geometry.Polygon(ibox_ea)

            weight = obox.intersection(ibox).area / obox.area

            if weight > 0:
                col = ravel_grid_index(grid_in, *ibox_idx)
                M_data.append(weight)
                M_i.append(out_row)
                M_j.append(col)
    new_M = scipy.sparse.coo_matrix((M_data, (M_i, M_j)), shape=(6 * grid_out.csres ** 2, 6 * grid_in.csres ** 2))
    return new_M


def get_grid_xy(grid, face_indexes, i_indexes, j_indexes):
    face_indexes = np.atleast_1d(face_indexes)
    i_indexes = np.atleast_1d(i_indexes)
    j_indexes = np.atleast_1d(j_indexes)
    all_xy = [get_minor_xy(grid.xe(f) % 360, grid.ye(f)) for f in range(6)]

    xy = []
    for face, i, j in zip(face_indexes, i_indexes, j_indexes):
        xy.append(all_xy[face][i, j])

    return np.array(xy)


def ravel_grid_index(grid, face, i, j):
    return np.ravel_multi_index([face, i, j], (6, grid.csres, grid.csres))


def unravel_grid_index(grid, index):
    return np.unravel_index(index, shape=(6, grid.csres, grid.csres))


def intersecting_boxes(M, row, grid_in):
    row = np.atleast_1d(row)
    f_indexes = []
    i_indexes = []
    j_indexes = []
    for r in row:
        col = np.argwhere(M.getrow(r).toarray().squeeze() > 0).squeeze()
        f, i, j = unravel_grid_index(grid_in, col)
        f_indexes.extend(np.atleast_1d(f))
        i_indexes.extend(np.atleast_1d(i))
        j_indexes.extend(np.atleast_1d(j))
    return f_indexes, i_indexes, j_indexes


def quick_map(projection=ccrs.PlateCarree(), set_global=True, coastlines=True):
    plt.figure()
    ax = plt.axes(projection=projection)
    if set_global:
        ax.set_global()
    if coastlines:
        ax.coastlines(linewidth=0.5)
    return ax


def draw_polygons(polygons_xy, transform=ccrs.PlateCarree(), ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    if len(polygons_xy.shape) == 2:
        polygons_xy = np.expand_dims(polygons_xy, 0)

    stacked = np.product(polygons_xy.shape[:-2])
    polygons_xy = np.reshape(polygons_xy, (stacked, *polygons_xy.shape[-2:]))
    for polygon_xy in polygons_xy:
        ax.plot(polygon_xy[:, 0], polygon_xy[:, 1], transform=transform, **kwargs)


def normalize(M):
    for row in tqdm(range(M.shape[0]), desc='Normalizing rows', unit='row'):
        row_sum = M.getrow(row).sum().item()
        M.data[np.argwhere(M.row == row)] /= row_sum
    return M


if __name__=='__main__':
    import sys



    sf=2
    target_lat=35
    target_lon=-55
    # target_lat = 30
    # target_lon = -75
    # dist_tol_abs=20e3
    # area_tol_rel=0.4

    control = sg.grids.CubeSphere(90)
    experiment = sg.grids.StretchedGrid(48, sf, target_lat, target_lon)

    # M = ciwam2(experiment, control)
    M = scipy.sparse.load_npz('foo2.npz')
    total_weights = M.sum(axis=-1)
    print('Weightings:')
    print(f' mean   : {total_weights.mean()}')
    print(f' # >1.02: {np.count_nonzero(total_weights > 1.02)}/{total_weights.size}')
    print(f' # <0.98: {np.count_nonzero(total_weights < 0.98)}/{total_weights.size}')
    print(f' # zero : {np.count_nonzero(total_weights == 0)}')

    M = normalize(M)
    total_weights = M.sum(axis=-1)

    print('Weightings (post-normalize):')
    print(f' mean   : {total_weights.mean()}')
    print(f' # >1.02: {np.count_nonzero(total_weights > 1.02)}/{total_weights.size}')
    print(f' # <0.98: {np.count_nonzero(total_weights < 0.98)}/{total_weights.size}')
    print(f' # zero : {np.count_nonzero(total_weights == 0)}')

    scipy.sparse.save_npz('foo-new.npz', M)
    # exit(1)

    # M = scipy.sparse.load_npz('foo-new.npz')
    # total_weights = M.sum(axis=-1)
    # print(f'Before revisit total_weight.mean() = {total_weights.mean()}')
    #
    # M2 = revist(M, experiment, control, 0.98)
    # scipy.sparse.save_npz('foo2.npz', M)
    total_weights = M.sum(axis=-1)

    # print(f'After revisit total_weight.mean() = {total_weights.mean()}')
    # print(f'              total_weight.max() = {total_weights.max()}')
    # print(f'              total_weight.min() = {total_weights.min()}')

    row = 3087
    # f, i, j = intersecting_boxes(M, row, experiment)
    # exp_xy = get_grid_xy(experiment, face, i, j)
    ctl_xy = get_grid_xy(control, *unravel_grid_index(control, row))

    # bad = np.where(total_weights < 0.8)
    # print(len(bad[0]))
    # f, i, j = intersecting_boxes(M2, bad[0], experiment)

    # bad_boxes = get_grid_xy(experiment, f, i, j)

    quick_map(ccrs.EqualEarth())
    # draw_polygons(bad_boxes, color='red')
    draw_polygons(ctl_xy, color='red')
    plt.show()
    #
    #
    #
    # ciwam(experiment, control)
    #
    # minor = get_minor_xy(experiment.xe(5) % 360, experiment.ye(5))
    #
    # minor_exp_xy = minor
    # minor_exp = xy_to_polygons(minor_exp_xy)
    # minor_ctl_xy = get_minor_xy(control.xe(4) % 360, control.ye(4))
    # minor_ctl = xy_to_polygons(minor_ctl_xy)
    #
    # p1_intersects = p1_intersects_in_p2_extent(minor_exp, minor_ctl)
    #
    #
    # latlon = pyproj.Proj('+init=epsg:4326')
    # laea = pyproj.Proj('+proj=laea +lat_0=-90 +lon_0=0  +x_0=0 +y_0=0 +a=6370997 +b=6370997 +units=m +no_defs')
    #
    # #foo = longest_dxdy(ctl_minor)
    #
    # gno_ccrs = ccrs.Gnomonic(30, -70)
    #
    # gno = pyproj.Proj(gno_ccrs.proj4_init)
    #
    # minor_gno = transform_xy(minor, latlon, gno)
    #
    # ax = quick_map(projection=ccrs.PlateCarree())
    # draw_polygons(ax, minor_ctl_xy[22,0], ccrs.PlateCarree(), color='k')
    # draw_polygons(ax, minor_exp_xy[p1_intersects[22,0]], ccrs.PlateCarree(), color='red')
    # plt.show()
