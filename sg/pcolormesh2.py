import numpy as np
import cartopy.crs as ccrs
import pyproj
import matplotlib.pyplot as plt
import shapely.geometry
import argparse
import os.path


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


def _pcolormesh2_internal(ax, X, Y, C, cmap, norm):
    X[X >= 180] -= 360

    am, pm, boxes_xy_pc = get_am_and_pm_masks_and_polygons_outline(X, Y)

    center_i = int(X.shape[0] / 2)
    center_j = int(X.shape[1] / 2)
    cX = X[center_i, center_j]
    cY = Y[center_i, center_j]

    gnomonic_crs = ccrs.Gnomonic(cY, cX)
    gnomonic_proj = pyproj.Proj(gnomonic_crs.proj4_init)

    X_gno, Y_gno = gnomonic_proj(X, Y)
    boxes_xy_gno = np.moveaxis(gnomonic_proj(boxes_xy_pc[..., 0], boxes_xy_pc[..., 1]), 0, -1)

    if np.any(np.isnan(X_gno)) or np.any(np.isnan(Y_gno)):
        raise ValueError('Block size is too big!')
    else:
        plt.pcolormesh(X_gno, Y_gno, np.ma.masked_array(C, ~am), transform=gnomonic_crs, cmap=cmap, norm=norm)

        for idx in np.argwhere(~am):
            c = cmap(norm(C[idx[0], idx[1]]))
            ax.add_geometries(
                [shapely.geometry.Polygon(boxes_xy_gno[idx[0], idx[1],...])],
                gnomonic_crs, edgecolor=c, facecolor=c
            )


def pcolormesh2(X, Y, C, blocksize, norm, **kwargs):
    kwargs.setdefault('cmap', 'viridis')
    cmap = plt.get_cmap(kwargs['cmap'])

    ax = plt.gca()

    for si, ei in [(s * blocksize, (s + 1) * blocksize + 1) for s in range(X.shape[0] // blocksize)]:
        for sj, ej in [(s * blocksize, (s + 1) * blocksize + 1) for s in range(X.shape[1] // blocksize)]:
            _pcolormesh2_internal(ax,
                X[si:ei, sj:ej],
                Y[si:ei, sj:ej],
                C[si:ei - 1, sj:ej - 1],
                cmap, norm
            )


if __name__ == '__main__':
    import xarray as xr
    import yaml
    from sg.grids import CubeSphere, StretchedGrid
    import  matplotlib.cm
    parser = argparse.ArgumentParser(description='Make a map')

    parser.add_argument('-i',
                        type=str,
                        nargs='+',
                        required=True,
                        help="output_file")
    parser.add_argument('-s',
                        metavar='L',
                        type=str,
                        nargs='+',
                        help='model level')
    parser.add_argument('-v',
                        metavar='V',
                        nargs='+',
                        type=str,
                        required=True,
                        help='path to the control\'s output directory')
    parser.add_argument('-c',
                        metavar='C',
                        type=str,
                        required=True,
                        help='conf file')
    parser.add_argument('-o',
                        metavar='O',
                        type=str,
                        default='output.png',
                        help='path to output')
    parser.add_argument('-w',
                        metavar='W',
                        type=int,
                        help='the hard face')
    parser.add_argument('-b',
                        metavar='B',
                        type=int,
                        help='block size')
    parser.add_argument('-a',
                        metavar='A',
                        nargs='+',
                        type=str,
                        choices=['sum', 'diff', 'div'])
    args = vars(parser.parse_args())
    plt.rc('text', usetex=False)

    if len(args['i']) > 1:
        if len(args['v']) != len(args['i']):
            raise ValueError('number of var arguments must be equal to the number of output files')
        if len(args['v']) - 1 != len(args['a']):
            raise ValueError('number of actions must be one less than the number of vars')

    # Load data
    data = []
    face_num = None
    for output_file, variable_name in zip(args['i'], args['v']):
        da = xr.open_dataset(os.path.join(output_file))[variable_name]
        for k, v in zip(args['s'][::2], args['s'][1::2]):
            if k == 'nf' or k =='face':
                face_num = int(v)
            da = da.isel(**{k: int(v)})
        da = da.squeeze()
        data.append(da)

    # Do actions
    total = data[0].copy()
    cmap = 'viridis'
    for action, da in zip(args['a'], data[1:]):
        if action == 'sum':
            total += da
        elif action == 'diff':
            total -= da
            cmap= 'RdBu_r'
        elif action == 'div':
            total /= da
            cmap = 'RdBu_r'

    # Make plot
    norm = plt.Normalize(vmin=total.quantile(0.05), vmax=total.quantile(0.95))

    with open(args['c'], 'r') as f:
        conf = yaml.safe_load(f)

    if 'stretch_factor' in conf['grid']:
        grid = StretchedGrid(
            conf['grid']['cs_res'],
            conf['grid']['stretch_factor'],
            conf['grid']['target_lat'],
            conf['grid']['target_lon'],
        )
    else:
        grid = CubeSphere(
            conf['grid']['cs_res'],
        )

    face_indexes = None
    if 'nf' in total.dims:
        face_indexes = total['nf'].values
    elif 'face' in total.dims:
        face_indexes = total['face'].values
    else:
        face_indexes = [face_num]

    def draw_major_grid_boxes_naive(ax, xx, yy, **kwargs):
        kwargs.setdefault('color', 'k')
        kwargs.setdefault('linewidth', 0.8)
        xx_majors = [xx[0, :], xx[-1, :], xx[:, 0], xx[:, -1]]
        yy_majors = [yy[0, :], yy[-1, :], yy[:, 0], yy[:, -1]]
        for xm, ym in zip(xx_majors, yy_majors):
            ax.plot(xm, ym, transform=ccrs.PlateCarree(), **kwargs)

    plt.figure(figsize=(8,6))
    ax = plt.axes(projection=ccrs.EqualEarth())
    ax.coastlines()
    ax.set_global()

    for face in face_indexes:
        if 'nf' in total.dims:
            da = total.sel(nf=face)
        elif 'face' in total.dims:
            da = total.sel(face=face)
        else:
            da = total

        if face == args['w']:
           pcolormesh2(grid.xe(face-1), grid.ye(face-1), da, args['b'], norm, cmap=cmap)
        else:
            pcolormesh2(grid.xe(face-1), grid.ye(face-1), da, conf['grid']['cs_res'], norm, cmap=cmap)
        draw_major_grid_boxes_naive(plt.gca(), grid.xe(face-1), grid.ye(face-1))

    plt.colorbar(matplotlib.cm.ScalarMappable(norm, cmap), orientation='horizontal')
    plt.tight_layout()
    plt.savefig(args['o'], dpi=300)