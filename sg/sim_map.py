import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import shapely.geometry
import shapely.errors
import pyproj

import sg.grids
import sg.plot

import tqdm

def draw_target_face_outline(ax: plt.Axes, sf, tlat, tlon, color, linewidth=1.5):
    grid = sg.grids.StretchedGrid(24, sf, tlat, tlon)
    xx = grid.xe(5)
    yy = grid.ye(5)
    xx_majors = [xx[0, :], xx[-1, :], xx[:, 0], xx[:, -1]]
    yy_majors = [yy[0, :], yy[-1, :], yy[:, 0], yy[:, -1]]
    for xm, ym in zip(xx_majors, yy_majors):
        style = '-' if linewidth == 1.5 else '--'
        ax.plot(xm, ym, transform=ccrs.PlateCarree(), color=color, linewidth=linewidth, linestyle=style)


def equal_extent_interpolation(sf0, sf1, dtfal):
    sf_y = np.linspace(20, 1, 100)
    tfal = lambda sf: np.pi - 2 * np.arcsin(1 - 2/((3+2*np.sqrt(2))*sf**2 + 1))

    tfal_x = tfal(sf_y)

    tfal1 = tfal(sf1)
    tfal0 = tfal(sf0)
    n = int(np.ceil((tfal0 - tfal1)/dtfal))
    x = np.linspace(tfal(sf0), tfal(sf1), n)
    sf_linspaced = np.interp(x, tfal_x, sf_y)

    return sf_linspaced

def fill_space(sf0, sf1, tlat0, tlat1, tlon0, tlon1, dtfal, **kwargs):
    sf = equal_extent_interpolation(sf0, sf1, dtfal)
    tlat = np.linspace(tlat0, tlat1, len(sf))
    tlon = np.linspace(tlon0, tlon1, len(sf))

    return [{'sf': sf[i], 'tlat': tlat[i], 'tlon': tlon[i], **kwargs} for i in range(1,len(sf)-1)]


def count_inscribed_boxes(control_grid, exp_grid: sg.grids.StretchedGrid):
    xx = exp_grid.xe(5)
    yy = exp_grid.ye(5)

    # Project to orthographic
    to_gnomonic = pyproj.Proj(ccrs.Gnomonic(
        central_longitude=np.mean(exp_grid.target_lon),
        central_latitude=np.mean(exp_grid.target_lat)
    ).proj4_init)

    xx, yy = to_gnomonic(xx, yy)

    target_face_xy = np.moveaxis(np.array([
        [xx[0, 0], xx[-1, 0], xx[-1, -1], xx[0, -1]],
        [yy[0, 0], yy[-1, 0], yy[-1, -1], yy[0, -1]]
    ]), 0, -1)

    target_face_polygon = shapely.geometry.Polygon(target_face_xy)

    count = 0
    for i in range(6):
        ctrl_xx = control_grid.xc(i)
        ctrl_yy = control_grid.yc(i)

        ctrl_xx, ctrl_yy = to_gnomonic(ctrl_xx, ctrl_yy)

        ctrl_xx = ctrl_xx.flatten()
        ctrl_yy = ctrl_yy.flatten()

        for x, y in zip(ctrl_xx, ctrl_yy):
            try:
                if target_face_polygon.contains(shapely.geometry.Point([x, y])):
                    count += 1
            except shapely.errors.ShapelyError:
                pass
    return count

cs_cache = {}
def find_optimal_res(cs_res, sf, tlat, tlon):
    if cs_res in cs_cache:
        cs = cs_cache[cs_res]
    else:
        cs = sg.grids.CubeSphere(cs_res)
        cs_cache[cs_res] = cs

    n = count_inscribed_boxes(cs, sg.grids.StretchedGrid(12, sf, tlat, tlon))
    n_side = np.sqrt(n)
    n_side = np.ceil(n_side/2)*2
    return int(n_side)




if __name__ == '__main__':

    cmap = plt.get_cmap('tab10')
    dtfal = 4*np.pi/180

    grids = [
        {'sf': 2, 'tlat': 35, 'tlon': 264, 'color': cmap(0)},
        *fill_space(2, 3.6, 35, 38, 264, 252, dtfal, **{'color': cmap(0), 'linewidth': 0.6}), # C180-C360
        {'sf': 3.6, 'tlat': 38, 'tlon': 252, 'color': cmap(0)},
        *fill_space(3.6, 6.8, 38, 37, 252, 244, dtfal, **{'color': cmap(0), 'linewidth': 0.6}), # C360-C720
        {'sf': 6.8, 'tlat': 37, 'tlon': 244, 'color': cmap(0)},
        *fill_space(6.8, 12.5, 37, 36, 244, 241, dtfal, **{'color': cmap(0), 'linewidth': 0.6}),
        {'sf': 12.5, 'tlat': 36, 'tlon': 241, 'color': cmap(0)},

        {'sf': 2, 'tlat': 48, 'tlon': 14, 'color': cmap(1)},
        *fill_space(2, 3.4, 48, 47, 14, 5, dtfal, **{'color': cmap(1), 'linewidth': 0.6}),
        {'sf': 3.4, 'tlat': 47, 'tlon': 5, 'color': cmap(1)},
        *fill_space(3.4, 6.8, 47, 42.5, 5, 12.5, dtfal, **{'color': cmap(1), 'linewidth': 0.6}),
        {'sf': 6.8, 'tlat': 42.5, 'tlon': 12.5, 'color': cmap(1)},
        *fill_space(6.8, 15, 42.5, 45, 12.5, 10.5, dtfal, **{'color': cmap(1), 'linewidth': 0.6}),
        {'sf': 15, 'tlat': 45, 'tlon': 10.5, 'color': cmap(1)},

        {'sf': 2.8, 'tlat': 21.5, 'tlon': 79, 'color': cmap(2)},
        *fill_space(2.8, 6, 21.5, 25, 79, 81, dtfal, **{'color': cmap(2), 'linewidth': 0.6}),
        {'sf': 6, 'tlat': 25, 'tlon': 81, 'color': cmap(2)},
        *fill_space(6, 14, 25, 27.5, 81, 78.5, dtfal, **{'color': cmap(2), 'linewidth': 0.6}),
        {'sf': 14, 'tlat': 27.5, 'tlon': 78.5, 'color': cmap(2)},

        {'sf': 2.7, 'tlat': 5, 'tlon': 111, 'color': cmap(3)},
        *fill_space(2.7, 4, 5, 0, 111, 108, dtfal, **{'color': cmap(3), 'linewidth': 0.6}),
        {'sf': 4, 'tlat': 0, 'tlon': 108, 'color': cmap(3)},
        *fill_space(4, 7, 0, -5, 108, 110, dtfal, **{'color': cmap(3), 'linewidth': 0.6}),
        {'sf': 7, 'tlat': -5, 'tlon': 110, 'color': cmap(3)},
        *fill_space(7, 15, -5, -7, 110, 108, dtfal, **{'color': cmap(3), 'linewidth': 0.6}),
        {'sf': 15, 'tlat': -7, 'tlon': 108, 'color': cmap(3)},
    ]

    # plt.figure()
    # ax = plt.axes(projection=ccrs.EqualEarth())
    # ax.set_global()
    #
    # ax.add_feature(cfeature.OCEAN)
    # ax.add_feature(cfeature.LAND, edgecolor='black', linewidth=0.2)
    # ax.add_feature(cfeature.LAKES, edgecolor='black', linewidth=0.2)
    # ax.add_feature(cfeature.STATES, edgecolor='black', linewidth=0.1)
    # ax.add_feature(cfeature.BORDERS, edgecolor='black', linewidth=0.2)
    #
    # for g in tqdm.tqdm(grids):
    #     draw_target_face_outline(ax, **g)
    #
    # plt.tight_layout()
    # plt.show()

    control_res = 720

    # Estimate get res

    # # 48 ~ 96 cores (3 nodes)
    sf = np.array([d['sf'] for d in grids])
    tlat = np.array([d['tlat'] for d in grids])
    tlon = np.array([d['tlon'] for d in grids])
    #
    # res = np.array([find_optimal_res(control_res, s, lat0, lon0) for s, lat0, lon0 in tqdm.tqdm(zip(sf, tlat, tlon))])
    #
    #
    # res_factor = res / 48
    # complexity = 2**(res_factor - 1)
    #
    # cores_per_node = 32
    # nnodes = (3*complexity + 0.5).astype(int)
    # ncores = cores_per_node*nnodes
    #
    # ncores = ncores.astype(int)
    # nnodes = nnodes.astype(int)
    # res = res.astype(int)
    #
    # grids2 = []
    #
    # print(f'{"Cores":>7},{"Nodes":>7},{"CPN":>7},{"SF":>7},{"RES":>7}{"TLAT":>7},{"TLON":>7}')
    # for nc, nn, r, s, lat0, lon0 in zip(ncores, nnodes, res, sf, tlat, tlon):
    #     gkw = {'sf': s, 'tlat': lat0, 'tlon': lon0}
    #     if r < 24 or r > 120:
    #         print(f'-- {nc:4d},{nn:7d},{32:7d},{s:7.2f},{r:7d}{lat0:7.2f},{lon0:7.2f}')
    #         gkw['color'] = 'red'
    #     else:
    #         print(f'{nc:7d},{nn:7d},{32:7d},{s:7.2f},{r:7d}{lat0:7.2f},{lon0:7.2f}')
    #         gkw['color'] = 'blue'
    #     grids2.append(gkw)

    for s, lat0, lon0 in zip(sf, tlat, tlon):
        print(f'{s:7.2f},{lat0:7.2f},{lon0:7.2f}')

    plt.figure()
    ax = plt.axes(projection=ccrs.EqualEarth())
    ax.set_global()

    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.LAND, edgecolor='black', linewidth=0.2)
    ax.add_feature(cfeature.LAKES, edgecolor='black', linewidth=0.2)
    ax.add_feature(cfeature.STATES, edgecolor='black', linewidth=0.1)
    ax.add_feature(cfeature.BORDERS, edgecolor='black', linewidth=0.2)

    for g in tqdm.tqdm(grids):
        draw_target_face_outline(ax, **g)

    plt.tight_layout()
    plt.show()




