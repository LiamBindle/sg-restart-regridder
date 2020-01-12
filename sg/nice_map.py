import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors
import cartopy
import shapely.geometry
from tqdm import tqdm
import pyproj

import xarray as xr

import sg.grids
import sg.plot

def plot_pcolomesh(ax, xx, yy, data, **kwargs):
    # xx must be [-180 to 180]
    p0 = slice(0, -1)
    p1 = slice(1, None)

    # Mask where bounding box crosses the prime meridian or antimeridian
    cross_pm_or_am_line1 = np.not_equal(np.sign(xx[p0, p0]), np.sign(xx[p1, p0]))
    cross_pm_or_am_line2 = np.not_equal(np.sign(xx[p1, p0]), np.sign(xx[p1, p1]))
    cross_pm_or_am_line3 = np.not_equal(np.sign(xx[p1, p1]), np.sign(xx[p0, p1]))
    cross_pm_or_am_line4 = np.not_equal(np.sign(xx[p0, p1]), np.sign(xx[p0, p0]))
    cross_pm_or_am = cross_pm_or_am_line1 | cross_pm_or_am_line2 | cross_pm_or_am_line3 | cross_pm_or_am_line4

    boxes_x = np.moveaxis(np.array([xx[p0, p0], xx[p1, p0], xx[p1, p1], xx[p0, p1]]), 0, -1)
    boxes_y = np.moveaxis(np.array([yy[p0, p0], yy[p1, p0], yy[p1, p1], yy[p0, p1]]), 0, -1)
    boxes = np.moveaxis(np.array([boxes_x, boxes_y]), 0, -1)

    pm = np.ones_like(data, dtype=bool)
    am = np.ones_like(data, dtype=bool)
    neither = np.copy(cross_pm_or_am)

    crossing_indexes = np.argwhere(cross_pm_or_am)
    for idx in crossing_indexes:
        box = shapely.geometry.LinearRing(boxes[tuple(idx)])
        far_from_the_prime_meridian = shapely.geometry.LineString([(-160, -90), (-160, 90)])
        if box.crosses(far_from_the_prime_meridian):
            am[tuple(idx)] = False
        else:
            pm[tuple(idx)] = False

    data0 = np.ma.masked_where(neither & am, data)
    data1 = np.ma.masked_where(neither & pm, data)
    xx2 = np.copy(xx)
    xx2[xx2 < 0] += 360
    ax.pcolormesh(xx2, yy, data0, transform=ccrs.PlateCarree(), **kwargs)
    ax.pcolormesh(xx, yy, data1, transform=ccrs.PlateCarree(), **kwargs)

def draw_minor_grid_boxes(ax, xx, yy, **kwargs):
    kwargs.setdefault('linewidth', 0.3)
    kwargs.setdefault('color', 'black')
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

cmap = plt.cm.get_cmap('cividis')
norm = plt.Normalize(0, 3e-20)

def enhance_gridbox_edges(box_xy: np.array, res):
    temp_proj = pyproj.Proj(ccrs.Gnomonic(
        central_longitude=np.mean(box_xy[:, 0]),
        central_latitude=np.mean(box_xy[:, 1])
    ).proj4_init)
    new_x = []
    new_y = []
    for (seg_x0, seg_y0), (seg_x1, seg_y1) in zip(box_xy[:-1, :], box_xy[1:, :]):
        seg_length = np.sqrt((seg_x1-seg_x0)**2 + (seg_y1-seg_y0)**2)
        gx, gy = temp_proj([seg_x0, seg_x1], [seg_y0, seg_y1])
        m = (gy[1] - gy[0]) / (gx[1] - gx[0])
        b = gy[0] - m*gx[0]
        interped_gx = np.linspace(gx[0], gx[1], res)
        interped_gy = m*interped_gx + b
        x, y = temp_proj(interped_gx, interped_gy, inverse=True)
        new_x.extend(x)
        new_y.extend(y)
    new_x.append(new_x[0])
    new_y.append(new_y[0])
    new_xy = np.moveaxis(np.array([new_x, new_y]), 0, -1)
    return new_xy



def draw_polygons(ax, xx, yy, data, **kwargs):
    if np.any(xx > 180):
        raise ValueError('xx must be in [-180, 180]')
    # xx must be [-180 to 180]
    p0 = slice(0, -1)
    p1 = slice(1, None)

    # Mask where bounding box crosses the prime meridian or antimeridian
    cross_pm_or_am_line1 = np.not_equal(np.sign(xx[p0, p0]), np.sign(xx[p1, p0]))
    cross_pm_or_am_line2 = np.not_equal(np.sign(xx[p1, p0]), np.sign(xx[p1, p1]))
    cross_pm_or_am_line3 = np.not_equal(np.sign(xx[p1, p1]), np.sign(xx[p0, p1]))
    cross_pm_or_am_line4 = np.not_equal(np.sign(xx[p0, p1]), np.sign(xx[p0, p0]))
    cross_pm_or_am = cross_pm_or_am_line1 | cross_pm_or_am_line2 | cross_pm_or_am_line3 | cross_pm_or_am_line4

    # Make xy polygons for each gridbox
    boxes_x = np.moveaxis(np.array([xx[p0, p0], xx[p1, p0], xx[p1, p1], xx[p0, p1], xx[p0, p0]]), 0, -1)
    boxes_y = np.moveaxis(np.array([yy[p0, p0], yy[p1, p0], yy[p1, p1], yy[p0, p1], yy[p0, p0]]), 0, -1)
    boxes = np.moveaxis(np.array([boxes_x, boxes_y]), 0, -1)

    pm = np.ones_like(data, dtype=bool)
    am = np.ones_like(data, dtype=bool)
    neither = np.copy(cross_pm_or_am)

    # Figure out which boxes cross the prime meridian and antimeridian
    crossing_indexes = np.argwhere(cross_pm_or_am)
    for idx in crossing_indexes:
        box = shapely.geometry.LinearRing(boxes[tuple(idx)])
        far_from_the_prime_meridian = shapely.geometry.LineString([(-160, -90), (-160, 90)])
        if box.crosses(far_from_the_prime_meridian):
            am[tuple(idx)] = False
        else:
            pm[tuple(idx)] = False

    eps = 0.2

    # Plot pm
    pm_idx = np.argwhere(~pm)
    for idx in pm_idx:
        enhanced_xy = enhance_gridbox_edges(boxes[tuple(idx)], 60)
        poly = shapely.geometry.Polygon(enhanced_xy)
        linewidth = max([0.06, np.log(poly.length / 4) * 0.2])
        linewidth=min([0.3, linewidth])
        ax.add_geometries([poly], ccrs.PlateCarree(), edgecolor='#15151550', facecolor='None', linewidth=linewidth, zorder=-0.5)
        poly = poly.buffer(eps)
        c = cmap(norm(data[tuple(idx)]))
        ax.add_geometries([poly], ccrs.PlateCarree(), edgecolor='None', facecolor=c, linewidth=0, zorder=-1)


    # Plot am
    am_idx = np.argwhere(~am)
    for idx in am_idx:
        b = boxes[tuple(idx)]
        b[:, 0] = b[:, 0] % 360
        enhanced_xy = enhance_gridbox_edges(b, 60)
        enhanced_xy[:,0] = enhanced_xy[:,0] % 360
        poly = shapely.geometry.Polygon(enhanced_xy)
        linewidth = max([0.06, np.log(poly.length / 4) * 0.2])
        linewidth=min([0.3, linewidth])
        ax.add_geometries([poly], ccrs.PlateCarree(), edgecolor='#15151550', facecolor='None', linewidth=linewidth,
                          zorder=-0.5)
        poly = poly.buffer(eps)
        c = cmap(norm(data[tuple(idx)]))
        ax.add_geometries([poly], ccrs.PlateCarree(), edgecolor='None', facecolor=c, linewidth=0.3, zorder=-1)


    # Plot others
    neither_idx = np.argwhere(~neither)
    for idx in tqdm(neither_idx):
        enhanced_xy = enhance_gridbox_edges(boxes[tuple(idx)], 90)
        poly = shapely.geometry.LinearRing(enhanced_xy)
        linewidth = max([0.06, np.log(poly.length / 4) * 0.2])
        linewidth=min([0.3, linewidth])
        ax.add_geometries([poly], ccrs.PlateCarree(), edgecolor='#15151550', facecolor='None', linewidth=linewidth,
                          zorder=-0.5)
        poly_buff = poly.buffer(eps)
        c = cmap(norm(data[tuple(idx)]))
        ax.add_geometries([poly, poly_buff.exterior], ccrs.PlateCarree(), edgecolor='None', facecolor=c, linewidth=0, zorder=-1)

grid = sg.grids.StretchedGrid(48, 15, 33.7, 275.6)
# grid = sg.grids.CubeSphere(48)

ds = xr.open_dataset('/extra-space/GCHP.SpeciesConc.20160113_1230z.nc4')

plt.figure(figsize=(12,6))
ax = plt.axes(projection=ccrs.Mollweide(), )
# ax.coastlines(linewidth=0.3, color='#656565')
ax.add_feature(cartopy.feature.BORDERS, linewidth=0.3, edgecolor='k')
ax.add_feature(cartopy.feature.COASTLINE, linewidth=0.3, edgecolor='k')
ax.set_global()
ax.outline_patch.set_edgecolor('#151515')
ax.background_patch.set_facecolor('#15151550')
#ax.background_patch.set_fill(False)

for i in [2]:#[5, 4, 3, 1, 0, 2][::-1]:
    level = 17
    da = ds['SpeciesConc_Rn222'].isel(time=0, nf=i, lev=level).squeeze()#.transpose('Xdim', 'Ydim')
    xx = grid.xe(i)
    yy = grid.ye(i)
    # plot_pcolomesh(ax, xx, yy, da.values, vmin=0, vmax=3e-20, cmap='cividis')
    xx[xx > 180] -= 360
    draw_polygons(ax, xx, yy, da.values)
for i in [5, 4, 3, 1, 0][::-1]:
    level = 17
    da = ds['SpeciesConc_Rn222'].isel(time=0, nf=i, lev=level).squeeze()#.transpose('Xdim', 'Ydim')
    xx = grid.xe(i)
    yy = grid.ye(i)
    xx[xx > 180] -= 360
    plot_pcolomesh(ax, xx, yy, da.values, vmin=0, vmax=3e-20, cmap='cividis')
    # draw_polygons(ax, xx, yy, da.values)
plt.savefig('temp-mo.png', dpi=100, facecolor='#151515', edgecolor='#151515')
# plt.show()