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

import cartopy.feature as cfeature


from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature

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

from datetime import datetime
import sys
time = datetime(2016, 1, sys.argv[1], sys.argv[2], 30)

ds = xr.open_dataset(f'GCHP.SpeciesConc.{time.year:4d}{time.month:02d}{time.day:02d}_{time.hour:02d}{time.minute:02d}z.nc4')

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

plt.text(-0.14, 0.025, f'{time.year:4d}-{time.month:02d}-{time.day:02d} {time.hour:02d}:{time.minute:02d}Z', color='white', fontsize=16, transform=ax.transAxes, horizontalalignment='left', verticalalignment='bottom')
plt.text(-0.14, -0.1, f'Radon-222 concentration (model level 17)\nGCHPctm 13.0.0-alpha.0\nC48 (stretch=15x) transport tracer simulation', color='white', fontsize=12, transform=ax.transAxes, horizontalalignment='left', verticalalignment='bottom', linespacing=1.4)


fname = r'/home/liam/sg-restart-regridder/atlanta/City_of_Atlanta_Neighborhood_Statistical_Areas.shp'
shape_feature = ShapelyFeature(Reader(fname).geometries(),
                               ccrs.PlateCarree(), edgecolor='white', linewidth=0.5, facecolor='white')


def add_subplot_axes(ax,rect,axisbg='w'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x,y,width,height], projection=ccrs.Mercator())
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)

    x1 = -84.4 - 6.25
    x2 = -84.4 + 6.25
    y1 = 33.7 - 4.5
    y2 = 33.7 + 4
    subax.set_extent([x1, x2, y1, y2], ccrs.Geodetic())
    #subax.coastlines()

    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none')
    subax.add_feature(cfeature.COASTLINE, edgecolor='k', linewidth=0.8)
    subax.add_feature(states_provinces, edgecolor='k', linewidth=0.5)

    subax.add_feature(shape_feature)


    return subax

def scale_bar(ax, length=None, location=(0.5, 0.05), linewidth=3):
    """
    ax is the axes to draw the scalebar on.
    length is the length of the scalebar in km.
    location is center of the scalebar in axis coordinates.
    (ie. 0.5 is the middle of the plot)
    linewidth is the thickness of the scalebar.
    """
    #Get the limits of the axis in lat long
    llx0, llx1, lly0, lly1 = ax.get_extent(ccrs.PlateCarree())
    #Make tmc horizontally centred on the middle of the map,
    #vertically at scale bar location
    # sbllx = (llx1 + llx0) / 2
    # sblly = lly0 + (lly1 - lly0) * location[1]
    sbllx = llx0 + (llx1 - llx0) * location[0]
    sblly = (lly1 + lly0) / 2
    tmc = ccrs.TransverseMercator(sbllx, sblly)
    #Get the extent of the plotted area in coordinates in metres
    x0, x1, y0, y1 = ax.get_extent(tmc)
    #Turn the specified scalebar location into coordinates in metres
    # sbx = x0 * location[0]
    # sby = y0 + (y1 - y0) * location[1]
    sbx = x0 + (x1 - x0) * location[0]
    sby = y0 * location[1]

    #Calculate a scale bar length if none has been given
    #(Theres probably a more pythonic way of rounding the number but this works)
    if not length:
        length = (y1 - y0) / 5000 #in km
        ndim = int(np.floor(np.log10(length))) #number of digits in number
        length = round(length, -ndim) #round to 1sf
        #Returns numbers starting with the list
        def scale_number(x):
            if str(x)[0] in ['1', '2', '5']: return int(x)
            else: return scale_number(x - 10 ** ndim)
        length = scale_number(length)

    #Generate the x coordinate for the ends of the scalebar
    bar_xs = [sbx, sbx]
    bar_ys = [sby, sby + length * 1000]
    sby0 = sby
    for i in range(3):
        # Plot the scalebar
        ax.plot(bar_xs, bar_ys, transform=tmc, color='k', linewidth=linewidth)
        bar_ys = np.array(bar_ys) + length * 1000
        ax.plot(bar_xs, bar_ys, transform=tmc, color='white', linewidth=linewidth)
        bar_ys = np.array(bar_ys) + length * 1000
    sbyF = bar_ys[1] - length*1000
    #Plot the scalebar label
    text = ax.text(sbx - length*200, sby0, '  0 km', transform=tmc,
            horizontalalignment='right', verticalalignment='center', color='white', weight='normal')
    text = ax.text(sbx - length*200, sbyF, '600 km', transform=tmc,
            horizontalalignment='right', verticalalignment='center', color='white', weight='normal')

subax = add_subplot_axes(ax, [0.4, 0.05, 0.6, 0.9])  # from left, from bot, width, height

subax.outline_patch.set_edgecolor('white')
subax.outline_patch.set_linewidth(1.4)

for face in [0, 1, 3, 4, 5]:
    da = ds['SpeciesConc_Rn222'].isel(time=0, nf=face, lev=level).squeeze()
    draw_minor_grid_boxes(subax, grid.xe(face), grid.ye(face), alpha=0.5)
    subax.pcolormesh(grid.xe(face), grid.ye(face), da.values, vmin=0, vmax=3e-20, cmap='cividis', transform=ccrs.PlateCarree())

x1 = -84.4 - 6.25
x2 = -84.4 + 6.25
y1 = 33.7 - 4.5
y2 = 33.7 + 4
ax.plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1], color='white', linewidth=2, transform=ccrs.PlateCarree())

# for face in range(6):
#     if face == 3: continue
#
#     xx = grid.xe(i)
#     yy = grid.ye(i)
#
#     draw_minor_grid_boxes(subax, xx, yy)
    #
    # xx, yy = figax.transform_xy(experiment.grid.xe(face), experiment.grid.ye(face))
    #
    # face_data = select_face(da, face=face, key_lut=experiment.key_lut)
    #
    # # Draw grid
    # if face in [0, 1, 3, 4]:
    #     draw_minor_grid_boxes(figax, xx, yy, linewidth=0.08)
    # if face in [5]:
    #     draw_minor_grid_boxes(figax, xx, yy, linewidth=0.06)
    # if face in [2]:
    #     draw_minor_grid_boxes(figax, xx, yy, linewidth=0.15)
    # draw_major_grid_boxes(figax, xx, yy, linewidth=1)
    #
    # # Plot data
    # pcolormesh = plot_pcolomesh(figax, xx, yy, face_data, vmin=vmin, vmax=vmax)

# scale_bar(figax.ax, 100, location=(-0.8, 0.2))
scale_bar(subax, 100, location=(0.95, 0.6))
#
atl_x = -84.4
atl_y = 34
subax.text(atl_x, atl_y, f'Atlanta',
              horizontalalignment='center', verticalalignment='bottom', weight='normal', color='white', transform=ccrs.PlateCarree())

plt.savefig(f'frames-overlaid/{time.year:4d}{time.month:02d}{time.day:02d}_{time.hour:02d}{time.minute:02d}z.jpeg', dpi=100, facecolor='#151515', edgecolor='#151515')
# plt.show()