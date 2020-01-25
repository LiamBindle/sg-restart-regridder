import sys
import datetime

import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.patheffects as path_effects
from matplotlib.colors import LightSource

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import pyproj
import shapely.geometry
import rasterio

import sg.grids
import sg.plot



time = datetime.datetime(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])

OutputDir = ''
GISDataDir=''
field_name = 'SpeciesConc_O3'
cmap_name = 'cividis'
vmin = 20
vmax = 60

overlay_x = [-124.9, -113.7]
overlay_y = [32, 40.1]




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

cmap = plt.cm.get_cmap(cmap_name)
norm = plt.Normalize(vmin, vmax)

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
        far_from_the_prime_meridian = shapely.geometry.LineString([(80, -90), (80, 90)])
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
        linewidth=min([0.5, linewidth])
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
        linewidth=min([0.5, linewidth])
        ax.add_geometries([poly], ccrs.PlateCarree(), edgecolor='#15151550', facecolor='None', linewidth=linewidth,
                          zorder=-0.5)
        poly = poly.buffer(eps)
        c = cmap(norm(data[tuple(idx)]))
        ax.add_geometries([poly], ccrs.PlateCarree(), edgecolor='None', facecolor=c, linewidth=0, zorder=-1)


    # Plot others
    neither_idx = np.argwhere(~neither)
    for idx in neither_idx:
        enhanced_xy = enhance_gridbox_edges(boxes[tuple(idx)], 90)
        poly = shapely.geometry.LinearRing(enhanced_xy)
        linewidth = max([0.06, np.log(poly.length / 4) * 0.2])
        linewidth=min([0.5, linewidth])
        ax.add_geometries([poly], ccrs.PlateCarree(), edgecolor='#15151550', facecolor='None', linewidth=linewidth,
                          zorder=-0.5)
        poly_buff = poly.buffer(eps)
        c = cmap(norm(data[tuple(idx)]))
        ax.add_geometries([poly, poly_buff.exterior], ccrs.PlateCarree(), edgecolor='None', facecolor=c, linewidth=0, zorder=-1)


#grid = sg.grids.StretchedGrid(48, 15, 33.7, 275.6)
grid = sg.grids.StretchedGrid(48, 15, 36, -120)
#grid = sg.grids.StretchedGrid(48, 3, 26, 115)
#grid = sg.grids.CubeSphere(48)

ds = xr.open_dataset(f'{OutputDir}/GCHP.SpeciesConc.{time.year:04d}{time.month:02d}{time.day:02d}_{time.hour:02d}{time.minute:02d}z.nc4')

plt.figure(figsize=(12,6))
ax = plt.axes(projection=ccrs.Robinson(), )
ax.add_feature(cartopy.feature.BORDERS, linewidth=0.5, edgecolor='#050505')
ax.add_feature(cartopy.feature.COASTLINE, linewidth=0.5, edgecolor='#050505')
ax.set_global()
ax.outline_patch.set_edgecolor('#151515')
ax.background_patch.set_facecolor('#15151550')
#ax.background_patch.set_fill(False)

for i in range(6):
    level = 0
    da = ds[field_name].isel(time=0, nf=i, lev=level).squeeze()#.transpose('Xdim', 'Ydim')
    xx = grid.xe(i)
    yy = grid.ye(i)
    xx[xx > 180] -= 360
    if i == 2:
        draw_polygons(ax, xx, yy, da.values * 1e9, vmin=vmin, vmax=vmax, cmap=cmap_name)
    else:
        plot_pcolomesh(ax, xx, yy, da.values * 1e9, vmin=vmin, vmax=vmax, cmap=cmap_name)
    xx_majors = [xx[0, :], xx[-1, :], xx[:, 0], xx[:, -1]]
    yy_majors = [yy[0, :], yy[-1, :], yy[:, 0], yy[:, -1]]
    for xm, ym in zip(xx_majors, yy_majors):
        ax.plot(xm, ym, transform=ccrs.PlateCarree(), color='#151515', linewidth=1.2, alpha=0.8)


def add_features(ax, fname, attr_filter=None, **kwargs):
    reader = Reader(fname)
    if attr_filter:
        geometries = []
        for r in reader.records():
            if r.attributes[attr_filter[0]] in attr_filter[1]:
                geometries.append(r.geometry)
        features = ShapelyFeature(geometries, crs=ccrs.PlateCarree())
    else:
        features = ShapelyFeature(Reader(fname).geometries(), ccrs.PlateCarree())
    kwargs.setdefault('edgecolor', '#151515')
    kwargs.setdefault('linewidth', 0.5)
    ax.add_feature(features, **kwargs)


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

    x1 = overlay_x[0]
    x2 = overlay_x[1]
    y1 = overlay_y[0]
    y2 = overlay_y[1]
    subax.set_extent([x1, x2, y1, y2], ccrs.Geodetic())
    #subax.coastlines()

    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none')
    subax.add_feature(cfeature.COASTLINE, edgecolor='#050505', linewidth=1.2)
    subax.add_feature(states_provinces, edgecolor='#050505', linewidth=1.2)

    # subax.add_feature(shape_feature)


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
            horizontalalignment='right', verticalalignment='center', color='k', weight='bold', fontsize=10)
    text.set_path_effects([path_effects.Stroke(linewidth=1.2, foreground='whitesmoke'),
                           path_effects.Normal()])
    text = ax.text(sbx - length*200, sbyF, '600 km', transform=tmc,
            horizontalalignment='right', verticalalignment='center', color='k', weight='bold', fontsize=10)
    text.set_path_effects([path_effects.Stroke(linewidth=1.2, foreground='whitesmoke'),
                           path_effects.Normal()])

subax = add_subplot_axes(ax, [0.25, 0.1, 0.8, 0.9])  # from left, from bot, width, height

subax.outline_patch.set_edgecolor('white')
subax.outline_patch.set_linewidth(4)
subax.add_feature(cartopy.feature.BORDERS, linewidth=1.2, edgecolor='#050505')

for face in [0, 1, 3, 4, 5]:
    da = ds[field_name].isel(time=0, nf=face, lev=level).squeeze()  # .transpose('Xdim', 'Ydim')
    xx = grid.xe(face)
    yy = grid.ye(face)
    xx[xx > 180] -= 360
    plot_pcolomesh(subax, xx, yy, da.values * 1e9, vmin=vmin, vmax=vmax, cmap=cmap_name)
    draw_minor_grid_boxes(subax, xx, yy, linewidth=0.3, alpha=0.4, color='#151515')


    xx_majors = [xx[0, :], xx[-1, :], xx[:, 0], xx[:, -1]]
    yy_majors = [yy[0, :], yy[-1, :], yy[:, 0], yy[:, -1]]
    for xm, ym in zip(xx_majors, yy_majors):
        subax.plot(xm, ym, transform=ccrs.PlateCarree(), color='#151515', linewidth=0.6, alpha=0.4)


add_features(subax, f'{GISDataDir}/tl_2015_06_prisecroads.shp', attr_filter=('RTTYP', ['U', 'I']), linewidth=0.4, edgecolor='#050505', facecolor='none')

x1 = overlay_x[0]
x2 = overlay_x[1]
y1 = overlay_y[0]
y2 = overlay_y[1]

img = rasterio.open(f'{GISDataDir}/SR_LR.tif')
img_top_left = img.index(x1, y2)
img_bot_right = img.index(x2, y1)
img = img.read(1)[img_top_left[0]:img_bot_right[0], img_top_left[1]:img_bot_right[1]]

ls = LightSource(azdeg=315, altdeg=45)
shade = ls.hillshade(img, vert_exag=10000, fraction=1.0)

cm = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap', [(0, 0, 0, 0.4), (0, 0, 0, 0)])
subax.imshow(img, origin='upper', extent=[x1, x2, y1, y2], transform=ccrs.PlateCarree(), cmap=cm, zorder=10, vmax=206)

ax.plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1], color='white', linewidth=2, transform=ccrs.PlateCarree())


cities = {
    'San Francisco': (-122.41, 37.77),
    'Sacramento': (-121.49, 38.58),
    'Los Angeles': (-118.24, 34.05),
    'San Jose': (-121.89, 37.33),
    'San Diego': (-117.16, 32.72),
}

for name, coord in cities.items():
    subax.scatter(*coord, color='whitesmoke', edgecolor='#151515', linewidth=1, s=15, transform=ccrs.PlateCarree(), zorder=100)
    v_offset = 0.1
    h_offset = 0
    if name == 'San Jose':
        h_offset = 0.4
    text = subax.text(
        coord[0] + h_offset, coord[1] + v_offset,
        name,
        transform=ccrs.PlateCarree(),
        horizontalalignment='center', verticalalignment='bottom',
        color='k', weight='bold', fontsize=10,
    )
    text.set_path_effects([path_effects.Stroke(linewidth=1.2, foreground='whitesmoke'),
                           path_effects.Normal()])

scale_bar(subax, 100, location=(0.95, 0.70))
cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap_name), orientation='horizontal', pad=0.04,  ax=ax,aspect=60, fraction=0.02, ticks=[vmin, (vmin+vmax)//2, vmax], drawedges=False)
cbxtick_obj = plt.getp(cb.ax.axes, 'xticklabels')
plt.setp(cbxtick_obj, color='white')
plt.tight_layout()
cb.ax.set_xticklabels([f'{vmin} ppb', f'{(vmin+vmax)//2} ppb', f'{vmax} ppb'])

plt.text(-0.07, 0.025, f'{time.year:4d}-{time.month:02d}-{time.day:02d} {time.hour:02d}:{time.minute:02d}Z', color='white', fontsize=14, transform=ax.transAxes, horizontalalignment='left', verticalalignment='bottom')
plt.text(-0.07, -0.08, f'Surface-level Ozone\nGCHPctm 13.0.0-alpha.0\nC48 (stretch=15x)', color='white', fontsize=10, transform=ax.transAxes, horizontalalignment='left', verticalalignment='bottom', linespacing=1.4)

plt.savefig(f'{field_name}-{time.year:04d}{time.month:02d}{time.day:02d}_{time.hour:02d}{time.minute:02d}z.jpeg', dpi=300, facecolor='#151515', edgecolor='#151515')
# plt.show()