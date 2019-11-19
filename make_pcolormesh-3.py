import sys
import yaml

from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib import rc
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cartopy.crs as ccrs
import cartopy.feature as cfeature


from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature

import pyproj
import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm

from sg.pipe_operations import *
from sg.experiment import Experiment
from sg.grids import CubeSphere, StretchedGrid
from sg.plot import *
from sg.framer import *

fname = r'/home/liam/sg-restart-regridder/atlanta/City_of_Atlanta_Neighborhood_Statistical_Areas.shp'
shape_feature = ShapelyFeature(Reader(fname).geometries(),
                               ccrs.PlateCarree(), edgecolor='black', linewidth=0.3, facecolor='black')


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
    subax.add_feature(states_provinces, edgecolor='k', linewidth=0.5, alpha=0.5)

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
            horizontalalignment='right', verticalalignment='center', weight='bold')
    text = ax.text(sbx - length*200, sbyF, '600 km', transform=tmc,
            horizontalalignment='right', verticalalignment='center', weight='bold')


if __name__ == '__main__':
    rc('text', usetex=True)
    rc('font', **{'family': 'sans-serif', 'size': 16, 'weight': 'bold'})

    plt.rcParams['text.latex.preamble'] = [
        r'\usepackage{siunitx}',  # i need upright \micro symbols, but you need...
        r'\sisetup{detect-all}',  # ...this to force siunitx to actually use your fonts
        r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
        r'\sansmath'  # <- tricky! -- gotta actually tell tex to use!
    ]

    working_dir = os.path.dirname(sys.argv[1])

    with open(sys.argv[1], 'r') as f:
        yaml_input = yaml.safe_load(f)

    # Make output directory
    output_dir = os.path.join(working_dir, os.path.dirname(yaml_input['pcolormesh']['output']))
    output_fname = os.path.basename(yaml_input['pcolormesh']['output'])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create Experiment object
    experiment = Experiment(
        directory=yaml_input['experiment']['directory'],
        grid=eval(yaml_input['experiment']['grid'])
    )

    # Get collections to read and variables to keep
    target = yaml_input['pcolormesh']['target']
    supplemental= yaml_input['pcolormesh']['supplemental']
    if isinstance(supplemental, str):
        supplemental = [supplemental]
    collections = list(set([variable.split(':', 1)[0] for variable in [target, *supplemental]]))
    variables = list(set([variable.split(':', 1)[1] for variable in [target, *supplemental]]))
    target = target.split(':', 1)[1]

    # General settings
    vmin = yaml_input['pcolormesh'].get('vmin', 0)
    vmax = yaml_input['pcolormesh'].get('vmax', None)

    # Loop over times
    dates = pd.date_range(**yaml_input['pcolormesh']['date_range'])
    for timestamp in tqdm(dates):
        # Load data
        ds = experiment.load(
            collections=collections,
            date=timestamp.to_pydatetime(),
            variables=variables
        )
        da = ds[target]

        # Perform operation
        operations = yaml_input['pcolormesh'].get('operation', 'noop')
        if isinstance(operations, str):
            operations = [eval(operations)]
        else:
            operations = [eval(operation) for operation in operations]
        for callable_operation in operations:
            da = callable_operation(da, supplemental=ds, key_lut=experiment.key_lut)

        # General figure settings
        if vmax is None:
            vmax = np.asscalar(da.max().values)

        # Create figure
        plt.figure(figsize=(16, 8), dpi=100)
        figax = plate_carree(experiment)

        # Loop over faces
        for face in range(6):
            xx, yy = figax.transform_xy(experiment.grid.xe(face), experiment.grid.ye(face))

            face_data = select_face(da, face=face, key_lut=experiment.key_lut)

            # Draw grid
            if face in [0, 1, 3, 4]:
                draw_minor_grid_boxes(figax, xx, yy, linewidth=0.08)
            if face in [5]:
                draw_minor_grid_boxes(figax, xx, yy, linewidth=0.03)
            if face in [2]:
                draw_minor_grid_boxes(figax, xx, yy, linewidth=0.15)
            draw_major_grid_boxes(figax, xx, yy, linewidth=1.5)

            # Plot data
            pcolormesh = plot_pcolomesh(figax, xx, yy, face_data, vmin=vmin, vmax=vmax)
            if face == 0:
                _, _, w, h= figax.ax.get_position().bounds
                cb = plt.colorbar(pcolormesh, fraction=0.02)
                cb.set_label(yaml_input['pcolormesh'].get('units', ''))
                ticks = yaml_input['pcolormesh'].get('colorbar_ticks', [vmin, (vmin + vmax)/2, vmax])
                cb.set_ticks(ticks)
                if log10 in operations:
                    tick_labels = yaml_input['pcolormesh'].get('colorbar_tick_labels', [f'$10^{{ {tick} }}$' for tick in ticks])
                    cb.set_ticklabels(tick_labels)

            # Draw face number
            #text = draw_face_number(figax, xx, yy, face)
            #draw_text_stroke(text)
        # plt.plot([-95.88, 21.44], [42.86, 37.39], color='black') # top-left
        # plt.plot([-72.07, 158], [22.06, -81.87], color='black') # bottom-right
        # plt.plot([-72.07, 158], [42.86, 37.39], color='black') # top-right
        # plt.plot([-95.88, 21.44], [22.06, -81.87], color='black')  # bottom-left
        x1 = -84.4 - 6.25
        x2 = -84.4 + 6.25
        y1 = 33.7 - 4.5
        y2 = 33.7 + 4
        plt.plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1], color='white', linewidth=1)
        plt.title(yaml_input['pcolormesh'].get('title', ''))

        figax.ax.text(75, -80, f'Time: {timestamp.strftime("%Y-%m-%d %H:%M")}',
                       horizontalalignment='center', verticalalignment='top', weight='bold')

        subax = add_subplot_axes(figax.ax, [0.4, 0.05 ,0.6,0.9]) # from left, from bot, width, height
        figax = FigureAxes(subax, ccrs.Mercator())
        for face in range(6):
            if face != 4: continue
            xx, yy = figax.transform_xy(experiment.grid.xe(face), experiment.grid.ye(face))

            face_data = select_face(da, face=face, key_lut=experiment.key_lut)

            # Draw grid
            if face in [0, 1, 3, 4]:
                draw_minor_grid_boxes(figax, xx, yy, linewidth=0.08)
            if face in [5]:
                draw_minor_grid_boxes(figax, xx, yy, linewidth=0.06)
            if face in [2]:
                draw_minor_grid_boxes(figax, xx, yy, linewidth=0.15)
            draw_major_grid_boxes(figax, xx, yy, linewidth=1)

            # Plot data
            pcolormesh = plot_pcolomesh(figax, xx, yy, face_data, vmin=vmin, vmax=vmax)

        #scale_bar(figax.ax, 100, location=(-0.8, 0.2))
        scale_bar(figax.ax, 100, location=(0.95, 0.6))

        atl_x, atl_y = figax.transform_xy(-84.4, 33.7)
        #figax.ax.scatter([atl_x], [atl_y], color='k', s=0.5)
        atl_x, atl_y = figax.transform_xy(-84.4, 34)
        figax.ax.text(atl_x, atl_y, f'Atlanta',
                       horizontalalignment='center', verticalalignment='bottom', weight='bold')


        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, output_fname.format(timestamp=timestamp)))
        #plt.show()
        plt.close()

