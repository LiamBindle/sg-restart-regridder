from typing import Callable

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import pyproj
import numpy as np
import xarray as xr
import pandas as pd

import sg.pipe_operations
from sg.experiment import Experiment
from sg.grids import CSData, StretchedGrid
from sg.plot import *


def plot(experiment: Experiment, dates: pd.DatetimeIndex, operator: Callable):

    #vmax = np.asscalar(da.max().values)

    # Get troposphere average
    da = da.pipe(
        sg.pipe_operations.mask_stratosphere,
        supplemental=ds,
        key_lut=experiment.key_lut
    ).pipe(
        sg.pipe_operations.vertical_average,
        key_lut=experiment.key_lut
    )




    for face in range(6):
        xx, yy = figax.transform_xy(experiment.grid.xe(face), experiment.grid.ye(face))

        face_data = sg.pipe_operations.select_face(da, face=face, key_lut=experiment.key_lut)

        # Draw grid
        draw_minor_grid_boxes(figax, xx, yy)
        draw_major_grid_boxes(figax, xx, yy)

        # Plot data
        plot_pcolomesh(figax, xx, yy, face_data, vmin=0, vmax=vmax)

        # Draw face number
        text = draw_face_number2(figax, xx, yy, face)
        draw_text_stroke(text)
    plt.show()



if __name__ == '__main__':
    experiment = Experiment(
        directory='/home/liam/stetson/BS1/gchp_TransportTracers-C180e/',
        grid=StretchedGrid(
            cs=48,
            sf=3.75,
            target_lat=36,
            target_lon=360 - 78
        )
    )
    date_range_args = {'start': '2016-01-01 0:30:00', 'periods': 1, 'freq': '1H'}
    dates = pd.date_range(**date_range_args)

    ds = experiment.load(
        collections=['SpeciesConc', 'StateMet_avg'],
        dates=dates,
        variables=['SpeciesConc_Rn222', 'Met_TropLev']
    )
    vmax = 5e-20
    da = ds['SpeciesConc_Rn222']
    da = sg.pipe_operations.tropospheric_average(da, ds, experiment.key_lut)

    for date in dates:
        plt.figure()
        proj = ccrs.PlateCarree()
        ax = plt.subplot(1, 1, 1, projection=proj)
        ax.set_global()
        ax.coastlines(linewidth=0.8)
        figax = FigureAxes(ax, proj)

        for face in range(6):
            xx, yy = figax.transform_xy(experiment.grid.xe(face), experiment.grid.ye(face))

            face_data = sg.pipe_operations.select_face(da, face=face, key_lut=experiment.key_lut)

            # Draw grid
            draw_minor_grid_boxes(figax, xx, yy)
            draw_major_grid_boxes(figax, xx, yy)

            # Plot data
            plot_pcolomesh(figax, xx, yy, face_data, vmin=0, vmax=vmax)

            # Draw face number
            text = draw_face_number2(figax, xx, yy, face)
            draw_text_stroke(text)
        plt.show()

