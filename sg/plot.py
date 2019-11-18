from typing import Tuple, List
import os.path

import pyproj
import numpy as np
import xarray as xr
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

from sg.figure_axes import FigureAxes
from sg.experiment import Experiment
from sg.grids import *


def draw_face_number(figax: FigureAxes, grid: CSDataBase, face, **kwargs):
    xx, yy = figax.transform_xy(grid.xe(face), grid.ye(face))
    middle = np.array(xx.shape, dtype=np.int) // 2
    figax.ax.text(xx[middle[0], middle[1]], yy[middle[0], middle[1]], f'{face}', **kwargs)


def draw_minor_grid_boxes(figax: FigureAxes, xx, yy, **kwargs):
    kwargs.setdefault('linewidth', 0.3)
    kwargs.setdefault('color', 'black')

    for x, y in zip(xx, yy):
        if figax.x_split:
            idx = np.argwhere(np.diff(np.sign(x % 360 - figax.x_split))).flatten()
            x360 = x % 360
            idx = idx[(x360[idx] > 10) & (x360[idx] < 350)]
            start = [0, *(idx + 1)]
            end = [*(idx + 1), len(x)]
        else:
            start = [0]
            end = [len(x)]
        for s, e in zip(start, end):
            figax.ax.plot(x[s:e], y[s:e], **kwargs)
    for x, y in zip(xx.transpose(), yy.transpose()):
        if figax.x_split:
            idx = np.argwhere(np.diff(np.sign(x % 360 - figax.x_split))).flatten()
            x360 = x % 360
            idx = idx[(x360[idx] > 10) & (x360[idx] < 350)]
            start = [0, *(idx + 1)]
            end = [*(idx + 1), len(x)]
        else:
            start = [0]
            end = [len(x)]
        for s, e in zip(start, end):
            figax.ax.plot(x[s:e], y[s:e], **kwargs)


def draw_major_grid_boxes(figax: FigureAxes, xx, yy, **kwargs):
    kwargs.setdefault('linewidth', 1.5)
    kwargs.setdefault('color', 'black')

    xx_majors = [xx[0, :], xx[-1, :], xx[:, 0], xx[:, -1]]
    yy_majors = [yy[0, :], yy[-1, :], yy[:, 0], yy[:, -1]]
    for x, y in zip(xx_majors, yy_majors):
        if figax.x_split:
            idx = np.argwhere(np.diff(np.sign(x % 360 - figax.x_split))).flatten()
            x360 = x % 360
            idx = idx[(x360[idx] > 10) & (x360[idx] < 350)]
            start = [0, *(idx + 1)]
            end = [*(idx + 1), len(x)]
        else:
            start = [0]
            end = [len(x)]
        for s, e in zip(start, end):
            figax.ax.plot(x[s:e], y[s:e], **kwargs)

def draw_face_number2(figax: FigureAxes, xx, yy, face, stroke_width=1, **kwargs):
    middle = np.array(xx.shape, dtype=np.int) // 2
    text = figax.ax.text(xx[middle[0], middle[1]], yy[middle[0], middle[1]], f'{face}', **kwargs)
    return text

def draw_text_stroke(text, **kwargs):
    kwargs.setdefault('linewidth', 1)
    kwargs.setdefault('foreground', 'white')
    text.set_path_effects([path_effects.Stroke(**kwargs),
                           path_effects.Normal()])


def draw_grid_boxes(figax: FigureAxes, grid: CSDataBase, face, major={}, minor={}):
    major.setdefault('linewidth', 1.5)
    major.setdefault('color', 'black')
    minor.setdefault('linewidth', 0.3)
    minor.setdefault('color', 'black')
    ax = figax.ax

    xe, ye = grid.xe(face), grid.ye(face)
    xx, yy = figax.transform_xy(xe, ye)
    for x, y in zip(xx, yy):
        if figax.x_split:
            idx = np.argwhere(np.diff(np.sign(x % 360 - figax.x_split))).flatten()
            x360 = x % 360
            idx = idx[(x360[idx] > 10) & (x360[idx] < 350)]
            start = [0, *(idx+1)]
            end = [*(idx+1), len(x)]
        else:
            start = [0]
            end = [len(x)]
        for s, e in zip(start, end):
            ax.plot(x[s:e], y[s:e], **minor)
    for x, y in zip(xx.transpose(), yy.transpose()):
        if figax.x_split:
            idx = np.argwhere(np.diff(np.sign(x % 360 - figax.x_split))).flatten()
            x360 = x % 360
            idx = idx[(x360[idx] > 10) & (x360[idx] < 350)]
            start = [0, *(idx + 1)]
            end = [*(idx + 1), len(x)]
        else:
            start = [0]
            end = [len(x)]
        for s, e in zip(start, end):
            ax.plot(x[s:e], y[s:e], **minor)

    xx_majors = [xx[0,:], xx[-1, :], xx[:, 0], xx[:, -1]]
    yy_majors = [yy[0,:], yy[-1, :], yy[:, 0], yy[:, -1]]
    for x, y in zip(xx_majors, yy_majors):
        if figax.x_split:
            idx = np.argwhere(np.diff(np.sign(x % 360 - figax.x_split))).flatten()
            x360 = x % 360
            idx = idx[(x360[idx] > 10) & (x360[idx] < 350)]
            start = [0, *(idx+1)]
            end = [*(idx+1), len(x)]
        else:
            start = [0]
            end = [len(x)]
        for s, e in zip(start, end):
            ax.plot(x[s:e], y[s:e], **major)


def plot_columnar_data(figax: FigureAxes, grid: CSDataBase, face, **kwargs):
    xx, yy = figax.transform_xy(grid.xe(face), grid.ye(face))

    mask = np.zeros(grid.data(face).shape, dtype=bool)

    for i, x in enumerate(xx[:-1,:-1]):
        if figax.x_split:
            idx = np.argwhere(np.diff(np.sign(x % 360 - figax.x_split))).flatten()
            x360 = x % 360
            idx = idx[(x360[idx] > 10) & (x360[idx] < 350)]
            mask[i, idx] = True
    for i, x in enumerate(xx[:-1,:-1].transpose()):
        if figax.x_split:
            idx = np.argwhere(np.diff(np.sign(x % 360 - figax.x_split))).flatten()
            x360 = x % 360
            idx = idx[(x360[idx] > 10) & (x360[idx] < 350)]
            mask[idx, i] = True

    data = np.ma.masked_where(mask, grid.data(face).values)
    pc = figax.ax.pcolormesh(xx, yy, data, **kwargs)

def plot_pcolomesh(figax: FigureAxes, xx, yy, data: xr.DataArray, **kwargs):
    data = data.squeeze()

    if len(data.shape) != 2:
        raise ValueError('Data passed to plot_pcolomesh is not 2 dimensional!')

    mask = np.zeros(data.shape, dtype=bool)

    for i, x in enumerate(xx[:-1,:-1]):
        if figax.x_split:
            idx = np.argwhere(np.diff(np.sign(x % 360 - figax.x_split))).flatten()
            x360 = x % 360
            idx = idx[(x360[idx] > 10) & (x360[idx] < 350)]
            mask[i, idx] = True
    for i, x in enumerate(xx[:-1,:-1].transpose()):
        if figax.x_split:
            idx = np.argwhere(np.diff(np.sign(x % 360 - figax.x_split))).flatten()
            x360 = x % 360
            idx = idx[(x360[idx] > 10) & (x360[idx] < 350)]
            mask[idx, i] = True

    data = np.ma.masked_where(mask, data.values)
    figax.ax.pcolormesh(xx, yy, data, **kwargs)


def get_output_data_path(output_dir, fname_template: str, date):
    return os.path.join(output_dir, fname_template.format(date.strftime('%Y%m%d_%H%M')))


# def simple_plot(projection: ccrs.Projection, experiment: Experiment, plot_variable, load_collections, keep_variables, operation_pipe, bounds: tuple=None, coastlines=True) -> plt.Figure:
#     figure = plt.figure()
#
#     ax = plt.subplot(1, 1, 1, projection=projection)
#     figax = FigureAxes(ax, projection)
#
#
#     ax = plt.Axes()
#     if bounds:
#         bounds_ll = bounds[0]  # lower left (x, y)
#         bounds_ur = bounds[1]  # upper right (x, y)
#         bounds_ll = figax.transform_xy(*bounds_ll)
#         bounds_ur = figax.transform_xy(*bounds_ur)
#         ax.set_xlim(bottom=bounds_ll[0], top=bounds_ur[0])
#         ax.set_ylim(bottom=bounds_ll[1], top=bounds_ur[1])
#     else:
#         ax.set_global()
#
#     if coastlines:
#         ax.coastlines(linewidth=0.8)


if __name__ == '__main__':
    # f = plt.figure()
    #
    # #output_dir = '/home/liam/stetson/BS1/gchp_TransportTracers/OutputDir'
    # output_dir = '/home/liam/stetson/BS1/gchp_TransportTracers-C180e/OutputDir'
    # dates = pd.date_range('2016-01-07 0:30:00', periods=1, freq="1H")
    # path = get_output_data_path(output_dir, 'GCHP.SpeciesConc.{}z.nc4', dates[0])
    #
    #
    #
    # d = xr.open_dataset(path)
    # #vmax = np.asscalar(d.SpeciesConc_Rn222.isel(lev=slice(0, 29)).mean(dim='lev')[0,::].max())
    # vmax= 5e-20
    # data = [d.SpeciesConc_Rn222.isel(lev=slice(0, 29)).mean(dim='lev')[0, face, ::] for face in range(6)]
    #
    #
    #
    # #grid = StretchedGrid(48, 15, 36,  360 - 78, data)
    # grid = StretchedGrid(48, 3.75, 36, 360 - 78, data)
    # #grid = CSData(48, data)
    #
    # proj = ccrs.PlateCarree()
    # #proj = ccrs.NearsidePerspective(360 - 78, 36)
    # ax = plt.subplot(1, 1, 1, projection=proj)
    # ax.set_global()
    # ax.coastlines(linewidth=0.8)
    #
    # figax = FigureAxes(ax, proj)
    #
    # for face in range(6):
    #     plot_columnar_data(figax, grid, face, vmin=0, vmax=vmax)
    #     #draw_grid_boxes(figax, grid, face)
    #     draw_face_number(figax, grid, face)
    #
    # plt.show()

    # from sg.experiment import Experiment
    #
    # f = plt.figure()
    #
    # c180e_experiment = Experiment('/home/liam/stetson/BS1/gchp_TransportTracers-C180e/')
    #
    # dates = pd.date_range('2016-01-01 0:30:00', periods=1, freq="1H")
    # ds1 = c180e_experiment.load('SpeciesConc', dates)
    # ds2 = c180e_experiment.load('StateMet_avg', dates)
    #
    # ds1 = c180e_experiment.keep_variables(ds1, 'SpeciesConc_Rn222')
    # ds2 = c180e_experiment.keep_variables(ds2, 'Met_TropLev')
    #
    # ds = ds1.merge(ds2)
    #
    # vmax = 5e-20
    # da = ds['SpeciesConc_Rn222']
    # da = c180e_experiment.select_troposphere(da, ds['Met_TropLev'])
    # da = c180e_experiment.unary_reduction(da, operator='mean')
    # # da = c180e_experiment.unary_reduction(da, levels=slice(0,29))
    # # da = c180e_experiment.unary_reduction(da, operator='mean')
    #
    # data = c180e_experiment.get_face_list(da)
    #
    # grid = StretchedGrid(48, 3.75, 36, 360 - 78, data)
    #
    # proj = ccrs.PlateCarree()
    #
    # ax = plt.subplot(1, 1, 1, projection=proj)
    # ax.set_global()
    # ax.coastlines(linewidth=0.8)
    #
    # figax = FigureAxes(ax, proj)
    #
    # for face in range(6):
    #     plot_columnar_data(figax, grid, face, vmin=0, vmax=vmax)
    #     # draw_grid_boxes(figax, grid, face)
    #     draw_face_number(figax, grid, face)
    #
    # plt.show()

    from sg.experiment import Experiment
    import sg.pipe_operations

    f = plt.figure()

    experiment = Experiment(
        directory='/home/liam/stetson/BS1/gchp_TransportTracers-C180e/',
        grid=StretchedGrid(
            cs=48,
            sf=3.75,
            target_lat=36,
            target_lon=360 - 78
        )
    )
    dates = pd.date_range('2016-01-01 0:30:00', periods=1, freq="1H")

    ds = experiment.load(
        collections=['SpeciesConc', 'StateMet_avg'],
        dates=dates,
        variables=['SpeciesConc_Rn222', 'Met_TropLev']
    )

    vmax = 5e-20

    da = ds['SpeciesConc_Rn222']
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