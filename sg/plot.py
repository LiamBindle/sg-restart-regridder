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


def draw_face_number(figax: FigureAxes, xx, yy, face, stroke_width=1, **kwargs):
    middle = np.array(xx.shape, dtype=np.int) // 2
    text = figax.ax.text(xx[middle[0], middle[1]], yy[middle[0], middle[1]], f'{face}', **kwargs)
    return text


def draw_text_stroke(text, **kwargs):
    kwargs.setdefault('linewidth', 1)
    kwargs.setdefault('foreground', 'white')
    text.set_path_effects([path_effects.Stroke(**kwargs),
                           path_effects.Normal()])


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
    return figax.ax.pcolormesh(xx, yy, data, **kwargs)
