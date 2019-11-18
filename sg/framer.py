import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from sg.figure_axes import FigureAxes
from sg.experiment import Experiment


def plate_carree(experiment: Experiment, coastlines={'linewidth': 0.8}):
    proj = ccrs.PlateCarree()
    ax = plt.subplot(1, 1, 1, projection=proj)
    ax.set_global()
    ax.coastlines(**coastlines)
    figax = FigureAxes(ax, proj)
    return figax


def nearside_perspective(experiment: Experiment):
    proj = ccrs.NearsidePerspective(experiment.grid.target_lon, experiment.grid.target_lat)
    ax = plt.subplot(1, 1, 1, projection=proj)
    ax.set_global()
    ax.coastlines(linewidth=0.8)
    figax = FigureAxes(ax, proj)
    return figax