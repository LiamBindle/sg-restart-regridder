import sys

import yaml

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import pyproj
import numpy as np
import xarray as xr
import pandas as pd

from sg.pipe_operations import *
from sg.experiment import Experiment
from sg.grids import CubeSphere, StretchedGrid
from sg.plot import *
from sg.framer import *


if __name__ == '__main__':
    with open(sys.argv[1], 'r') as f:
        yaml_input = yaml.safe_load(f)

    experiment = Experiment(
        directory=yaml_input['experiment']['directory'],
        grid=eval(yaml_input['experiment']['grid'])
    )

    target = yaml_input['pcolormesh']['target']
    supplemental= yaml_input['pcolormesh']['supplemental']

    dates = pd.date_range(**yaml_input['pcolormesh']['date_range'])

    if isinstance(supplemental, str):
        supplemental = [supplemental]

    collections = list(set([variable.split(':', 1)[0] for variable in [target, *supplemental]]))
    variables = list(set([variable.split(':', 1)[1] for variable in [target, *supplemental]]))
    target = target.split(':', 1)[1]

    vmin = yaml_input['pcolormesh'].get('vmin', 0)
    vmax = yaml_input['pcolormesh'].get('vmax', None)

    for timestamp in dates:
        ds = experiment.load(
            collections=collections,
            date=timestamp.to_pydatetime(),
            variables=variables
        )
        da = ds[target]

        if vmax is None:
            vmax = np.asscalar(da.max().values)

        callable_operation = eval(yaml_input['pcolormesh']['operation'])
        da = callable_operation(da, ds, experiment.key_lut)




        plt.figure(figsize=(16, 8), dpi=100)
        figax = plate_carree(experiment)

        for face in range(6):
            xx, yy = figax.transform_xy(experiment.grid.xe(face), experiment.grid.ye(face))

            face_data = select_face(da, face=face, key_lut=experiment.key_lut)

            # Draw grid
            draw_minor_grid_boxes(figax, xx, yy)
            draw_major_grid_boxes(figax, xx, yy)

            # Plot data
            pcolormesh = plot_pcolomesh(figax, xx, yy, face_data, vmin=vmin, vmax=vmax)
            if face == 0:
                _, _, w, h= figax.ax.get_position().bounds
                cb = plt.colorbar(pcolormesh, fraction=0.02)
                cb.set_label(yaml_input['pcolormesh'].get('units', ''))

            # Draw face number
            text = draw_face_number(figax, xx, yy, face)
            draw_text_stroke(text)
        plt.title(yaml_input['pcolormesh'].get('title', ''))
        plt.tight_layout()
        plt.show()

