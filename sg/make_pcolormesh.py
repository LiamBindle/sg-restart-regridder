import sys
import yaml

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

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


if __name__ == '__main__':
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
            #draw_minor_grid_boxes(figax, xx, yy)
            #draw_major_grid_boxes(figax, xx, yy)

            # Plot data
            pcolormesh = plot_pcolomesh(figax, xx, yy, face_data, vmin=vmin, vmax=vmax)
            if face == 0:
                _, _, w, h= figax.ax.get_position().bounds
                cb = plt.colorbar(pcolormesh, fraction=0.02)
                cb.set_label(yaml_input['pcolormesh'].get('units', ''))

            # Draw face number
            # text = draw_face_number(figax, xx, yy, face)
            # draw_text_stroke(text)
        plt.title(yaml_input['pcolormesh'].get('title', ''))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, output_fname.format(timestamp=timestamp)))
        plt.close()

