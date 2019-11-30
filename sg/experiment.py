from typing import List
import os.path
from datetime import datetime
import functools
import operator

import xarray as xr
import pandas as pd

import sg.pipe_operations
from sg.grids import CSDataBase

class Experiment:
    def __init__(self, directory, grid: CSDataBase, templates={}):
        templates.setdefault('output_dir', 'OutputDir/')
        templates.setdefault('timestamp_strftime', '%Y%m%d_%H%M')
        templates.setdefault('collection_fname', 'GCHP.{collection}.{timestamp}z.nc4')
        templates.setdefault('collections_time_dimension', 'time')
        templates.setdefault('collections_level_dimension', 'lev')
        templates.setdefault('collections_face_dimension', 'nf')
        templates.setdefault('collections_x_dimension', 'Xdim')
        templates.setdefault('collections_y_dimension', 'Ydim')
        templates.setdefault('tropopause_level_key', 'Met_TropLev')
        templates.setdefault('pbl_level_key', 'Met_PBLTOPL')
        templates.setdefault('grid_box_area_key', 'Met_AREAM2')
        self.experiment_directory = directory
        self.templates = templates
        self.grid = grid

    @property
    def key_lut(self) -> dict:
        return self.templates

    def load(self, collections: list, date: datetime, variables) -> xr.Dataset:
        files = [os.path.join(
            self.experiment_directory,
            self.templates['output_dir'],
            self.templates['collection_fname'].format(collection=collection, timestamp=date.strftime('%Y%m%d_%H%M'))
        ) for collection in collections]
        #datasets = [xr.open_mfdataset(collection_files, combine='by_coords') for collection_files in files]
        datasets = [xr.open_mfdataset(collection_files) for collection_files in files]
        datasets = [sg.pipe_operations.drop_all_except(ds, *variables) for ds in datasets]
        dataset = xr.Dataset()
        for ds in datasets:
            dataset = dataset.merge(ds)
        return dataset