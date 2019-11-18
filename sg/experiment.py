from typing import List
import os.path

import xarray as xr
import pandas as pd


class Experiment:
    def __init__(self, directory, templates={}):
        templates.setdefault('output_dir', 'OutputDir/')
        templates.setdefault('timestamp_strftime', '%Y%m%d_%H%M')
        templates.setdefault('collection_fname', 'GCHP.{collection}.{timestamp}z.nc4')
        templates.setdefault('collections_time_dimension', 'time')
        templates.setdefault('collections_level_dimension', 'lev')
        templates.setdefault('collections_face_dimension', 'nf')
        templates.setdefault('collections_x_dimension', 'Xdim')
        templates.setdefault('collections_y_dimension', 'Ydim')
        self.experiment_directory = directory
        self.templates = templates

    @property
    def key_lut(self) -> dict:
        return self.templates

    def load(self, collection: str, dates: pd.DatetimeIndex) -> xr.Dataset:
        files = [os.path.join(
            self.experiment_directory,
            self.templates['output_dir'],
            self.templates['collection_fname'].format(collection=collection, timestamp=date.strftime('%Y%m%d_%H%M'))
        ) for date in dates]
        return xr.open_mfdataset(files, combine='by_coords')

    def unary_reduction(self,
                    da: xr.DataArray,
                    levels=slice(0, None),
                    time=slice(0, None),
                    face=slice(0, None),
                    x=slice(0, None),
                    y=slice(0, None),
                    operator=None, operator_arguments={}):
        operator_arguments.setdefault('dim', self.templates['collections_level_dimension'])

        possible_operators = ['max', 'mean', 'min', 'median', 'prod', 'sum', 'std', 'var']
        if (operator is not None) and not (operator in possible_operators):
            raise ValueError(f'Unknown operator: {str(operator)}')

        indexes = {
            self.templates['collections_level_dimension']: levels,
            self.templates['collections_time_dimension']: time,
            self.templates['collections_face_dimension']: face,
            self.templates['collections_x_dimension']: x,
            self.templates['collections_y_dimension']: y
        }
        indexes = {k: indexes[k] for k in list(da.indexes)}  # drop keys not in the DataArray's indexes
        da = da.isel(**indexes)
        da = da.squeeze()

        if operator == 'max':
            return da.max(**operator_arguments)
        if operator == 'mean':
            return da.mean(**operator_arguments)
        if operator == 'min':
            return da.min(**operator_arguments)
        if operator == 'median':
            return da.median(**operator_arguments)
        if operator == 'prod':
            return da.prod(**operator_arguments)
        if operator == 'sum':
            return da.sum(**operator_arguments)
        if operator == 'std':
            return da.std(**operator_arguments)
        if operator == 'var':
            return da.var(**operator_arguments)

        return da

    def select_troposphere(self, da: xr.DataArray, tropopause_level: xr.DataArray):
        da = da.where(da[self.templates['collections_level_dimension']] < tropopause_level)
        return da

    def select_troposphere_quick(self, da: xr.DataArray, ds: xr.Dataset):
        return self.unary_reduction(da, levels=slice(0, 33))

    def keep_variables(self, ds: xr.Dataset, *variables):
        drop_keys = set(ds.data_vars)
        keep_keys = set(variables)
        drop_keys = drop_keys - keep_keys
        return ds.drop(drop_keys)



    def get_face_list(self, da: xr.DataArray) -> List[xr.DataArray]:
        return [da.isel(**{self.templates['collections_face_dimension']: face}) for face in range(6)]