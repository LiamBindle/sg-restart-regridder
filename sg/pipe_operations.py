import numpy as np
import xarray as xr



def noop(da: xr.DataArray, **kwargs):
    return da


def mask_stratosphere(da: xr.DataArray, supplemental: xr.Dataset, key_lut: dict, **kwargs):
    da_level = da[key_lut['collections_level_dimension']]
    tropopause_level_key = key_lut['tropopause_level_key']
    if tropopause_level_key in supplemental:
        tropopause_level = supplemental[key_lut['tropopause_level_key']]
    else:
        tropopause_level = 36
    da = da.where(da_level < tropopause_level)
    return da


def mask_troposphere(da: xr.DataArray,  supplemental: xr.Dataset, key_lut: dict, **kwargs):
    da_level = da[key_lut['collections_level_dimension']]
    tropopause_level_key = key_lut['tropopause_level_key']
    if tropopause_level_key in supplemental:
        tropopause_level = supplemental[key_lut['tropopause_level_key']]
    else:
        tropopause_level = 36
    da = da.where(da_level > tropopause_level)
    return da


def vertical_average(da: xr.DataArray, key_lut: dict, **kwargs):
    return da.mean(dim=key_lut['collections_level_dimension'])


def select_face(da: xr.DataArray, face: int, key_lut: dict, **kwargs):
    return da.isel(**{key_lut['collections_face_dimension']: face})


def drop_all_except(ds: xr.Dataset, *variables):
    drop_keys = set(ds.data_vars)
    keep_keys = set(variables)
    drop_keys = drop_keys - keep_keys
    return ds.drop(drop_keys)


def tropospheric_average(da: xr.DataArray,  supplemental: xr.Dataset, key_lut: dict, **kwargs):
    return da.pipe(
        mask_stratosphere,
        supplemental=supplemental,
        key_lut=key_lut
    ).pipe(
        vertical_average,
        key_lut=key_lut
    )


def per_m2(da: xr.DataArray, supplemental: xr.Dataset, key_lut: dict, **kwargs):
    return da / supplemental[key_lut['grid_box_area_key']]


def log10(da: xr.DataArray, supplemental: xr.Dataset, key_lut: dict, **kwargs):
    return np.log10(da)


def stratospheric_average(da: xr.DataArray,  supplemental: xr.Dataset, key_lut: dict, **kwargs):
    return da.pipe(
        mask_troposphere,
        supplemental=supplemental,
        key_lut=key_lut
    ).pipe(
        vertical_average,
        key_lut=key_lut
    )