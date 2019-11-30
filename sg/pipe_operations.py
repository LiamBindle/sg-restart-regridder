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


def m2_to_km2(da: xr.DataArray,  supplemental: xr.Dataset, key_lut: dict, **kwargs):
    return da / 1e6


def square_root(da: xr.DataArray,  supplemental: xr.Dataset, key_lut: dict, **kwargs):
    return np.sqrt(da)


def mask_not_tropopause_layer(da: xr.DataArray,  supplemental: xr.Dataset, key_lut: dict, **kwargs):
    da_level = da[key_lut['collections_level_dimension']]
    tropopause_level_key = key_lut['tropopause_level_key']
    if tropopause_level_key in supplemental:
        tropopause_level = supplemental[key_lut['tropopause_level_key']]
    else:
        tropopause_level = 36
    da = da.where(da_level != tropopause_level)
    return da


def mask_not_pbl(da: xr.DataArray,  supplemental: xr.Dataset, key_lut: dict, **kwargs):
    da_level = da[key_lut['collections_level_dimension']]
    pbl_level = (supplemental[key_lut['pbl_level_key']] + 0.5).astype(int)
    da = da.where(da_level <= pbl_level)
    return da


def pbl(da: xr.DataArray,  supplemental: xr.Dataset, key_lut: dict, **kwargs):
    return da.pipe(
        mask_not_pbl,
        supplemental=supplemental,
        key_lut=key_lut
    ).pipe(
        vertical_average,
        key_lut=key_lut
    )


def layer18(da: xr.DataArray,  supplemental: xr.Dataset, key_lut: dict, **kwargs):
    da_level = key_lut['collections_level_dimension']
    return da.isel(**{da_level: 18})


def surface_level(da: xr.DataArray,  supplemental: xr.Dataset, key_lut: dict, **kwargs):
    return da.pipe(
        mask_not_pbl,
        supplemental=supplemental,
        key_lut=key_lut
    ).pipe(
        vertical_average,
        key_lut=key_lut
    )


def tropopause_layer(da: xr.DataArray,  supplemental: xr.Dataset, key_lut: dict, **kwargs):
    tropopause_level = supplemental[key_lut['tropopause_level_key']]
    return da.isel(**{key_lut['collections_level_dimension']: tropopause_level})


def per_m2(da: xr.DataArray, supplemental: xr.Dataset, key_lut: dict, **kwargs):
    return da / supplemental[key_lut['grid_box_area_key']]


def multiply_area(da: xr.DataArray, supplemental: xr.Dataset, key_lut: dict, **kwargs):
    return da * supplemental[key_lut['grid_box_area_key']]


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


def surface_layer(da: xr.DataArray,  supplemental: xr.Dataset, key_lut: dict, **kwargs):
    da_level = da[key_lut['collections_level_dimension']]
    return da.isel(**{da_level: 0})
