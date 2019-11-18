import xarray as xr


def mask_stratosphere(da: xr.DataArray, tropopause_level: xr.DataArray, key_lut: dict, **kwargs):
    da_level = da[key_lut['collections_level_dimension']]
    da = da.where(da_level < tropopause_level)
    return da


def mask_troposphere(da: xr.DataArray, tropopause_level: xr.DataArray, key_lut: dict, **kwargs):
    da_level = da[key_lut['collections_level_dimension']]
    da = da.where(da_level > tropopause_level)
    return da


def vertical_average(da: xr.DataArray, key_lut: dict, **kwargs):
    return da.mean(dim=key_lut['collections_level_dimension'])


def drop_all_except(ds: xr.Dataset, *variables):
    drop_keys = set(ds.data_vars)
    keep_keys = set(variables)
    drop_keys = drop_keys - keep_keys
    return ds.drop(drop_keys)
