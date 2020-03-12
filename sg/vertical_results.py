
import numpy as np
import xarray as xr
from tqdm import tqdm


def ufunc_mean(x, mask=None):
    if mask is None:
        mask = np.isfinite(x[0, :, :, :])
    return np.mean(x[:, mask], axis=-1)


def ufunc_std(x, mask=None):
    if mask is None:
        mask = np.isfinite(x[0, :, :, :])
    return np.std(x[:, mask], axis=-1)


def ufunc_rmse(x, y, mask):
    x = x[:, mask]
    y = y[:, mask]
    return np.sqrt(np.sum((y-x)**2, axis=-1)/np.count_nonzero(mask))


def ufunc_mae(x, y, mask):
    x = x[:, mask]
    y = y[:, mask]
    return np.sum(np.abs(y-x), axis=-1)/np.count_nonzero(mask)


def ufunc_mb(x, y, mask):
    x = x[:, mask]
    y = y[:, mask]
    return np.sum(y-x, axis=-1)/np.count_nonzero(mask)


def get_tfmask(e):
    return np.isfinite(e[list(e.data_vars.keys())[0]].values[0, :, :, :])


def add_metrics(c, e, mask, **ufuncs):
    ds = xr.Dataset()
    for metric, ufunc in ufuncs.items():
        score = xr.apply_ufunc(ufunc, c, e, mask, input_core_dims=[['face', 'Ydim', 'Xdim'],['face', 'Ydim', 'Xdim'],[]])
        # score = score.expand_dims({'lineno': e.lineno}, axis=0)
        score = score.expand_dims({'metric': [metric]}, axis=0)
        ds = ds.merge(score)
    return ds


def add_stats(d, mask, **ufuncs):
    ds = xr.Dataset()
    for metric, ufunc in ufuncs.items():
        score = xr.apply_ufunc(ufunc, d, mask, input_core_dims=[['face', 'Ydim', 'Xdim'],[]])
        ds = ds.merge(score.expand_dims({'metric': [metric]}, axis=0))
    return ds


if __name__ == '__main__':
    import sys
    ds = xr.open_dataset(sys.argv[1])

    control = ds.sel(lineno=0)

    processed = xr.Dataset(coords={'lineno': []})

    for lineno in tqdm(ds['lineno'][:5]):
        if lineno == 0:
            continue

        experiment = ds.sel(lineno=lineno)

        tfmask = get_tfmask(experiment)

        metrics = add_metrics(
            control, experiment, tfmask,
            RMSE=ufunc_rmse,
            MB=ufunc_mb,
            MAE=ufunc_mae,
        )

        experiment_stats = add_stats(
            experiment, tfmask,
            EXP_MEAN=ufunc_mean,
            EXP_STD=ufunc_std,
        )

        control_stats = add_stats(
            control, tfmask,
            CTL_MEAN=ufunc_mean,
            CTL_STD=ufunc_std,
        )

        processed = xr.merge([
            processed,
            xr.merge([
                metrics.expand_dims({'lineno': [lineno]}),
                experiment_stats.expand_dims({'lineno': [lineno]}),
                control_stats.assign_coords({'lineno': [lineno]})
            ])]
        )

    encoding = {k: {'dtype': np.float32, 'complevel': 1, 'zlib': True} for k in processed.data_vars}
    processed.to_netcdf(sys.argv[2], encoding=encoding)
