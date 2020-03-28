
import numpy as np
import xarray as xr
from tqdm import tqdm
import argparse
import sklearn.metrics


def ufunc_mean(x, mask):
    if len(x.shape) == 4:
        x = x[:, mask]
    else:
        x = x[mask]
    return np.mean(x, axis=-1)


def ufunc_std(x, mask):
    if len(x.shape) == 4:
        x = x[:, mask]
    else:
        x = x[mask]
    return np.std(x, axis=-1)


def ufunc_rmse(x, y, mask):
    if len(x.shape) == 4:
        x = x[:, mask]
        y = y[:, mask]
    else:
        x = x[mask]
        y = y[mask]
    return np.sqrt(np.sum((y-x)**2, axis=-1)/np.count_nonzero(mask))


def ufunc_mae(x, y, mask):
    if len(x.shape) == 4:
        x = x[:, mask]
        y = y[:, mask]
    else:
        x = x[mask]
        y = y[mask]
    return np.sum(np.abs(y-x), axis=-1)/np.count_nonzero(mask)


def ufunc_mb(x, y, mask):
    if len(x.shape) == 4:
        x = x[:, mask]
        y = y[:, mask]
    else:
        x = x[mask]
        y = y[mask]
    return np.sum(y-x, axis=-1)/np.count_nonzero(mask)


def ufunc_r2(x, y, mask):
    if len(x.shape) == 4:
        # x = x[:, mask]
        # y = y[:, mask]
        if np.count_nonzero(mask) == 0:
            return [np.nan] * x.shape[0]
        r2 = [sklearn.metrics.r2_score(y_true=x[i, mask], y_pred=y[i, mask]) for i in range(x.shape[0])]
    else:
        x = x[mask]
        y = y[mask]
        if np.count_nonzero(mask) == 0:
            return np.nan
        r2 = sklearn.metrics.r2_score(y_true=x, y_pred=y)
    return r2


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
    parser = argparse.ArgumentParser(description='Generates a stretched grid initial restart file for GCHP.')
    parser.add_argument('fin',
                        type=str,
                        help='input file')
    parser.add_argument('-o',
                        type=str,
                        required=True,
                        help='output file')
    parser.add_argument('-id', '--ID',
                        type=str,
                        nargs='+',
                        default=[],
                        help='lines to process')
    parser.add_argument('-m', '--mask',
                        type=str,
                        required=False,
                        help='selects a specific line\'s mask')
    args = parser.parse_args()
    ds = xr.open_dataset(args.fin).squeeze()

    param_keys = ['stretch_factor', 'target_lat', 'target_lon', 'cs_res', 'Met_TropLev', 'Met_PBLTOPL']
    natives = [name for name in ds.data_vars.keys() if '_native_' in name]
    natives.extend([name for name in ds.coords.keys() if '_native_' in name])

    params = xr.Dataset({k: ds[k] for k in param_keys})
    ds = ds.drop([*param_keys, *natives])


    control = ds.sel(ID='CTL')

    processed = xr.Dataset(coords={'ID': []})

    IDs = ds['ID'].values if len(args.ID) == 0 else args.ID

    if args.mask is not None:
        supermask = get_tfmask(ds.sel(ID=args.mask))
    else:
        supermask = None

    for ID in tqdm(IDs):
        if ID == 'CTL':
            continue

        experiment = ds.sel(ID=ID)

        tfmask = get_tfmask(experiment)
        if supermask is not None:
            tfmask &= supermask

        metrics = add_metrics(
            control, experiment, tfmask,
            RMSE=ufunc_rmse,
            MB=ufunc_mb,
            MAE=ufunc_mae,
            R2=ufunc_r2,
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
                metrics.expand_dims({'ID': [ID]}),
                experiment_stats.expand_dims({'ID': [ID]}),
                control_stats.assign_coords({'ID': [ID]})
            ])]
        )

    processed = processed.merge(params)

    encoding = {k: {'dtype': np.float32, 'complevel': 1, 'zlib': True} for k in processed.data_vars}
    processed.to_netcdf(args.o, encoding=encoding)
