import argparse
import hashlib
import json
import logging

import warnings

import numpy as np
import xarray as xr
import pandas as pd
import sklearn.metrics

from sg.grids import CubeSphere, StretchedGrid
from sg.compare_grids import many_comparable_gridboxes

# From https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy
def weighted_quantile(values, quantiles, sample_weight=None,
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='foo')
    parser.add_argument('experiment_output',
                        type=str,
                        help="path to experiment's output")
    parser.add_argument('control_output',
                        type=str,
                        help="path to control's output")
    parser.add_argument('-sf','--stretch-factor',
                        metavar='S',
                        type=float,
                        required=True,
                        help='stretching factor')
    parser.add_argument('-lon0','--target-lon',
                        metavar='X',
                        type=float,
                        required=True,
                        help='target longitude')
    parser.add_argument('-lat0', '--target-lat',
                        metavar='Y',
                        type=float,
                        required=True,
                        help='target latitude')
    parser.add_argument('-l', '--levels',
                        metavar='C',
                        type=int,
                        nargs='+',
                        required=True,
                        help='model level')
    parser.add_argument('-v', '--vars',
                        metavar='C',
                        nargs='+',
                        type=str,
                        required=True,
                        help='variables to keep')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    warnings.simplefilter("ignore")
    logging.info('Opening input datasets...')

    ds_exp = xr.open_dataset(args.experiment_output)
    ds_ctl = xr.open_dataset(args.control_output)

    ctl_res = len(ds_ctl.Ydim)
    exp_res = len(ds_exp.Ydim)

    logging.info('Computing grid-box intersections...')
    ctl_indexes, exp_indexes, weights = many_comparable_gridboxes(
        CubeSphere(ctl_res),
        StretchedGrid(exp_res, args.stretch_factor, args.target_lat, args.target_lon)
    )

    df = {}

    logging.info('Slicing data...')
    for var in args.vars:
        for level in args.levels:
            da_ctl = ds_ctl[var].squeeze().isel(lev=level).transpose('nf', 'Ydim', 'Xdim').values
            da_exp = ds_exp[var].squeeze().isel(lev=level, nf=5).transpose('Ydim', 'Xdim').values
            df[f'{var}:CTL:{level}'] = [da_ctl[i] for i in ctl_indexes]
            df[f'{var}:EXP:{level}'] = [np.dot(w, da_exp[i]) for w, i in zip(weights, exp_indexes)]
            df[f'{var}:EXP:{level}:VAR'] = [
                np.average((da_exp[y_idx] - ymean) ** 2, weights=w) for y_idx, ymean, w in zip(exp_indexes, df[f'{var}:EXP:{level}'], weights)
            ]

            r2 = sklearn.metrics.r2_score(
                y_true=df[f'{var}:CTL:{level}'],
                y_pred=df[f'{var}:EXP:{level}']
            )
            rmse = np.sqrt(sklearn.metrics.mean_squared_error(

                y_true=df[f'{var}:CTL:{level}'],
                y_pred=df[f'{var}:EXP:{level}']
            )) / np.mean(df[f'{var}:CTL:{level}'])
            mae = sklearn.metrics.mean_absolute_error(
                y_true=df[f'{var}:CTL:{level}'],
                y_pred=df[f'{var}:EXP:{level}']
            ) / np.mean(df[f'{var}:CTL:{level}'])
            logging.info(f'Metrics: {var}(lev={level}): r2={r2:4.2f}, nrmse={rmse:4.2f}, nmae={mae:4.2f}')

    df = pd.DataFrame(df)

    sg_hash = hashlib.sha1('sf={stretch_factor:.5f},tx={target_lon:.5f},ty={target_lat:.5f}'.format(
        stretch_factor=args.stretch_factor,
        target_lat=args.target_lat,
        target_lon=args.target_lon,
    ).encode()).hexdigest()[:7]

    logging.info('Writing output files...')

    fname = f'sg{sg_hash}-c{ctl_res}'
    df.to_csv(f'{fname}.csv', index=False)
    metadata = {}
    metadata['ctl_grid'] = {
        'type': 'cubed-sphere',
        'cs': ctl_res,
    }
    metadata['exp_grid'] = {
        'type': 'stretched-grid',
        'cs': exp_res,
        'sf': args.stretch_factor,
        'target_lat': args.target_lat,
        'target_lon': args.target_lon,
    }
    metadata['ctl_datafile'] = args.control_output
    metadata['exp_datafile'] = args.experiment_output
    metadata['variables'] = args.vars
    metadata['levels'] = args.levels

    with open(f'{fname}.json', 'w') as f:
        json.dump(metadata, f)