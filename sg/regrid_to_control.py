import argparse
import logging
import re
import os

import warnings

import numpy as np
import xarray as xr

from sg.grids import CubeSphere, StretchedGrid
from sg.compare_grids import many_comparable_gridboxes




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='foo')

    parser.add_argument('output_files',
                        type=str,
                        nargs='+',
                        help="files names in the output directories to look in")
    parser.add_argument('-ep', '--exp-prefix',
                        metavar='K',
                        type=str,
                        required=True,
                        help='path to the experiment\'s output directory')
    parser.add_argument('-cp', '--ctl-prefix',
                        metavar='J',
                        type=str,
                        required=True,
                        help='path to the control\'s output directory')
    parser.add_argument('-o', '--output-prefix',
                        metavar='O',
                        type=str,
                        required=True,
                        help='path to output prefix where the generate file will go')
    parser.add_argument('-v', '--vars',
                        metavar='C',
                        nargs='+',
                        type=str,
                        required=True,
                        help='variables to keep')
    args = vars(parser.parse_args())
    logging.basicConfig(level=logging.INFO)
    warnings.simplefilter("ignore")
    logging.info('Opening input datasets...')

    lineno = int(os.path.basename(os.path.dirname(args["exp_prefix"])))

    exp_res = len(xr.open_dataset(f'{args["exp_prefix"]}/{args["output_files"][0]}').Ydim)
    ctl_res = len(xr.open_dataset(f'{args["ctl_prefix"]}/{args["output_files"][0]}').Ydim)
    nlev = len(xr.open_dataset(f'{args["ctl_prefix"]}/{args["output_files"][0]}').lev)

    with open(f'{args["exp_prefix"]}/../configure.sh', 'r') as conf:
        conf_txt=conf.read()
        sf = re.search(r'STRETCH_FACTOR=([\d.]+)', conf_txt).group(1)
        lat0 = re.search(r'TARGET_LAT=([-\d.]+)', conf_txt).group(1)
        lon0 = re.search(r'TARGET_LON=([-\d.]+)', conf_txt).group(1)
        sf = float(sf)
        lat0 = float(lat0)
        lon0 = float(lon0)

    logging.info('Computing grid-box intersections...')
    ctl_indexes, exp_indexes, weights = many_comparable_gridboxes(
        CubeSphere(ctl_res),
        StretchedGrid(exp_res, sf, lat0, lon0)
    )
    ctl_indexes = tuple(zip(*ctl_indexes))

    ds_out = xr.Dataset(
        coords={'lev': range(nlev), 'face': range(6), 'Ydim': range(ctl_res), 'Xdim': range(ctl_res), 'lineno': [lineno]}
    )

    logging.info('Slicing data...')
    for output_file in args["output_files"]:
        ds_exp = xr.open_dataset(f'{args["exp_prefix"]}/{output_file}')
        ds_ctl = xr.open_dataset(f'{args["ctl_prefix"]}/{output_file}')
        for var in args["vars"]:
            if var not in ds_exp:
                continue

            sub_dims = ['lev', 'face', 'Ydim', 'Xdim']
            sub_coords = {k: ds_out.coords[k] for k in sub_dims}
            exp_on_ctl = xr.DataArray(np.nan, coords=sub_coords, dims=sub_dims)
            #exp_on_ctl_var = xr.DataArray(np.nan, coords=sub_coords, dims=sub_dims)

            for lev in range(nlev):
                da_exp = ds_exp[var].squeeze().isel(lev=lev, nf=5).transpose('Ydim', 'Xdim').values
                regridded = [np.dot(w, da_exp[i]) for w, i in zip(weights, exp_indexes)]
                exp_on_ctl.isel(lev=lev).values[ctl_indexes] = [np.dot(w, da_exp[i]) for w, i in zip(weights, exp_indexes)]
                # exp_on_ctl_var.isel(lev=lev).values[ctl_indexes] = [
                #     np.average((da_exp[y_idx] - ymean) ** 2, weights=w) for y_idx, ymean, w in zip(exp_indexes, regridded, weights)
                # ]

            # ds_out[var + '_SUBGRID_VARIANCE'] = xr.DataArray(np.nan, coords=sub_coords, dims=sub_dims)
            ds_out[var] = xr.DataArray(np.nan, coords=sub_coords, dims=sub_dims)
            ds_out[var] = exp_on_ctl
            # ds_out[var + '_SUBGRID_VARIANCE'] = exp_on_ctl_var

    logging.info('Writing output files...')
    fname = f'{args["output_prefix"]}/lineno-{lineno}.nc'
    encoding = { k: {'dtype': np.float32, 'complevel': 9, 'zlib': True} for k in ds_out.data_vars }
    ds_out.to_netcdf(fname, encoding=encoding)
    logging.info('Done') # REPLACE_EXP_OUTPUT_DIR REPLACE_CTL_OUTPUT_DIR REPLACE_RESULTS_DIR