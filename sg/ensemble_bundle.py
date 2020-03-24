import argparse
import os

import warnings

import yaml
from tqdm import tqdm

import numpy as np
import xarray as xr

from dask.diagnostics import ProgressBar

from sg.grids import CubeSphere, StretchedGrid
from sg.compare_grids import many_comparable_gridboxes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='foo')

    parser.add_argument('output_files',
                        type=str,
                        nargs='+',
                        help="files names in the output directories to look in")
    parser.add_argument('-cp', '--ctl-prefix',
                        metavar='J',
                        type=str,
                        required=True,
                        help='path to the control run directory')
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
    parser.add_argument('--sum',
                        metavar='V',
                        nargs='+',
                        type=str,
                        action='append',
                        required=False,
                        help='variables to keep')
    parser.add_argument('--keep',
                        metavar='V',
                        nargs='+',
                        type=str,
                        required=False,
                        help='variables to keep')
    args = vars(parser.parse_args())
    warnings.simplefilter("ignore")

    with open('simulations.yaml', 'r') as f:
        sims = yaml.safe_load(f)

    with open(os.path.join(args['ctl_prefix'], 'conf.yml')) as f:
        ctl_conf = yaml.safe_load(f)

    ctl_res = ctl_conf['grid']['cs_res']
    ctl_grid = CubeSphere(ctl_res)

    nlev = 72

    ds_ctl = {
        fname: xr.open_dataset(os.path.join(args['ctl_prefix'], 'OutputDir', fname)) for fname in args['output_files']
    }

    ds_out = xr.Dataset(
        coords={
            'lev': range(nlev),
            'face': range(6),
            'Ydim': range(ctl_res),
            'Xdim': range(ctl_res),
            'ID': [*[sim['short_name'] for sim in sims], 'CTL']
        }
    )

    sub_dims = ['lev', 'face', 'Ydim', 'Xdim']
    sub_coords = {k: ds_out.coords[k] for k in sub_dims}

    for sim in tqdm(sims):
        sim_conf = os.path.join(sim['short_name'], 'conf.yml')

        if not os.path.exists(sim_conf):
            continue

        with open(sim_conf, 'r') as f:
            exp_conf = yaml.safe_load(f)['grid']
        exp_grid = StretchedGrid(exp_conf['cs_res'], exp_conf['stretch_factor'], exp_conf['target_lat'], exp_conf['target_lon'])

        ctl_indexes, exp_indexes, weights = many_comparable_gridboxes(
            ctl_grid,
            exp_grid
        )
        ctl_indexes = tuple(zip(*ctl_indexes))

        for output_file in args["output_files"]:
            output_filename = os.path.join(sim['short_name'], 'OutputDir', output_file)
            if not os.path.exists(output_filename):
                continue
            ds_exp = xr.open_dataset(output_filename)
            for var in args["vars"]:
                if var not in ds_exp:
                    continue

                exp_on_ctl = xr.DataArray(np.nan, coords=sub_coords, dims=sub_dims)

                for lev in range(nlev):
                    da_exp = ds_exp[var].squeeze().isel(lev=lev, nf=5).transpose('Ydim', 'Xdim').values
                    regridded = [np.dot(w, da_exp[i]) for w, i in zip(weights, exp_indexes)]
                    exp_on_ctl.isel(lev=lev).values[ctl_indexes] = [
                        np.dot(w, da_exp[i]) for w, i in zip(weights, exp_indexes)
                    ]
                ds_out = ds_out.merge({var: exp_on_ctl.expand_dims('ID', 0).assign_coords({'ID': [sim['short_name']]})})

    for var in args["vars"]:
        for output_file in args['output_files']:
            if var in ds_ctl[output_file]:
                ds_out = ds_out.merge({
                    var: ds_ctl[output_file][var].expand_dims(
                        'ID', 0
                    ).rename({
                        'nf': 'face'
                    }).assign_coords({
                        'ID': ['CTL'], **sub_coords
                    })})

    if args['sum'] is not None:
        for s in args['sum']:
            ds_out[s[0]] = ds_out[s[1]]
            for ds_index in range(2,len(s)):
                ds_out[s[0]] += ds_out[s[ds_index]]

    if args['keep'] is not None:
        ds_out = ds_out.drop([label for label in ds_out.data_vars if label not in args['keep']])

    encoding = {k: {'dtype': np.float32, 'complevel': 9, 'zlib': True} for k in ds_out.data_vars}
    delayed_obj = ds_out.to_netcdf(os.path.join(args["output_prefix"], 'summary.nc'), encoding=encoding, compute=False)
    with ProgressBar():
        delayed_obj.compute()
