import yaml
import argparse
import os.path

import xarray as xr
import numpy as np
from dask.diagnostics import ProgressBar
from sg.compare_grids2 import central_angle


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates a sparse intersect matrix')
    parser.add_argument('-c',
                        metavar='CTL_PATH',
                        type=str,
                        required=True,
                        help='path to the control output file')
    parser.add_argument('-e',
                        metavar='EXP_PATH',
                        type=str,
                        required=True,
                        help='path to the experiment run directory')
    parser.add_argument('-o',
                        metavar='OUT',
                        type=str,
                        required=True,
                        help='name of output')
    args = vars(parser.parse_args())

    DEG2RAD = np.pi/180
    R_EARTH=6378.1e3

    ds = xr.open_dataset(args['c'])

    with open(os.path.join(args['e'], 'conf.yml'), 'r') as f:
        exp_conf = yaml.safe_load(f)

    da = xr.apply_ufunc(
        central_angle,
        ds.lons, ds.lats, exp_conf['grid']['target_lon'], exp_conf['grid']['target_lat'],
        input_core_dims=[['nf', 'Ydim', 'Xdim'],['nf', 'Ydim', 'Xdim'],[],[]],
        output_core_dims=[['nf', 'Ydim', 'Xdim']]
    )

    da.attrs['long_name'] = 'distance from grid-box center to target'
    da.attrs['units'] = 'm'
    da = da.drop(['lats', 'lons'])

    ds_out = xr.Dataset({'distance_from_target': da})
    ds_out = ds_out.drop(['nf', 'Ydim', 'Xdim'])
    encoding = {k: {'dtype': np.float32, 'complevel': 9, 'zlib': True} for k in ds_out.data_vars}
    ds_out.to_netcdf(args['o'], encoding=encoding)


