import sys
import os.path
import argparse
from datetime import datetime

import numpy as np
import xarray as xr
import xesmf as xe

from gcpy.grid.horiz import make_grid_LL

from sg.transform import scs_transform, make_grid_SCS


def sg_spec_to_str(stretch_factor: float, target_lat: float, target_lon: float):
    return f's{stretch_factor:.1f}_x{target_lon:.1f}_y{target_lat:.1f}'

def make_regridder_L2S(llres_in, csres_out, stretch_factor, target_lat, target_lon, weightsdir='.'):
    csgrid, csgrid_list = make_grid_SCS(csres_out, stretch_factor, target_lat, target_lon)
    llgrid = make_grid_LL(llres_in)
    regridder_list = []
    for i in range(6):
        weightsfile = os.path.join(weightsdir, f'conservative_{llres_in}_c{csres_out}_{sg_spec_to_str(stretch_factor, target_lat, target_lon)}_{i}.nc')
        reuse_weights = os.path.exists(weightsfile)
        regridder = xe.Regridder(llgrid,
                                 csgrid_list[i],
                                 method='conservative',
                                 filename=weightsfile,
                                 reuse_weights=reuse_weights)
        regridder_list.append(regridder)
    return regridder_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates a stretched grid initial restart file for GCHP.')
    parser.add_argument('--stretch-factor',
                        metavar='S',
                        nargs=1,
                        type=float,
                        required=True,
                        help='stretching factor')
    parser.add_argument('--target-lon',
                        metavar='X',
                        nargs=1,
                        type=float,
                        required=True,
                        help='target longitude')
    parser.add_argument('--target-lat',
                        metavar='Y',
                        nargs=1,
                        type=float,
                        required=True,
                        help='target latitude')
    parser.add_argument('--cs-res',
                        metavar='R',
                        nargs=1,
                        type=int,
                        required=True,
                        help='cube-sphere resolution')
    parser.add_argument('--sim',
                        metavar='R',
                        nargs=1,
                        type=str,
                        default=['Standard'],
                        choices=['TransportTracers', 'Standard'],
                        help='GCHP simulation type')
    parser.add_argument('--llres',
                        metavar='R',
                        nargs=1,
                        type=str,
                        default=['2x2.5'],
                        choices=['2x2.5'],
                        help='lat x lon resolution of input restart file')
    parser.add_argument('file_in',
                        type=str,
                        nargs=1,
                        help='path to the input restart file')
    args = parser.parse_args()

    stretch_factor = args.stretch_factor[0]
    csres = args.cs_res[0]
    target_lat = args.target_lat[0]
    target_lon = args.target_lon[0]
    sim = args.sim[0]
    llres = args.llres[0]

    # Open the input dataset
    ds_in = xr.open_dataset(args.file_in[0], decode_cf=False)

    # Regrid
    regridders = make_regridder_L2S(llres, csres, stretch_factor, target_lat, target_lon)
    ds_out = [regridder(ds_in, keep_attrs=True) for regridder in regridders]
    ds_out = xr.concat(ds_out, 'face')

    # Add standard names
    for v in ds_out:
        ds_out[v].attrs['standard_name'] = v

    # Make aliases
    if sim == 'TransportTracers':
        aliases = {
            'SPC_Rn222': 'SPC_Rn',
            'SPC_Pb210': 'SPC_Pb',
            'SPC_Pb210Strat': 'SPC_Pb',
            'SPC_Be7Strat': 'SPC_Be7',
            'SPC_Be10': 'SPC_Be7',
            'SPC_Be10Strat': 'SPC_Be7',
            'SPC_Passive': 'SPC_PASV',
            'SPC_PassiveTracer': 'SPC_PASV',
        }
    elif sim == 'Standard':
        aliases = {}
    else:
        raise ValueError(f'Unknown simulation type: {sim}')
    for alias, real in aliases.items():
        ds_out[alias] = ds_out[real]

    ds_out.attrs['history'] = datetime.now().strftime('%c:') + ' '.join(sys.argv) + '\n' + ds_out.attrs['history']

    # Drop lat and lon
    ds_out = ds_out.drop(['lat', 'lon'])

    # Stack face and y coordinates
    ds_out = ds_out.stack(newy=['face', 'y'])
    ds_out = ds_out.assign_coords(newy=np.linspace(1.0, 6*csres, 6*csres), x=np.linspace(1.0, csres, csres))
    ds_out = ds_out.rename({'newy': 'lat', 'x': 'lon'})

    # Transpose
    ds_out = ds_out.transpose('time', 'lev', 'lat', 'lon')

    # Sort so that lev is in ascending order
    ds_out = ds_out.sortby(['lev'], ascending=True)

    # Change to float32
    for v in ds_out.variables:
        ds_out[v].values = ds_out[v].values.astype(np.float32)

    # Fix coordinate attributes
    ds_out['lev'].attrs = {
        'standard_name': 'level',
        'long_name': 'Level',
        'units': 'eta_level',
        'axis': 'Z',
    }
    ds_out['lat'].attrs = {
        'standard_name': 'latitude',
        'long_name': 'Latitude',
        'units': 'degrees_north',
        'axis': 'Y',
    }
    ds_out['lon'].attrs = {
        'standard_name': 'longitude',
        'long_name': 'Longitude',
        'units': 'degrees_east',
        'axis': 'X',
    }
    ds_out['time'].attrs = {
        'standard_name': 'time',
        'long_name': 'Time',
        'units': 'hours since 1985-1-1 00:00:0.0',
        'axis': 'T',
        'calendar': 'gregorian',
    }

    # Write dataset
    ds_out.to_netcdf(
        f'initial_GEOSChem_rst.c{csres}_{sg_spec_to_str(stretch_factor, target_lat, target_lon)}_TransportTracers.nc',
        format='NETCDF4_CLASSIC'
    )
