import os.path
import argparse

import numpy as np
import pandas as pd
import xarray as xr
import sklearn.metrics

from tqdm import tqdm


def central_angle(x0, y0, x1, y1):
    RAD2DEG = 180 / np.pi
    DEG2RAD = np.pi / 180
    x0 = x0 * DEG2RAD
    x1 = x1 * DEG2RAD
    y0 = y0 * DEG2RAD
    y1 = y1 * DEG2RAD
    return np.arccos(np.sin(y0) * np.sin(y1) + np.cos(y0) * np.cos(y1) * np.cos(np.abs(x0-x1))) * RAD2DEG


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('datadir',
                        type=str,)
    parser.add_argument('--date',
                        type=str,
                        default=None)
    parser.add_argument('--overpass-time',
                        type=str,
                        default='13:30Z')
    parser.add_argument('--ascending',
                        action='store_false')
    parser.add_argument('--orbits-per-day',
                        type=float,
                        default=14.2)
    parser.add_argument('--collection',
                        type=str,
                        default='SpeciesConc')
    parser.add_argument('-o',
                        type=str,
                        default='daily_aqs_{species}_comparison.csv')
    args = parser.parse_args()
    args = vars(args)

    base_date = pd.to_datetime(f"{args['date']}T{args['overpass_time']}")
    start_date = base_date - pd.DateOffset(hours=13)
    end_date = base_date + pd.DateOffset(hours=13)

    unresolved_path = 'GCHP.{collection}.{date.year:04d}{date.month:02d}{date.day:02d}_{date.hour:02d}{date.minute:02d}z.nc4'
    files = [os.path.join(args['datadir'], unresolved_path.format(collection=args['collection'], date=date)) for date in pd.date_range(start_date, end_date, freq='30min')]
    # files = [f for f in files if os.path.exists(f)]

    ds = xr.open_mfdataset(
        files,
        combine='nested',
        concat_dim='time',
        compat='override',
        data_vars='minimal',
        coords='minimal',
    )

    lats = ds['lats'].values
    lons = ds['lons'].values
    lons[lons > 180] -= 360
    ds['lons'].values = lons

    overpass_offset = lats/90 * 24/args['orbits_per_day']/4 * 60  # vary overpass time with latitude
    if args['ascending']:
        overpass_offset = -overpass_offset  # overpass delayed at high northern latitudes if ascending

    overpass_time_timedelta_min = lons/360 * 24 * 60 + overpass_offset
    overpass_time_timedelta_min = overpass_time_timedelta_min.astype('timedelta64[m]')
    overpass_time = base_date.to_datetime64() - overpass_time_timedelta_min

    overpass_time_floor_timedelta_min = (np.floor(overpass_time_timedelta_min.astype(float) / 30) * 30).astype('timedelta64[m]')
    overpass_time_ceil_timedelta_min = (np.ceil(overpass_time_timedelta_min.astype(float) / 30) * 30).astype('timedelta64[m]')
    overpass_time_floor = base_date.to_datetime64() - overpass_time_floor_timedelta_min
    overpass_time_ceil = base_date.to_datetime64() - overpass_time_ceil_timedelta_min

    floor_weight = 1 - (overpass_time_timedelta_min.astype(int) % 30) / 30
    ceil_weight = 1 - floor_weight

    ds.coords['overpass_time'] = (['nf', 'Ydim', 'Xdim'], overpass_time)
    ds.coords['overpass_time_floor'] = (['nf', 'Ydim', 'Xdim'], overpass_time_floor)
    ds.coords['overpass_time_ceil'] = (['nf', 'Ydim', 'Xdim'], overpass_time_ceil)
    ds.coords['overpass_time_floor_weight'] = (['nf', 'Ydim', 'Xdim'], floor_weight)
    ds.coords['overpass_time_ceil_weight'] = (['nf', 'Ydim', 'Xdim'], ceil_weight)

    drop_vars = [v for v in ds.data_vars if not set(ds.overpass_time.dims).issubset(set(ds[v].dims))]
    ds = ds.drop(drop_vars)

    floor = ds.sel(time=ds.overpass_time_floor)
    ceil = ds.sel(time=ds.overpass_time_ceil)

    ds_out = floor * ds.overpass_time_floor_weight + ceil * ds.overpass_time_ceil_weight

    ds_out = ds_out.expand_dims('time', 0)
    ds_out = ds_out.assign_coords({'time': [base_date]})
    ds_out = ds_out.transpose('time', 'lev', 'nf', 'Ydim', 'Xdim')


    encoding={}
    for v in ds_out.data_vars:
        ds_out[v].attrs = {
            'long_name': ds[v].attrs['long_name'],
            'units': ds[v].attrs['units'],
        }
        ds_out[v].encoding['coordinates'] = "time nf lev Ydim Xdim"

    fname_out = os.path.join(
        args['datadir'],
        unresolved_path.format(collection=args['collection']+'.OVERPASS', date=base_date)
    )
    for varname in ds_out.data_vars:
        ds_out[varname].attrs = ds[varname].attrs
    ds_out.to_netcdf(fname_out, unlimited_dims=['time'])

    print(ds_out)
