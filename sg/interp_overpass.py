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

    lons = ds['lons'].values
    lons[lons > 180] -= 360
    ds['lons'].values = lons

    solar_time_timedelta_min = lons/360 * 24 * 60
    solar_time_timedelta_min = solar_time_timedelta_min.astype('timedelta64[m]')
    solar_time = solar_time_timedelta_min + base_date.to_datetime64()

    solar_time_floor_timedelta_min = (np.floor(solar_time_timedelta_min.astype(float) / 30) * 30).astype('timedelta64[m]')
    solar_time_ceil_timedelta_min = (np.ceil(solar_time_timedelta_min.astype(float) / 30) * 30).astype('timedelta64[m]')
    solar_time_floor = solar_time_floor_timedelta_min + base_date.to_datetime64()
    solar_time_ceil = solar_time_ceil_timedelta_min + base_date.to_datetime64()

    floor_weight = 1 - (solar_time_timedelta_min.astype(int) % 30) / 30
    ceil_weight = 1 - floor_weight

    ds.coords['solar_time'] = (['nf', 'Ydim', 'Xdim'], solar_time)
    ds.coords['solar_time_floor'] = (['nf', 'Ydim', 'Xdim'], solar_time_floor)
    ds.coords['solar_time_ceil'] = (['nf', 'Ydim', 'Xdim'], solar_time_ceil)
    ds.coords['solar_time_floor_weight'] = (['nf', 'Ydim', 'Xdim'], floor_weight)
    ds.coords['solar_time_ceil_weight'] = (['nf', 'Ydim', 'Xdim'], ceil_weight)

    drop_vars = [v for v in ds.data_vars if not set(ds.solar_time.dims).issubset(set(ds[v].dims))]
    ds = ds.drop(drop_vars)

    floor = ds.sel(time=ds.solar_time_floor)
    ceil = ds.sel(time=ds.solar_time_ceil)

    ds_out = floor * ds.solar_time_floor_weight + ceil * ds.solar_time_ceil_weight

    fname_out = os.path.join(
        args['datadir'],
        unresolved_path.format(collection=args['collection']+'.OVERPASS', date=base_date)
    )
    ds_out.to_netcdf(fname_out)

    print(ds_out)
