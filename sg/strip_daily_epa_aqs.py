import os.path
import argparse

import numpy as np
import pandas as pd
import xarray as xr
import sklearn.metrics

from tqdm import tqdm


def load_daily(year=2016, product=42602):

    return aqs

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
    parser.add_argument('-a', '--aqs',
                        type=str,
                        default='daily_42602_2016.csv',)
    parser.add_argument('-v', '--var',
                        type=str,
                        default='SpeciesConc_NO2',)
    parser.add_argument('--start-date',
                        type=str,
                        default=None)
    parser.add_argument('--end-date',
                        type=str,
                        default=None)
    parser.add_argument('-o',
                        type=str,
                        default='daily_aqs_comparison.csv')
    args = parser.parse_args()
    aqs = pd.read_csv(args.aqs)
    aqs['Date Local'] = pd.to_datetime(aqs['Date Local'])
    aqs['Date of Last Change'] = pd.to_datetime(aqs['Date of Last Change'])

    aqs = aqs.loc[aqs['State Name'] == 'California']    # Only California
    aqs = aqs.loc[aqs['Observation Count'] == 24]       # Only full samples

    aqs = aqs.set_index(['Date Local', 'Site Num']).sort_index()

    diag_fpath = [
        # 'GCHP.SpeciesConc.{date.year:04d}{date.month:02d}{date.day:02d}_0830z.nc4',
        # 'GCHP.SpeciesConc.{date.year:04d}{date.month:02d}{date.day:02d}_0930z.nc4',
        # 'GCHP.SpeciesConc.{date.year:04d}{date.month:02d}{date.day:02d}_1030z.nc4',
        # 'GCHP.SpeciesConc.{date.year:04d}{date.month:02d}{date.day:02d}_1130z.nc4',
        # 'GCHP.SpeciesConc.{date.year:04d}{date.month:02d}{date.day:02d}_1230z.nc4',
        # 'GCHP.SpeciesConc.{date.year:04d}{date.month:02d}{date.day:02d}_1330z.nc4',
        # 'GCHP.SpeciesConc.{date.year:04d}{date.month:02d}{date.day:02d}_1430z.nc4',
        # 'GCHP.SpeciesConc.{date.year:04d}{date.month:02d}{date.day:02d}_1530z.nc4',
        'GCHP.SpeciesConc.{date.year:04d}{date.month:02d}{date.day:02d}_1630z.nc4',
        'GCHP.SpeciesConc.{date.year:04d}{date.month:02d}{date.day:02d}_1730z.nc4',
        # 'GCHP.SpeciesConc.{date.year:04d}{date.month:02d}{date.day:02d}_1830z.nc4',
        # 'GCHP.SpeciesConc.{date.year:04d}{date.month:02d}{date.day:02d}_1930z.nc4',
        # 'GCHP.SpeciesConc.{date.year:04d}{date.month:02d}{date.day:02d}_2030z.nc4',
        # 'GCHP.SpeciesConc.{date.year:04d}{date.month:02d}{date.day:02d}_2130z.nc4',
        # 'GCHP.SpeciesConc.{date.year:04d}{date.month:02d}{date.day:02d}_2230z.nc4',
        # 'GCHP.SpeciesConc.{date.year:04d}{date.month:02d}{date.day:02d}_2330z.nc4',
        # 'GCHP.SpeciesConc.{next_day.year:04d}{next_day.month:02d}{next_day.day:02d}_0030z.nc4',
        # 'GCHP.SpeciesConc.{next_day.year:04d}{next_day.month:02d}{next_day.day:02d}_0130z.nc4',
        # 'GCHP.SpeciesConc.{next_day.year:04d}{next_day.month:02d}{next_day.day:02d}_0230z.nc4',
        # 'GCHP.SpeciesConc.{next_day.year:04d}{next_day.month:02d}{next_day.day:02d}_0330z.nc4',
        # 'GCHP.SpeciesConc.{next_day.year:04d}{next_day.month:02d}{next_day.day:02d}_0430z.nc4',
        # 'GCHP.SpeciesConc.{next_day.year:04d}{next_day.month:02d}{next_day.day:02d}_0530z.nc4',
        # 'GCHP.SpeciesConc.{next_day.year:04d}{next_day.month:02d}{next_day.day:02d}_0630z.nc4',
        # 'GCHP.SpeciesConc.{next_day.year:04d}{next_day.month:02d}{next_day.day:02d}_0730z.nc4',
    ]

    index_cache = {}

    aqs_new = aqs.copy()
    aqs_new['Simulated Mean'] = np.nan

    for date, new_df in tqdm(aqs.groupby(level=0), desc='Date'):
        rpaths = [os.path.join(args.datadir, fpath.format(date=date, next_day=date + pd.Timedelta(days=1))) for fpath in diag_fpath]
        rpaths_missing = any([not os.path.exists(rpath) for rpath in rpaths])

        if rpaths_missing:
            continue

        if args.start_date is not None:
            if date < pd.to_datetime(args.start_date):
                continue

        if args.end_date is not None:
            if date > pd.to_datetime(args.end_date):
                continue

        da = xr.open_mfdataset(
            rpaths,
            combine='nested', concat_dim='time',
            data_vars='minimal', coords='minimal',
            compat='override'
        ).isel(lev=0).squeeze()['SpeciesConc_NO2'].mean('time') * 1e9

        sites = new_df.loc[date, :]

        new_sites = set(sites.index) - set(index_cache.keys())

        for new_site in new_sites:
            distances = central_angle(da.lons, da.lats, sites.loc[new_site]['Longitude'], sites.loc[new_site]['Latitude'])
            index = np.unravel_index(distances.argmin(), distances.shape)
            index_cache[new_site] = index

        slices = [index_cache[site] for site in sites.index]
        slices = list(zip(*slices))

        mean_values = [float(da[nf, Ydim, Xdim]) for nf, Ydim, Xdim in zip(*slices)]

        da.close()

        mi = pd.MultiIndex.from_product([[date], sites.index])
        aqs_new = aqs_new._set_value(mi, 'Simulated Mean', mean_values)

    aqs_new = aqs_new.dropna()

    aqs_new.to_csv(args.o)

    try:
        def mean_bias(y_true, y_pred):
            return (y_pred.mean() - y_true.mean()).item()
        mb = mean_bias(aqs_new['Arithmetic Mean'], aqs_new['Simulated Mean'])
        mae = sklearn.metrics.mean_absolute_error(aqs_new['Arithmetic Mean'], aqs_new['Simulated Mean'])
        rmse = np.sqrt(sklearn.metrics.mean_squared_error(aqs_new['Arithmetic Mean'], aqs_new['Simulated Mean']))
        r2 = sklearn.metrics.r2_score(aqs_new['Arithmetic Mean'], aqs_new['Simulated Mean'])

        print("Observation stats")
        print(f"  Min:  {aqs_new['Arithmetic Mean'].min():7.3f} [ppb]")
        print(f"  P25:  {aqs_new['Arithmetic Mean'].quantile(0.25):7.3f} [ppb]")
        print(f"  P50:  {aqs_new['Arithmetic Mean'].quantile(0.25):7.3f} [ppb]")
        print(f"  P75:  {aqs_new['Arithmetic Mean'].quantile(0.25):7.3f} [ppb]")
        print(f"  Max:  {aqs_new['Arithmetic Mean'].max():7.3f} [ppb]")
        print("Metrics")
        print(f"  MB:   {mb:7.3f} [ppb]")
        print(f"  MAE:  {mae:7.3f} [ppb]")
        print(f"  RMSE: {rmse:7.3f} [ppb]")
        print(f"  R2:   {r2:7.3f}")
    except:
        pass