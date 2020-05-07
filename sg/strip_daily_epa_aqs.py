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
    parser.add_argument('-a', '--aqs',
                        type=str,
                        default='daily_42602_2016.csv')
    parser.add_argument('-v', '--var',
                        type=str,
                        default='SpeciesConc_NO2',)
    parser.add_argument('--coverage-thresh',
                        type=float,
                        default=100)
    parser.add_argument('--start-date',
                        type=str,
                        default=None)
    parser.add_argument('--end-date',
                        type=str,
                        default=None)
    parser.add_argument('--aqs-year',
                        type=int,
                        default=2016)
    parser.add_argument('-o',
                        type=str,
                        default='daily_aqs_{species}_comparison.csv')
    args = parser.parse_args()

    if os.path.isdir(args.aqs):
        aqs_products = {
            'SpeciesConc_O3':  44201,
            'SpeciesConc_SO2': 42401,
            'SpeciesConc_CO':  42101,
            'SpeciesConc_NO2': 42602,
            'PM25': 88101,
        }
        aqs_fpath = os.path.join(args.aqs, f'daily_{aqs_products[args.var]}_{args.aqs_year}.csv')
    else:
        aqs_fpath = args.aqs

    # Load AQS table
    aqs = pd.read_csv(aqs_fpath)
    # Convert date rows to type datetime
    aqs['Date Local'] = pd.to_datetime(aqs['Date Local'])
    aqs['Date of Last Change'] = pd.to_datetime(aqs['Date of Last Change'])

    # Retain relevant rows
    keep_states = [
        'Georgia', 'Alabama', 'Mississippi', 'Tennessee', 'North Carolina', 'South Carolina', 'Kentucky',
        'Virginia', 'Florida', 'Maryland', 'Delaware'
    ]
    aqs = aqs.loc[aqs['State Name']isin(keep_states)]
    #aqs = aqs.loc[aqs['State Name'] == 'California']                # Only California
    aqs = aqs.loc[aqs['Observation Percent'] >= args.coverage_thresh]   # Only full samples
    aqs = aqs.loc[aqs['Event Type'] == 'None']  # Only samples with no event

    # Reindex according to dates (for comparison w/ simulation) and the station identifier
    aqs = aqs.set_index(['Date Local', 'State Code', 'County Code', 'Site Num', 'Pollutant Standard', 'POC', 'Parameter Code']).sort_index()

    # Declare diagnostic file path templates for 24-hr average
    diag_fpath = [
        'GCHP.{collection}.{date.year:04d}{date.month:02d}{date.day:02d}_0830z.nc4',
        'GCHP.{collection}.{date.year:04d}{date.month:02d}{date.day:02d}_0930z.nc4',
        'GCHP.{collection}.{date.year:04d}{date.month:02d}{date.day:02d}_1030z.nc4',
        'GCHP.{collection}.{date.year:04d}{date.month:02d}{date.day:02d}_1130z.nc4',
        'GCHP.{collection}.{date.year:04d}{date.month:02d}{date.day:02d}_1230z.nc4',
        'GCHP.{collection}.{date.year:04d}{date.month:02d}{date.day:02d}_1330z.nc4',
        'GCHP.{collection}.{date.year:04d}{date.month:02d}{date.day:02d}_1430z.nc4',
        'GCHP.{collection}.{date.year:04d}{date.month:02d}{date.day:02d}_1530z.nc4',
        'GCHP.{collection}.{date.year:04d}{date.month:02d}{date.day:02d}_1630z.nc4',
        'GCHP.{collection}.{date.year:04d}{date.month:02d}{date.day:02d}_1730z.nc4',
        'GCHP.{collection}.{date.year:04d}{date.month:02d}{date.day:02d}_1830z.nc4',
        'GCHP.{collection}.{date.year:04d}{date.month:02d}{date.day:02d}_1930z.nc4',
        'GCHP.{collection}.{date.year:04d}{date.month:02d}{date.day:02d}_2030z.nc4',
        'GCHP.{collection}.{date.year:04d}{date.month:02d}{date.day:02d}_2130z.nc4',
        'GCHP.{collection}.{date.year:04d}{date.month:02d}{date.day:02d}_2230z.nc4',
        'GCHP.{collection}.{date.year:04d}{date.month:02d}{date.day:02d}_2330z.nc4',
        'GCHP.{collection}.{next_day.year:04d}{next_day.month:02d}{next_day.day:02d}_0030z.nc4',
        'GCHP.{collection}.{next_day.year:04d}{next_day.month:02d}{next_day.day:02d}_0130z.nc4',
        'GCHP.{collection}.{next_day.year:04d}{next_day.month:02d}{next_day.day:02d}_0230z.nc4',
        'GCHP.{collection}.{next_day.year:04d}{next_day.month:02d}{next_day.day:02d}_0330z.nc4',
        'GCHP.{collection}.{next_day.year:04d}{next_day.month:02d}{next_day.day:02d}_0430z.nc4',
        'GCHP.{collection}.{next_day.year:04d}{next_day.month:02d}{next_day.day:02d}_0530z.nc4',
        'GCHP.{collection}.{next_day.year:04d}{next_day.month:02d}{next_day.day:02d}_0630z.nc4',
        'GCHP.{collection}.{next_day.year:04d}{next_day.month:02d}{next_day.day:02d}_0730z.nc4',
    ]
    if args.var in ['SpeciesConc_NO2', 'SpeciesConc_O3', 'SpeciesConc_SO2', 'SpeciesConc_CO']:
        collection='SpeciesConc'
    elif args.var == 'PM25':
        collection = 'AerosolMass'
    else:
        raise ValueError(f"Unknown species: {args.var}")


    # Define scale factors that are applied to simulated concentrations
    scale_factors = {
        'SpeciesConc_NO2':  1e9,
        'SpeciesConc_R4N2': 1e9,
        'SpeciesConc_PAN':  1e9,
        'SpeciesConc_HNO3': 1e9,
        'SpeciesConc_O3': 1e6,
        'SpeciesConc_CO': 1e6,
        'SpeciesConc_SO2': 1e9,
        'PM25': 1.0,
    }

    # Cache of station identifier -> simulation grid index
    index_cache = {}

    # Make the new table
    aqs_new = aqs.copy()
    aqs_new['Simulated Mean'] = np.nan
    if args.var in ['SpeciesConc_NO2']:
        aqs_new['Corrected Arithmetic Mean'] = np.nan
        calculate_corrected_NO2 = True
    else:
        calculate_corrected_NO2 = False

    # Dataset variables that are kept after being loaded
    if args.var in ['SpeciesConc_NO2']:
        keep_vars = ['SpeciesConc_NO2', 'SpeciesConc_R4N2', 'SpeciesConc_PAN', 'SpeciesConc_HNO3']
    else:
        keep_vars = [args.var]

    # Loop through dates
    for date, new_df in tqdm(aqs.groupby(level=0), desc='Date'):
        rpaths = [os.path.join(args.datadir, fpath.format(date=date, next_day=date + pd.Timedelta(days=1), collection=collection)) for fpath in diag_fpath]
        rpaths_missing = any([not os.path.exists(rpath) for rpath in rpaths])

        if rpaths_missing:
            continue

        if args.start_date is not None:
            if date < pd.to_datetime(args.start_date):
                continue

        if args.end_date is not None:
            if date > pd.to_datetime(args.end_date):
                continue

        ds = xr.open_mfdataset(
            rpaths,
            combine='nested', concat_dim='time',
            data_vars='minimal', coords='minimal',
            compat='override'
        ).isel(lev=slice(0,3)).squeeze()
        drop_vars = [v for v in ds.data_vars if v not in keep_vars]
        ds = ds.drop(drop_vars)
        ds = ds.mean(['lev', 'time'])

        sites = new_df.loc[date]

        site_indexes = list(zip(*[sites.index.get_level_values(i) for i in range(3)]))
        new_sites = set(site_indexes) - set(index_cache.keys())

        for new_site in new_sites:
            site_lon = sites.loc[new_site]['Longitude'][0].item()
            site_lat = sites.loc[new_site]['Latitude'][0].item()
            distances = central_angle(ds.lons, ds.lats, site_lon, site_lat)
            index = np.unravel_index(distances.argmin(), distances.shape)
            index_cache[new_site] = index

        slices = [index_cache[site] for site in site_indexes]
        slices = list(zip(*slices))

        simulated_means = {
            v: np.array([float(ds[v][nf, Ydim, Xdim]) for nf, Ydim, Xdim in zip(*slices)])*scale_factors[v] for v in keep_vars
        }

        aqs_new.at[new_df.index, 'Simulated Mean'] = simulated_means[args.var]

        if calculate_corrected_NO2:
            corrected = aqs_new.loc[new_df.index]['Arithmetic Mean'] / (simulated_means['SpeciesConc_NO2'] + simulated_means['SpeciesConc_R4N2'] + 0.95 * simulated_means['SpeciesConc_PAN'] + 0.15 * simulated_means['SpeciesConc_HNO3'])
            aqs_new.at[new_df.index, 'Corrected Arithmetic Mean'] = corrected

        ds.close()

    aqs_new = aqs_new.dropna(subset=['Simulated Mean'])

    aqs_new.to_csv(args.o.format(species=args.var))

    try:
        def mean_bias(y_true, y_pred):
            return (y_pred.mean() - y_true.mean()).item()
        mb = mean_bias(aqs_new['Arithmetic Mean'], aqs_new['Simulated Mean'])
        mae = sklearn.metrics.mean_absolute_error(aqs_new['Arithmetic Mean'], aqs_new['Simulated Mean'])
        rmse = np.sqrt(sklearn.metrics.mean_squared_error(aqs_new['Arithmetic Mean'], aqs_new['Simulated Mean']))
        r2 = sklearn.metrics.r2_score(aqs_new['Arithmetic Mean'], aqs_new['Simulated Mean'])

        print("Observation stats")
        print(f"  N:    {aqs_new['Simulated Mean'].count():7d}")
        print(f"  Min:  {aqs_new['Arithmetic Mean'].min():7.3f}")
        print(f"  P25:  {aqs_new['Arithmetic Mean'].quantile(0.25):7.3f}")
        print(f"  P50:  {aqs_new['Arithmetic Mean'].quantile(0.25):7.3f}")
        print(f"  P75:  {aqs_new['Arithmetic Mean'].quantile(0.25):7.3f}")
        print(f"  Max:  {aqs_new['Arithmetic Mean'].max():7.3f}")
        print("Metrics")
        print(f"  MB:   {mb:7.3f}")
        print(f"  MAE:  {mae:7.3f}")
        print(f"  RMSE: {rmse:7.3f}")
        print(f"  R2:   {r2:7.3f}")
        if 'Corrected Arithmetic Mean' in aqs_new.columns:
            mb = mean_bias(aqs_new['Corrected Arithmetic Mean'], aqs_new['Simulated Mean'])
            mae = sklearn.metrics.mean_absolute_error(aqs_new['Corrected Arithmetic Mean'], aqs_new['Simulated Mean'])
            rmse = np.sqrt(sklearn.metrics.mean_squared_error(aqs_new['Corrected Arithmetic Mean'], aqs_new['Simulated Mean']))
            r2 = sklearn.metrics.r2_score(aqs_new['Corrected Arithmetic Mean'], aqs_new['Simulated Mean'])
            print("Metrics (corrected observations)")
            print(f"  MB:   {mb:7.3f}")
            print(f"  MAE:  {mae:7.3f}")
            print(f"  RMSE: {rmse:7.3f}")
            print(f"  R2:   {r2:7.3f}")
    except:
        pass