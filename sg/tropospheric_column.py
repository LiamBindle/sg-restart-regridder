import argparse
import numpy as np
import pandas as pd
import xarray as xr
import shapely.geometry

from tqdm import tqdm


def compute_no2_column(ds_species, ds_metc, ds_mete):
    Md = 28.9647e-3 / 6.0221409e+23  # [kg molec-1]
    no2_area_density = ds_metc['Met_AIRDEN'] * ds_metc['Met_BXHEIGHT'] * ds_species['SpeciesConc_NO2'] / Md

    def sum_below_tropopause(column, pfloor, tropp):
        in_troposphere = pfloor > tropp
        return column[in_troposphere].sum(axis=0)

    pfloor = ds_mete['Met_PEDGE'].isel(lev=slice(0, -1)).assign_coords({'lev': no2_area_density['lev']})
    tropospheric_no2 = xr.apply_ufunc(
        sum_below_tropopause,
        no2_area_density,
        pfloor,
        ds_metc['Met_TropP'],
        input_core_dims=[['lev'], ['lev'], []],
        vectorize=True
    )
    tropospheric_no2 = tropospheric_no2 / 100**2  # [molec m-2] -> [molec cm-2]

    ds_out = xr.Dataset({'TroposphericColumn_NO2': tropospheric_no2})
    ds_out['TroposphericColumn_NO2'].attrs['units'] = 'molec cm-2'
    return ds_out



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--species',
                        type=str,
                        required=True),
    parser.add_argument('--metc',
                        type=str,
                        required=True)
    parser.add_argument('--mete',
                        type=str,
                        required=True)
    args = parser.parse_args()

    ds_metc = xr.open_dataset(args.metc)
    ds_mete = xr.open_dataset(args.mete)
    ds_species = xr.open_dataset(args.species)

    ds_out = compute_no2_column(ds_species, ds_metc, ds_mete)

    ofile = f'GCHP.Tropospheric_NO2_Column.{ds_out.time[0].dt.strftime("%Y%m%d").item()}.nc'
    ds_out.to_netcdf(ofile)