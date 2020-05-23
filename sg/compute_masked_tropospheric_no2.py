import argparse
import re

import numpy as np
import pandas as pd
import xarray as xr

from sg.tropospheric_column import compute_no2_column


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gchp_data',
                        type=str,
                        required=True),
    parser.add_argument('--daily_tropomi',
                        type=str,
                        required=True)
    args = parser.parse_args()

    # ds_gchp = xr.open_dataset(args.daily_gchp)
    ds_tropomi = xr.open_dataset(args.daily_tropomi)

    date = re.search('201[0-9][0-9]{2}[0-9]{2}', args.daily_tropomi).group(0)
    date = pd.to_datetime(date, format='%Y%m%d')

    gchp_species = xr.open_dataset(f"{args.gchp_data}/GCHP.TROPOMI_Species.OVERPASS.{date.strftime('%Y%m%d')}_1330z.nc4")
    gchp_metc = xr.open_dataset(f"{args.gchp_data}/GCHP.TROPOMI_MetC.OVERPASS.{date.strftime('%Y%m%d')}_1330z.nc4")
    gchp_mete = xr.open_dataset(f"{args.gchp_data}/GCHP.TROPOMI_MetE.OVERPASS.{date.strftime('%Y%m%d')}_1330z.nc4")

    ds_gchp = compute_no2_column(gchp_species, gchp_metc, gchp_mete)

    tropomi_no2 = ds_tropomi['TROPOMI_NO2']
    gchp_no2 = ds_gchp['TroposphericColumn_NO2']

    gchp_no2 = gchp_no2.assign_coords({c: tropomi_no2[c] for c in ['nf', 'Ydim', 'Xdim']})

    def mask_where_tropomi_is_nan(gchp, tropomi):
        flat_gchp = gchp.flatten()
        flat_tropomi = tropomi.flatten()
        flat_gchp[np.isnan(flat_tropomi)] = np.nan
        return flat_gchp.reshape(gchp.shape)

    gchp_no2 = xr.apply_ufunc(
        mask_where_tropomi_is_nan,
        gchp_no2, tropomi_no2,
        input_core_dims=[['nf', 'Ydim', 'Xdim'], ['nf', 'Ydim', 'Xdim']],
        output_core_dims=[['nf', 'Ydim', 'Xdim']],
        vectorize=True
    )

    ds_out = xr.Dataset(data_vars={'GCHP_NO2': gchp_no2})
    ds_out.to_netcdf(f'GCHP_NO2_{date.strftime("%Y%m%d")}.nc')