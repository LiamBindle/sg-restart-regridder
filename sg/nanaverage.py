import argparse
import xarray as xr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_data',
                        nargs='+',
                        type=str,
                        required=True),
    parser.add_argument('--dim',
                        type=str,
                        required=True)
    parser.add_argument('-o',
                        type=str,
                        required=True)
    args = parser.parse_args()

    ds = xr.open_mfdataset(
        args.input_data,
        combine='nested',
        concat_dim=args.dim,
        data_vars='minimal',
        coords='minimal',
        join='override'
    )

    ds = ds.mean(dims=args.dim)
    ds.to_netcdf(args.o)
