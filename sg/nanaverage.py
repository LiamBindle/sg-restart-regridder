import argparse
import xarray as xr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_data',
                        nargs='+',
                        type=str,),
    parser.add_argument('--dim',
                        type=str,
                        required=True)
    parser.add_argument('--weekly',
                        action='store_true')
    parser.add_argument('--missing_time',
                        action='store_true')
    parser.add_argument('-o',
                        type=str,
                        required=True)
    args = parser.parse_args()

    if args.missing_time:
        extra_mfdataset_kwargs = dict()
    else:
        extra_mfdataset_kwargs = dict(
            data_vars='minimal',
            coords='minimal',
            join='override',
        )

    ds = xr.open_mfdataset(
        args.input_data,
        combine='nested',
        concat_dim=args.dim,
        **extra_mfdataset_kwargs
    )

    if args.weekly:
        week1 = ds.isel(**{args.dim: slice(0, 7)}).mean(dim=args.dim)
        week2 = ds.isel(**{args.dim: slice(7, 14)}).mean(dim=args.dim)
        week3 = ds.isel(**{args.dim: slice(14, 21)}).mean(dim=args.dim)
        week4 = ds.isel(**{args.dim: slice(21, 28)}).mean(dim=args.dim)

        ds = xr.concat([week1, week2, week3, week4], dim=args.dim)
    else:
        ds = ds.mean(dim=args.dim)
    ds.to_netcdf(args.o)
