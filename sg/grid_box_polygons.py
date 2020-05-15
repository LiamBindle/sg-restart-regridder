
import argparse

import sg.grids
import sg.compare_grids2

import numpy as np
import xarray as xr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csres',
                        type=int,
                        required=True),
    parser.add_argument('--sg',
                        nargs=3,
                        type=float,
                        default=None)
    args = parser.parse_args()

    if args.sg is None:
        grid = sg.grids.CubeSphere(args.csres)
    else:
        grid = sg.grids.StretchedGrid(args.csres, *args.sg)

    polygons = np.ones((6, grid.csres, grid.csres, 4, 2)) * np.nan
    centers = np.ones((6, grid.csres, grid.csres, 2)) * np.nan

    for nf in range(6):
        polygons[nf, :, :, :, :] = sg.compare_grids2.get_minor_xy(grid.xe(nf), grid.ye(nf))[:, :, :-1, :]
        centers[nf, :, :, 0] = grid.xc(nf)
        centers[nf, :, :, 1] = grid.yc(nf)

    centers[..., 0][centers[..., 0] > 180] -= 360

    xe = np.array([grid.xe(nf) for nf in range(6)])
    xe[xe > 180] -= 360
    ye = np.array([grid.ye(nf) for nf in range(6)])

    ds = xr.Dataset(
        data_vars={
            'xe': (['nf', 'YdimE', 'XdimE'], xe),
            'ye': (['nf', 'YdimE', 'XdimE'], ye),
            'grid_boxes': (['nf', 'Ydim', 'Xdim', 'POLYGON_PTS', 'XY'], polygons),
            'grid_boxes_centers': (['nf', 'Ydim', 'Xdim', 'XY'], centers)
        },
        coords={
            'nf': np.arange(6),
            'Ydim': np.arange(grid.csres),
            'Xdim': np.arange(grid.csres),
            'YdimE': np.arange(grid.csres+1),
            'XdimE': np.arange(grid.csres+1),
            'POLYGON_PTS': np.arange(4),
            'XY': np.arange(2),
        }
    )

    ds['grid_boxes'].attrs = {
        'long_name': 'XY outline of GCHP\'s grid-boxes',
        'notes': 'XY tuple is lonlat where longitudes are in [-180, 180)',
    }
    ds['grid_boxes_centers'].attrs = {
        'long_name': 'XY centers of GCHP\'s grid-boxes',
        'notes': 'XY tuple is lonlat where longitudes are in [-180, 180)',
    }
    ds.attrs = {
        'cubed-sphere-resolution': int(grid.csres)
    }

    if isinstance(grid, sg.grids.StretchedGrid):
        ds.attrs['stretch-factor'] = float(grid.sf)
        ds.attrs['target-latitude'] = float(grid.target_lat)
        ds.attrs['target-longitude'] = float(grid.target_lon)

    ds.to_netcdf('grid_box_outlines_and_centers.nc')