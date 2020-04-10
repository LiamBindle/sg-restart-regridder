import argparse

import numpy as np
import scipy.sparse
import xarray as xr
from dask.diagnostics import ProgressBar

def ufunc_multiply(x, M):
    y = M @ x
    return y

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generates a sparse intersect matrix')
    parser.add_argument('-i',
                        metavar='FILEIN',
                        type=str,
                        required=True,
                        help='path to input file')
    parser.add_argument('-m',
                        metavar='MATRIX_FILE',
                        type=str,
                        required=True,
                        help='path to intersect matrix file')
    parser.add_argument('--drop',
                        metavar='V',
                        type=str,
                        nargs='+',
                        required=False,
                        default=[],
                        help='path to intersect matrix file')
    parser.add_argument('-o',
                        metavar='FILEOUT',
                        type=str,
                        required=True,
                        help='path to output file')
    args = vars(parser.parse_args())

    ds = xr.open_dataset(args['i'])
    M = scipy.sparse.load_npz(args['m'])
    out_res = np.sqrt(M.shape[0]//6).astype(int)

    ds = ds.drop(args['drop'])

    ds.coords.update({
        'nf_out': range(6),
        'Xdim_out': range(out_res),
        'Ydim_out': range(out_res)
    })

    ds = ds.stack(iboxes=['nf', 'Ydim', 'Xdim'])
    ds = ds.stack(oboxes=['nf_out', 'Ydim_out', 'Xdim_out'])

    ds2 = xr.apply_ufunc(
        ufunc_multiply,
        ds, M,
        input_core_dims=[['iboxes'], []],
        output_core_dims=[['oboxes']],
        vectorize=True
    )
    ds2 = ds2.unstack('oboxes')

    ds2 = ds2.rename({'nf_out': 'nf', 'Ydim_out': 'Ydim', 'Xdim_out': 'Xdim'})

    encoding = {k: {'dtype': np.float32, 'complevel': 9, 'zlib': True} for k in ds2.data_vars}
    delayed_obj = ds2.to_netcdf(args['o'], encoding=encoding, compute=False)
    with ProgressBar():
        delayed_obj.compute()

    # ds2 = xr.open_dataset('/extra-space/temp/CTL/NOx-CTL.nc')
    # import sg.pcolormesh2
    # import matplotlib.pyplot as plt
    # from sg.compare_grids2 import determine_blocksize, get_minor_xy
    # from sg.grids import CubeSphere
    # import cartopy.crs as ccrs
    #
    # print('Making grid')
    # grid = CubeSphere(180)
    #
    # ax = plt.axes(projection=ccrs.EqualEarth())
    # ax.coastlines()
    # ax.set_global()
    #
    # da = ds2.NOx.isel(lev=0).squeeze()
    # norm = plt.Normalize(0, da.quantile(0.95))
    # for i in range(6):
    #     print(f'Plotting face {i}')
    #     xy = get_minor_xy(grid.xe(i) % 360, grid.ye(i))
    #     blocksize = determine_blocksize(xy, grid.xc(i) % 360, grid.yc(i))
    #
    #     #sg.pcolormesh2.pcolormesh2(grid.xe(i), grid.ye(i), da.isel(nf_out=i).transpose('Ydim_out', 'Xdim_out').data, blocksize, norm)
    #     sg.pcolormesh2.pcolormesh2(grid.xe(i), grid.ye(i), da.isel(nf=i).data,
    #                                blocksize, norm)
    # # sg.pcolormesh2.draw_major_grid_boxes_naive(plt.gca(), grid.xe(0), grid.ye(0))
    # # sg.pcolormesh2.draw_major_grid_boxes_naive(plt.gca(), grid.xe(1), grid.ye(1))
    # # sg.pcolormesh2.draw_major_grid_boxes_naive(plt.gca(), grid.xe(3), grid.ye(3))
    # # sg.pcolormesh2.draw_major_grid_boxes_naive(plt.gca(), grid.xe(4), grid.ye(4))
    # plt.tight_layout()
    # print('Saving figure')
    # plt.savefig('foo-ctl.png', dpi=300)