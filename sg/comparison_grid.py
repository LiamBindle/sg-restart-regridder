
import argparse

import sg.grids
import sg.compare_grids2

import numpy as np
import xarray as xr


from shapely.geometry import Polygon
from shapely.ops import transform
import pyproj

def draw_minor_grid_boxes_naive(xx, yy):
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    plt.figure()
    ax = plt.axes(projection=ccrs.epsg(2163))
    ax.coastlines()

    for x, y in zip(xx[1:, :], yy[1:, :]):
        ax. plot(x, y, transform=ccrs.PlateCarree(), linewidth=0.2)
    for x, y in zip(xx[:, 1:].transpose(), yy[:, 1:].transpose()):
        ax. plot(x, y, transform=ccrs.PlateCarree(), linewidth=0.2)

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--extent',
                        nargs=4,
                        type=float,
                        default=[-129, -64, 22, 51]),
    parser.add_argument('--extent_proj',
                        type=str,
                        default='epsg:4326'),
    parser.add_argument('--dx',
                        type=float,
                        required=True),
    parser.add_argument('--dy',
                        type=float,
                        required=True),
    parser.add_argument('--proj',
                        type=str,
                        default='epsg:2163'),
    args = parser.parse_args()

    xmin, xmax, ymin, ymax = args.extent

    xbounds = np.array([xmin, xmin, xmax, xmax])  # bottom-left, top-left, top-right, bottom-right
    ybounds = np.array([ymin, ymax, ymax, ymin])

    extent_polygon = Polygon(np.moveaxis([xbounds, ybounds], 0, -1))
    projection = pyproj.Transformer.from_crs(crs_from=args.extent_proj, crs_to=args.proj, always_xy=True).transform
    extent_polygon = transform(projection, extent_polygon)
    xmin, ymin, xmax, ymax = extent_polygon.bounds

    dx = args.dx
    dy = args.dy
    nx = np.ceil((xmax-xmin)/dx).astype(int)
    ny = np.ceil((ymax-ymin)/dy).astype(int)

    xe = np.linspace(xmin, xmin+dx*nx, nx)
    ye = np.linspace(ymin, ymin+dy*ny, ny)

    xc = (xe[:-1] + xe[1:])/2
    yc = (ye[:-1] + ye[1:])/2

    xe, ye = np.meshgrid(xe, ye, indexing='ij')
    xc, yc = np.meshgrid(xc, yc, indexing='ij')

    xy = sg.compare_grids2.get_minor_xy(xe, ye)

    # Convert back to LL
    transformer = pyproj.Transformer.from_crs(crs_from=args.proj, crs_to=args.extent_proj, always_xy=True)
    xc, yc = transformer.transform(xc, yc)
    xy_x, xy_y = transformer.transform(xy[..., 0], xy[..., 1])

    xy = np.moveaxis([xy_x, xy_y], 0, -1)[:, :, :-1, :]
    centers = np.moveaxis([xc, yc], 0, -1)


    ds = xr.Dataset(
        data_vars={
            'xe': (['ie', 'je'], xe),
            'ye': (['ie', 'je'], ye),
            'grid_boxes': (['i', 'j', 'POLYGON_PTS', 'XY'], xy),
            'grid_boxes_centers': (['i', 'j', 'XY'], centers)
        },
        coords={
            'i': np.arange(nx-1),
            'j': np.arange(ny-1),
            'ie': np.arange(nx),
            'je': np.arange(ny),
            'POLYGON_PTS': np.arange(4),
            'XY': np.arange(2),
        }
    )

    ds.to_netcdf('comparison_grid.nc')