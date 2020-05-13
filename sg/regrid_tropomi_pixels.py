
import argparse
import re

import sg.grids
import sg.compare_grids2

import numpy as np
import pandas as pd
import xarray as xr
import shapely.geometry
import pyproj

from tqdm import tqdm

import multiprocessing.dummy as mp


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
    parser.add_argument('--grid_boxes',
                        type=str,
                        required=True),
    parser.add_argument('--daily_tropomi',
                        type=str,
                        required=True)
    args = parser.parse_args()

    ds_grid_boxes = xr.open_dataset(args.grid_boxes)
    ds_daily_tropomi = xr.open_dataset(args.daily_tropomi)

    date = re.search('201[0-9][0-9]{2}[0-9]{2}', args.daily_tropomi).group(0)
    date = pd.to_datetime(date, format='%Y%m%d')

    gb_centers_x = ds_grid_boxes['grid_boxes_centers'].isel(XY=0).squeeze()
    gb_centers_y = ds_grid_boxes['grid_boxes_centers'].isel(XY=1).squeeze()

    grid_shape = (ds_grid_boxes.dims['nf'], ds_grid_boxes.dims['Ydim'], ds_grid_boxes.dims['Xdim'])

    npixels = ds_daily_tropomi.dims['dim2']

    observations_sum = np.zeros(grid_shape)
    observations_count = np.zeros(grid_shape)

    pbar = tqdm(total=npixels, desc='Observations')
    def exec_loop(pixel_idx):
    # for pixel_idx in tqdm(range(npixels), desc='Observation'):
        ds_pixel = ds_daily_tropomi.isel(dim2=pixel_idx)
        pixel_xc = ds_pixel['LON_CENTER'].squeeze().item()
        pixel_yc = ds_pixel['LAT_CENTER'].squeeze().item()
        pixel_xy = np.array([ds_pixel['LON_CORNERS'].squeeze().values, ds_pixel['LAT_CORNERS'].squeeze().values]).transpose()

        distances = central_angle(pixel_xc, pixel_yc, gb_centers_x, gb_centers_y).values
        closest_indexes = np.argpartition(distances.flatten(), 4)[:4]
        closest_indexes = np.unravel_index(closest_indexes, distances.shape)

        # proj = pyproj.Proj(f'+proj=laea +lat_0={pixel_yc} +lon_0={pixel_xc} +units=m')
        # proj = pyproj.Proj(f'+proj=gnom +lat_0={pixel_yc} +lon_0={pixel_xc}')

        nearby_gb_xy = ds_grid_boxes['grid_boxes'].values[closest_indexes]
        # nearby_gb_xy = np.transpose(proj(nearby_gb_xy[..., 0], nearby_gb_xy[..., 1]))
        nearby_boxes = [shapely.geometry.Polygon(xy) for xy in nearby_gb_xy]

        # pixel_xy = np.transpose(proj(pixel_xy[..., 0], pixel_xy[..., 1]))
        # pixel_box = shapely.geometry.Polygon(pixel_xy)

        pixel_center = shapely.geometry.Point(pixel_xc, pixel_yc)
        nearby_containing_pixel = np.argmax([grid_box.contains(pixel_center) for grid_box in nearby_boxes])

        containing_grid_box = tuple([dim_indexes[nearby_containing_pixel] for dim_indexes in closest_indexes])

        observations_sum[containing_grid_box] += ds_pixel['TROPOMI_NO2_molec_per_m2'].item()
        observations_count[containing_grid_box] += 1

        pbar.update()
    pool = mp.Pool(2)
    pool.map(exec_loop, range(npixels))
    pool.close()
    pool.join()


    old_settings = np.seterr(divide='ignore', invalid='ignore')
    observations = observations_sum / observations_count
    np.seterr(**old_settings)

    ds_out = xr.Dataset(
        data_vars={'TROPOMI_NO2': (('date', 'nf', 'Ydim', 'Xdim'), observations[np.newaxis, ...])},
        coords={
            'date': [date],
            **{coord: ds_grid_boxes.coords[coord] for coord in ['nf', 'Ydim', 'Xdim']}
        }
    )
    ds_out.to_netcdf(f'TROPOMI_NO2_{date.strftime("%Y%m%d")}.nc')