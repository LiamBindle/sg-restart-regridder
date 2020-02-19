import sg.compare_grids
import sklearn.linear_model
import sklearn.metrics
import scipy.stats


def get_linreg_metrics(var, ctl_dataset, exp_dataset, ctl_indexes, exp_indexes, lev=0):
    vertical_reduction = lambda x: x.squeeze().isel(lev=lev).transpose('nf', 'Ydim', 'Xdim')
    x = ctl_dataset[var].pipe(vertical_reduction).values[ctl_indexes]
    y = exp_dataset[var].pipe(vertical_reduction).values[exp_indexes]

    # Get metrics
    r2 = sklearn.metrics.r2_score(y_true=x, y_pred=y)
    rmse = np.sqrt(sklearn.metrics.mean_squared_error(y_true=x, y_pred=y))
    mae = sklearn.metrics.mean_absolute_error(y_true=x, y_pred=y)
    return r2, rmse, mae

def format_linreg(r2, rmse, mae, scale=1e9):
    return np.array([f'{r2: 4.2f}', f'{rmse*scale: 5.1f}', f'{mae*scale: 5.1f}'])

def scatter_plot(ax, var, ctl_grid, exp_grid, ctl_dataset, exp_dataset, tol_dist=50e3, tol_intersect=0.4, lev=0):
    vertical_reduction = lambda x: x.squeeze().isel(lev=lev).transpose('nf', 'Ydim', 'Xdim')
    x = ctl_dataset[var].pipe(vertical_reduction).values[ctl_indexes]
    y = exp_dataset[var].pipe(vertical_reduction).values[exp_indexes]
    plt.scatter(x, y, )


if __name__ == '__main__':
    import sg.grids

    import xarray as xr
    import numpy as np
    import pyproj
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(color_codes=True)

    LEVEL=0
    COMP_VARIABLES=['SpeciesConc_O3']
    COMP_VAR_SCALE=1e9
    TOLERANCE_DISTANCE = 50e3
    TOLERANCE_INTERSECT=0.4

    CTL_RES = 90
    CTL_DATA_PATH = '/extra-space/sgv-scatter/GCHP.C90.SpeciesConc.20160716_1200z.nc4'
    ctl_data = xr.open_dataset(CTL_DATA_PATH)
    ctl_grid = sg.grids.CubeSphere(CTL_RES)

    CS_RES=48,
    STRETCH_FACTOR = 2.015625,
    TARGET_LAT = 33.5,
    TARGET_LON = 275.5,
    PATH='/extra-space/sgv-scatter/GCHP.S48.SpeciesConc.20160716_1200z.nc4'

    for var in COMP_VARIABLES:



    plt.figure()
    table_text = []
    table_cols = []
    all_values = []


    for sim in sims:
        exp_data = xr.open_dataset(sim['PATH'])
        exp_grid = sg.grids.StretchedGrid(sim['CS_RES'], sim['STRETCH_FACTOR'], sim['TARGET_LAT'], sim['TARGET_LON'])
        ctl_indexes, exp_indexes = sg.compare_grids.comparable_gridboxes(ctl_grid, exp_grid,
                                                                         TOLERANCE_DISTANCE,
                                                                         None,
                                                                         TOLERANCE_INTERSECT,
                                                                         exp_grid.target_lat, exp_grid.target_lon)

        # Keep only the target face?
        only_target_face = True
        if only_target_face:
            target_face = exp_indexes[0] == 5
            ctl_indexes = tuple(i[target_face] for i in ctl_indexes)
            exp_indexes = tuple(i[target_face] for i in exp_indexes)

        # Get x and y
        vertical_reduction = lambda x: x.squeeze().isel(lev=LEVEL).transpose('nf', 'Ydim', 'Xdim')
        x = ctl_data[COMP_VAR].pipe(vertical_reduction).values[ctl_indexes]
        y = exp_data[COMP_VAR].pipe(vertical_reduction).values[exp_indexes]

        # Get metrics
        r2 = sklearn.metrics.r2_score(y_true=x, y_pred=y)
        rmse = np.sqrt(sklearn.metrics.mean_squared_error(y_true=x, y_pred=y))
        mae = sklearn.metrics.mean_absolute_error(y_true=x, y_pred=y)

        # Format table text
        table_text.append([f'{r2:4.2f}', f'{rmse*COMP_VAR_SCALE:5.1f}', f'{mae*COMP_VAR_SCALE:5.1f}'])
        table_text.append([f'{r2:4.2f}', f'{rmse * COMP_VAR_SCALE:5.1f}', f'{mae * COMP_VAR_SCALE:5.1f}'])
        table_cols.append(f'S{sim["CS_RES"]}')
        table_cols.append(f'S{sim["CS_RES"]}')

        # Scatter plot
        plt.scatter(x, y)
        all_values.extend(x)
        all_values.extend(y)

    plt.table(cellText=np.array(table_text).transpose(), rowLabels=['R^2', 'RMSE', 'MAE'], colLabels=table_cols, bbox=[0.15, 0.7, 0.5, 0.2], cellLoc='center', zorder=10)

    lim_pad = 0.95
    xmin = np.min(all_values) * lim_pad
    xmax = np.max(all_values) / lim_pad
    plt.xlim((xmin, xmax))
    plt.ylim((xmin, xmax))
    plt.tight_layout()
    plt.show()


    # Parameters
    # CS_RES=48
    # STRETCH_FACTOR = 2.015625
    # TARGET_LAT = 33.5
    # TARGET_LON = 275.5
    #
    # CS_RES = 12
    # STRETCH_FACTOR = 8.0625
    # TARGET_LAT = 33.7
    # TARGET_LON = 275.5
    exit(1)


    TOLERANCE_DISTANCE = 50e3
    TOLERANCE_AREA=None
    TOLERANCE_INTERSECT=0.3


    EXP_DATA_PATH = f'/extra-space/sgv-scatter/GCHP.S{CS_RES}.SpeciesConc.20160716_1200z.nc4'
    CTL_DATA_PATH = '/extra-space/sgv-scatter/GCHP.C90.SpeciesConc.20160716_1200z.nc4'
    CTL_RES = 90

    ctl_data = xr.open_dataset(CTL_DATA_PATH)
    exp_data = xr.open_dataset(EXP_DATA_PATH)

    # Create the grids
    ctl_grid = sg.grids.CubeSphere(CTL_RES)
    exp_grid = sg.grids.StretchedGrid(CS_RES, STRETCH_FACTOR, TARGET_LAT, TARGET_LON)

    # Get comparable indexes
    ctl_indexes, exp_indexes = sg.compare_grids.comparable_gridboxes(ctl_grid, exp_grid,
                                                                     TOLERANCE_DISTANCE,
                                                                     TOLERANCE_AREA,
                                                                     TOLERANCE_INTERSECT,
                                                                     TARGET_LAT, TARGET_LON)

    plt.figure()
    plt.axis('off')
    r2, rmse, mae = get_linreg_metrics('SpeciesConc_O3', ctl_data, exp_data, ctl_indexes, exp_indexes)
    text = format_linreg(r2, rmse, mae)
    plt.table(cellText=np.array([text]).transpose(),rowLabels=['R2', 'RMSE', 'MAE'], colLabels=['S48'], loc='center')
    plt.show()


    plt.close()

    exit(1)
    area_earth = 510.1e6
    length_gridbox = np.sqrt(area_earth / (6*CTL_RES**2))

    print('\nTolerances')
    print(f'\tDistance:  {TOLERANCE_DISTANCE/1000: 5.1f} (box-length: {length_gridbox: 5.1f}) [km]')
    print(f'\tArea:      {TOLERANCE_AREA if TOLERANCE_AREA else np.nan:4.1f} [1]')
    print(f'\tIntersect: {TOLERANCE_INTERSECT if TOLERANCE_INTERSECT else np.nan:4.1f} [1]')
    num_comparable = len(ctl_indexes[0])
    num_comparable_target_face = np.count_nonzero(exp_indexes[0] == 5)
    print(f'\nComparable points')
    print(f'\tTotal:       {num_comparable:4d} ({100*num_comparable/(6*CS_RES**2):4.1f}%)')
    print(f'\tTarget face: {num_comparable_target_face:4d} ({100*num_comparable_target_face/CS_RES**2:4.1f}%)')

    only_target_face = True
    if only_target_face:
        target_face = exp_indexes[0] == 5
        ctl_indexes = tuple(i[target_face] for i in ctl_indexes)
        exp_indexes = tuple(i[target_face] for i in exp_indexes)

    vertical_reduction = lambda x: x.squeeze().isel(lev=0).transpose('nf', 'Ydim', 'Xdim')
    # Do linear regression
    x = ctl_data['SpeciesConc_O3'].pipe(vertical_reduction).values[ctl_indexes]
    y = exp_data['SpeciesConc_O3'].pipe(vertical_reduction).values[exp_indexes]

    r2 = sklearn.metrics.r2_score(y_true=x, y_pred=y)
    rmse = np.sqrt(sklearn.metrics.mean_squared_error(y_true=x, y_pred=y))
    mae = sklearn.metrics.mean_absolute_error(y_true=x, y_pred=y)
    print('\nMetrics')
    print(f'\tr2:   {r2:5.3f}')
    print(f'\trmse: {rmse:5.1e}')
    print(f'\tmae: {mae:5.1e}')

    lim_pad = 0.95
    sns.regplot(x, y)
    plt.gca().set_aspect('equal', 'box')
    xmin = min(np.min(x) * lim_pad, np.min(y) * lim_pad)
    xmax = max(np.max(x) / lim_pad, np.max(y) / lim_pad)
    plt.xlim((xmin, xmax))
    plt.ylim((xmin, xmax))
    plt.show()
