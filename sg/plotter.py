import yaml

import matplotlib.pyplot as plt
import matplotlib.gridspec
import numpy as np
import xarray as xr
import sklearn.metrics
import scipy.stats
import cartopy.feature
import cartopy.crs as ccrs
import shapely.geometry

import sg.compare_grids
import sg.grids


dataset_cache = {}
def dataset_factory(path, **kwargs):
    # See it it's cached
    arg_hash = hash(str(dict(path=path, **kwargs)))
    if arg_hash in dataset_cache:
        return dataset_cache[arg_hash]
    else:
        ds = xr.open_dataset(path)
        dataset_cache[arg_hash] = ds
        return ds

grid_cache = {}
def grid_factory(type, **kwargs):
    # See if it's cached
    arg_hash = hash(str(dict(type=type, **kwargs)))
    if arg_hash in grid_cache:
        return grid_cache[arg_hash]

    if type == 'cubed-sphere':
        grid = sg.grids.CubeSphere(**kwargs)
    elif type == 'stretched-grid':
        grid = sg.grids.StretchedGrid(**kwargs)

    grid_cache[arg_hash] = grid
    return grid

dim_red_cache = {}
def dimensionality_reduction_factory(type, **kwargs):
    # See if it's cached
    arg_hash = hash(str(dict(type=type, **kwargs)))
    if arg_hash in dim_red_cache:
        return dim_red_cache[arg_hash]

    if type == 'colocated-grid-boxes':
        rv = sg.compare_grids.comparable_gridboxes(**kwargs)
        dim_red_cache[arg_hash] = rv
        return rv


def format_axes_correlation_plot(ax: plt.Axes):
    ax.set_aspect('equal', 'box')
    ax.grid(True, which='major')

def outline_colocated_boxes(ax, xx, yy, indexes, **kwargs):
    p0 = slice(0, -1)
    p1 = slice(1, None)
    boxes_x = np.moveaxis(np.array([xx[p0, p0], xx[p1, p0], xx[p1, p1], xx[p0, p1], xx[p0, p0]]), 0, -1)
    boxes_y = np.moveaxis(np.array([yy[p0, p0], yy[p1, p0], yy[p1, p1], yy[p0, p1], yy[p0, p0]]), 0, -1)
    for x_idx, y_idx in zip(indexes[0], indexes[1]):
        ax.plot(boxes_x[x_idx, y_idx], boxes_y[x_idx, y_idx], **kwargs)

def colocated_scatter_plot(ax: plt.Axes, var: str, x_dataset: dict, y_dataset: dict, distance_tolerance: float, intersect_tolerance: float, only_target_face=True, **kwargs):
    kwargs = kwargs.copy()
    kwargs.setdefault('scale', 1)
    kwargs.setdefault('units', "1")
    kwargs.setdefault('xlabel', f"Reference, [{kwargs['units']}]")
    kwargs.setdefault('ylabel', f"Experiment, [{kwargs['units']}]")
    kwargs.setdefault('level', 0)

    # Get indexes
    x_grid = grid_factory(**x_dataset['grid'])
    y_grid = grid_factory(**y_dataset['grid'])
    x_indexes, y_indexes = dimensionality_reduction_factory(
        'colocated-grid-boxes',
        control_grid=x_grid,
        exp_grid=y_grid,
        dist_tol_abs=distance_tolerance,
        intersect_tol_rel=intersect_tolerance,
        target_lat=y_grid.target_lat,
        target_lon=y_grid.target_lon
    )

    if only_target_face:
        target_face = y_indexes[0] == 5
        x_indexes = tuple(i[target_face] for i in x_indexes)
        y_indexes = tuple(i[target_face] for i in y_indexes)

    # Get variables
    vertical_reduction = lambda x: x.squeeze().isel(lev=kwargs['level']).transpose('nf', 'Ydim', 'Xdim')
    x = dataset_factory(**x_dataset)[var].pipe(vertical_reduction).values[x_indexes]
    y = dataset_factory(**y_dataset)[var].pipe(vertical_reduction).values[y_indexes]

    # Look for outliers
    if 'outlier_threshold' in kwargs:
        outlier_threshold = kwargs['outlier_threshold']
        rel_diff = np.abs(y-x)/x
        outliers = np.where(rel_diff >= outlier_threshold)
        good_points = np.where(rel_diff < outlier_threshold)

        ax.scatter(x[good_points] * kwargs['scale'], y[good_points] * kwargs['scale'])
        ax.scatter(x[outliers] * kwargs['scale'], y[outliers] * kwargs['scale'], color='red')
    else:
        # Plot
        ax.scatter(x * kwargs['scale'], y * kwargs['scale'])

    # Format axes
    format_axes_correlation_plot(ax)
    ax.set_title(var)
    ax.set_xlabel(kwargs['xlabel'])
    ax.set_ylabel(kwargs['ylabel'])

    lim_pad=0.95
    xmin = np.min([*x, *y]) * kwargs['scale'] * lim_pad
    xmax = np.max([*x, *y]) * kwargs['scale'] / lim_pad
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)

    # Compute metrics
    r2 = sklearn.metrics.r2_score(y_true=x, y_pred=y)
    rmse = np.sqrt(sklearn.metrics.mean_squared_error(y_true=x, y_pred=y))
    mae = sklearn.metrics.mean_absolute_error(y_true=x, y_pred=y)
    ax.text(
        0.1, 0.8,
        f'$\\mathrm{{R}}^2$={r2:5.3f}\nRMSE={rmse * kwargs["scale"]:5.1f}\nMAE={mae * kwargs["scale"]:5.1f}',
        transform=ax.transAxes
    )

def plot_pcolomesh(ax, xx, yy, data, **kwargs):
    cmap = plt.cm.get_cmap(kwargs['cmap'])
    norm = plt.Normalize(kwargs['vmin'], kwargs['vmax'])

    # xx must be [-180 to 180]
    xx[xx > 180] -= 360
    p0 = slice(0, -1)
    p1 = slice(1, None)

    # Mask where bounding box crosses the prime meridian or antimeridian
    cross_pm_or_am_line1 = np.not_equal(np.sign(xx[p0, p0]), np.sign(xx[p1, p0]))
    cross_pm_or_am_line2 = np.not_equal(np.sign(xx[p1, p0]), np.sign(xx[p1, p1]))
    cross_pm_or_am_line3 = np.not_equal(np.sign(xx[p1, p1]), np.sign(xx[p0, p1]))
    cross_pm_or_am_line4 = np.not_equal(np.sign(xx[p0, p1]), np.sign(xx[p0, p0]))
    cross_pm_or_am = cross_pm_or_am_line1 | cross_pm_or_am_line2 | cross_pm_or_am_line3 | cross_pm_or_am_line4

    boxes_x = np.moveaxis(np.array([xx[p0, p0], xx[p1, p0], xx[p1, p1], xx[p0, p1]]), 0, -1)
    boxes_y = np.moveaxis(np.array([yy[p0, p0], yy[p1, p0], yy[p1, p1], yy[p0, p1]]), 0, -1)
    boxes = np.moveaxis(np.array([boxes_x, boxes_y]), 0, -1)

    pm = np.ones_like(data, dtype=bool)
    am = np.ones_like(data, dtype=bool)
    neither = np.copy(cross_pm_or_am)

    crossing_indexes = np.argwhere(cross_pm_or_am)
    for idx in crossing_indexes:
        box = shapely.geometry.LinearRing(boxes[tuple(idx)])
        far_from_the_prime_meridian = shapely.geometry.LineString([(-160, -90), (-160, 90)])
        if box.crosses(far_from_the_prime_meridian):
            am[tuple(idx)] = False
        else:
            pm[tuple(idx)] = False


    ## Data that crosses the antimeridian
    # data_am = np.ma.masked_where(am, data)
    # xx_am = np.copy(xx)
    # xx_am[xx_am < 0] += 360
    # ax.pcolormesh(xx_am, yy, data_am, transform=ccrs.PlateCarree(), **kwargs)

    # Data that doesn't cross a meridian
    data_not_crossing = np.ma.masked_where(neither, data)
    ax.pcolormesh(xx, yy, data_not_crossing, transform=ccrs.PlateCarree(), **kwargs)

    # Data that crosses the prime meridian
    data_pm = np.ma.masked_where(pm, data)
    return ax.pcolormesh(xx, yy, data_pm, transform=ccrs.PlateCarree(), **kwargs)



def map_axes_formatter_1(ax, **kwargs):
    kwargs.setdefault('extent', (-180, 180, -90, 90))
    kwargs.setdefault('coastlines', True)
    kwargs.setdefault('coastlines_kwargs', {'linewidth': 0.8})
    kwargs.setdefault('borders', True)
    kwargs.setdefault('borders_kwargs', {'linewidth': 0.6})
    kwargs.setdefault('states', True)
    kwargs.setdefault('states_kwargs', {'linewidth': 0.4})

    ax.set_extent(kwargs['extent'])

    if kwargs['coastlines']:
        ax.add_feature(cartopy.feature.COASTLINE, **kwargs['coastlines_kwargs'])
    if kwargs['borders']:
        ax.add_feature(cartopy.feature.BORDERS, **kwargs['borders_kwargs'])
    if kwargs['states']:
        states_provinces = cartopy.feature.NaturalEarthFeature(
            category='cultural',
            name='admin_1_states_provinces_lines',
            scale='50m',
            facecolor='none')
        ax.add_feature(states_provinces, **kwargs['states_kwargs'])


def pcolormesh_comparison(var, dataset1, dataset2, map_conf={}, pcolormesh_kwargs={}, **kwargs):
    kwargs = kwargs.copy()
    map_conf = map_conf.copy()
    pcolormesh_kwargs = pcolormesh_kwargs.copy()
    kwargs.setdefault('faces', [0, 1, 2, 3, 4, 5])
    kwargs.setdefault('level', 0)
    kwargs.setdefault('scale', 1e9)
    kwargs.setdefault('units', 'ppb')
    kwargs.setdefault('outlier_threshold', 0.1)

    extremes = np.array([ds.quantile(0.95).item() for ds in [dataset_factory(**dataset1)[var].isel(lev=kwargs['level']), dataset_factory(**dataset2)[var].isel(lev=kwargs['level'])]])
    pcolormesh_kwargs.setdefault('vmin', 0)
    pcolormesh_kwargs.setdefault('vmax', np.max(extremes) * kwargs['scale'])
    pcolormesh_kwargs.setdefault('cmap', 'cividis')

    map_conf.setdefault('projection', 'ccrs.PlateCarree()')

    projection = eval(map_conf.pop('projection'))

    if kwargs.get('target_face_extent', False):
        exp_grid = grid_factory(**dataset2['grid'])
        x_extent = [exp_grid.xe(5).min(), exp_grid.xe(5).max()]
        y_extent = [exp_grid.ye(5).min(), exp_grid.ye(5).max()]
        map_conf.setdefault('extent', [*x_extent, *y_extent])

    gs = matplotlib.gridspec.GridSpec(nrows=21, ncols=1, hspace=0, wspace=0)
    fig = plt.gcf()

    axes = [
        fig.add_subplot(gs[:9,0], projection=projection),
        fig.add_subplot(gs[10:19,0], projection=projection),
    ]



    for ax, dataset in zip(axes, [dataset1, dataset2]):
        # ax = plt.subplot(2,1,i+1, projection=projection)
        map_axes_formatter_1(ax, **map_conf)

        ds = dataset_factory(**dataset)
        grid = grid_factory(**dataset['grid'])

        if isinstance(grid, sg.grids.CSDataBase):
            for face in kwargs['faces']:
                xx, yy = grid.xe(face), grid.ye(face)
                data = ds[var].isel(lev=kwargs['level'], nf=face).squeeze().values * kwargs['scale']
                pc = plot_pcolomesh(ax, xx, yy, data, **pcolormesh_kwargs)
        else:
            raise NotADirectoryError('Only CS data is supported right now')
        # if i == 0:
        #     plt.gcf().colorbar(pc, orientation='horizontal')
    plt.suptitle(var)
    cbar_axes = fig.add_subplot(gs[20:])
    cbar = plt.colorbar(pc, cax=cbar_axes, orientation='horizontal')
    cbar.set_label(kwargs['units'])

    ref_grid = grid_factory(**dataset1['grid'])
    exp_grid = grid_factory(**dataset2['grid'])
    ref, exp = sg.compare_grids.comparable_gridboxes(
        ref_grid,
        exp_grid,
        dist_tol_abs=kwargs['distance_tolerance'],
        intersect_tol_rel=kwargs['intersect_tolerance'],
        target_lat=exp_grid.target_lat,
        target_lon=exp_grid.target_lon
    )

    # Outline colocated boxes on target face
    colocated_indexes = np.argwhere(exp[0] == 5)

    target_face = exp[0] == 5
    ref = tuple(i[target_face] for i in ref)
    exp = tuple(i[target_face] for i in exp)

    vertical_reduction = lambda x: x.squeeze().isel(lev=kwargs['level']).transpose('nf', 'Ydim', 'Xdim')
    x = dataset_factory(**dataset1)[var].pipe(vertical_reduction).values[ref]
    y = dataset_factory(**dataset2)[var].pipe(vertical_reduction).values[exp]
    rel_diff = np.abs(y - x) / x

    for face in range(6):
        ref_face = (ref[0] == face) & (rel_diff < kwargs['outlier_threshold'])
        exp_face = (exp[0] == face) & (rel_diff < kwargs['outlier_threshold'])
        outline_colocated_boxes(axes[0], ref_grid.xe(face), ref_grid.ye(face), tuple([ref[1][ref_face], ref[2][ref_face]]), color='k', linewidth=0.8, zorder=10)
        outline_colocated_boxes(axes[1], exp_grid.xe(face), exp_grid.ye(face), tuple([exp[1][exp_face], exp[2][exp_face]]), color='k', linewidth=0.8, zorder=10)

        ref_face = (ref[0] == face) & (rel_diff >= kwargs['outlier_threshold'])
        exp_face = (exp[0] == face) & (rel_diff >= kwargs['outlier_threshold'])
        outline_colocated_boxes(axes[0], ref_grid.xe(face), ref_grid.ye(face), tuple([ref[1][ref_face], ref[2][ref_face]]), color='r', linewidth=0.8, zorder=11)
        outline_colocated_boxes(axes[1], exp_grid.xe(face), exp_grid.ye(face), tuple([exp[1][exp_face], exp[2][exp_face]]), color='r', linewidth=0.8, zorder=11)


def r2_vs_sf_line_plot(fig: plt.Figure, var, cs_dataset, sg_datasets, distance_tolerance: float, intersect_tolerance: float, only_target_face=True, **kwargs):
    kwargs.setdefault('scale', 1)
    kwargs.setdefault('units', "1")
    kwargs.setdefault('xlabel', f"Stretch-factor")
    kwargs.setdefault('ylabel', f"$R^2$")
    kwargs.setdefault('level', 0)

    # Get indexes
    x_grid = grid_factory(**cs_dataset['grid'])
    r2 = []
    rmse = []
    mae = []
    sf = []

    ax1 = plt.subplot(2,1,1)
    format_axes_correlation_plot(ax1)

    running_min = None
    running_max = None
    for ds in sg_datasets:
        y_grid = grid_factory(**ds['grid'])
        x_indexes, y_indexes = dimensionality_reduction_factory(
            'colocated-grid-boxes',
            control_grid=x_grid,
            exp_grid=y_grid,
            dist_tol_abs=distance_tolerance,
            intersect_tol_rel=intersect_tolerance,
            target_lat=y_grid.target_lat,
            target_lon=y_grid.target_lon
        )

        if only_target_face:
            target_face = y_indexes[0] == 5
            x_indexes = tuple(i[target_face] for i in x_indexes)
            y_indexes = tuple(i[target_face] for i in y_indexes)

        # Get variables
        vertical_reduction = lambda x: x.squeeze().isel(lev=kwargs['level']).transpose('nf', 'Ydim', 'Xdim')
        x = dataset_factory(**cs_dataset)[var].pipe(vertical_reduction).values[x_indexes]
        y = dataset_factory(**ds)[var].pipe(vertical_reduction).values[y_indexes]

        # Get metrics
        r2.append(sklearn.metrics.r2_score(y_true=x, y_pred=y))
        rmse.append(np.sqrt(sklearn.metrics.mean_squared_error(y_true=x, y_pred=y)))
        mae.append(sklearn.metrics.mean_absolute_error(y_true=x, y_pred=y))
        sf.append(ds['grid']['sf'])

        # Scatter plot
        ax1.scatter(x * kwargs['scale'], y * kwargs['scale'], label=f'S{ds["grid"]["cs"]} vs C{cs_dataset["grid"]["cs"]}', alpha=0.6)

        ax1.set_title(var)
        ax1.set_xlabel(f'Cubed-sphere, [{kwargs["units"]}]')
        ax1.set_ylabel(f'Stretched-grid, [{kwargs["units"]}]')

        lim_pad = 0.95
        if running_min is None:
            running_min = np.min([*x, *y]) * kwargs['scale'] * lim_pad
        else:
            running_min = min(running_min, np.min([*x, *y]) * kwargs['scale'] * lim_pad)
        if running_max is None:
            running_max = np.max([*x, *y]) * kwargs['scale'] / lim_pad
        else:
            running_max = max(running_max, np.min([*x, *y]) * kwargs['scale'] * lim_pad)
    ax1.set_xlim(running_min, running_max)
    ax1.set_ylim(running_min, running_max)
    m1_line = np.linspace(running_min, running_max, 50)
    ax1.plot(m1_line, m1_line, 'k--')
    ax1.legend()

    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(sf, r2)
    for an_sf, an_r2 in zip(sf, r2):
        ax2.scatter(an_sf, an_r2)
    ax2.set_xlabel(kwargs['xlabel'])
    ax2.set_ylabel(kwargs['ylabel'])
    ax2.set_ylim(0.0, 1.2)




if __name__ == '__main__':
    import matplotlib.backends.backend_pdf
    with open('foo.yaml', 'r') as f:
        input = yaml.load(f)

    if input.get('output', {}).get('type', None) == 'pdf':
        pdf = matplotlib.backends.backend_pdf.PdfPages(input['output']['path'])
    for plot_conf in input.get('plots', []):

        fig = plt.figure(figsize=(8.5,11))

        # Call the appropriate plotter
        if plot_conf['type'] == 'colocated-scatter-plot':
            colocated_scatter_plot(plt.gca(), var=plot_conf['var'], **plot_conf['what'])
        elif plot_conf['type'] == 'r2_vs_sf_line_plot':
            r2_vs_sf_line_plot(fig, var=plot_conf['var'], **plot_conf['what'])
        elif plot_conf['type'] == 'side_by_side_pcolormesh':
            pcolormesh_comparison(var=plot_conf['var'], **plot_conf['what'])

        # Handle output
        if input.get('output', {}).get('type', None) == 'pdf':
            pdf.savefig(fig)
        else:
            plt.show()
        plt.close(fig)

    if input.get('output', {}).get('type', None) == 'pdf':
        pdf.close()