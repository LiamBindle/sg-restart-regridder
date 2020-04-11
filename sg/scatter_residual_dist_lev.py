import numpy as np
import cartopy.crs as ccrs
import pyproj
import matplotlib.pyplot as plt
import shapely.geometry
import argparse
import os.path
import ast
import matplotlib.gridspec

from sg.compare_grids2 import determine_blocksize, get_minor_xy


def schmidt_transform(y, s):
    y = y * np.pi / 180
    D = (1 - s ** 2) / (1 + s ** 2)
    y = np.arcsin((D + np.sin(y)) / (1 + D * np.sin(y)))
    return y * 180 / np.pi


from mpl_toolkits.axes_grid1 import make_axes_locatable


if __name__ == '__main__':
    import xarray as xr
    import yaml
    import  matplotlib.cm
    parser = argparse.ArgumentParser(description='Make a map')
    parser.add_argument('filein',
                        metavar='FILEIN',
                        type=str,
                        help="input file")
    parser.add_argument('--ctl',
                        metavar='CONTROL_FILE',
                        type=str,
                        required=True,
                        help='control input file')
    parser.add_argument('-v', '--var',
                        metavar='NAME',
                        type=str,
                        required=True,
                        help='path to the control\'s output directory')
    parser.add_argument('--conf',
                        metavar='C',
                        type=str,
                        required=True,
                        help='conf file')
    parser.add_argument('--sel',
                        type=str,
                        nargs='+',
                        default=[],
                        help='selectors')
    parser.add_argument('--isel',
                        metavar='NAME',
                        type=str,
                        nargs='+',
                        default=[],
                        help='index selectors')
    parser.add_argument('-o',
                        metavar='O',
                        type=str,
                        default='output.png',
                        help='path to output')
    parser.add_argument('--cmap',
                        metavar='CMAP',
                        type=str,
                        default='cividis',
                        help='color map')
    args = vars(parser.parse_args())
    plt.rc('text', usetex=False)

    ds = xr.open_dataset(args['filein'])
    ds_ctl = xr.open_dataset(args['ctl'])

    for sel_k, sel_v_str in zip(args['sel'][::2], args['sel'][1::2]):
        try:
            sel_v = ast.literal_eval(sel_v_str)
        except ValueError:
            sel_v = sel_v_str
        ds = ds.sel(**{sel_k: sel_v})
        ds_ctl = ds_ctl.sel(**{sel_k: sel_v})

    for isel_k, isel_v in zip(args['isel'][::2], args['isel'][1::2]):
        ds = ds.isel(**{isel_k: eval(isel_v)})
        ds_ctl = ds_ctl.isel(**{isel_k: eval(isel_v)})

    with open(args['conf'], 'r') as f:
        conf = yaml.safe_load(f)

    stackable = ['nf', 'Ydim', 'Xdim', 'lev']
    stacked = [s for s in stackable if s in ds[args['var']].dims]

    ds = ds.stack(pts=stacked)
    ds_ctl = ds_ctl.stack(pts=stacked)

    da = ds[args['var']].squeeze()
    da_ctl = ds_ctl[args['var']].squeeze()

    residual = (da - da_ctl) / da_ctl

    ylim = [-0.3, 0.3]

    DEG2RAD = np.pi/180
    R_EARTH=6378.1e3

    level = da.lev
    distance = ds['distance_from_target'] * DEG2RAD * R_EARTH / 1000

    compute_nmae = lambda y_pred, y_true: np.sum(abs(y_pred - y_true)) / sum(y_true)
    compute_nmb = lambda y_pred, y_true: np.sum(y_pred - y_true) / np.sum(y_true)
    compute_nrmse = lambda y_pred, y_true: np.sqrt(np.sum((y_pred - y_true)**2 / y_pred.size)) / y_true.mean()


    vline1 = (schmidt_transform(-45, conf['grid']['stretch_factor']) + 90) * DEG2RAD * R_EARTH / 1000
    vline2 = (schmidt_transform(-45+90, conf['grid']['stretch_factor']) + 90) * DEG2RAD * R_EARTH / 1000

    dist_bins = [*np.linspace(distance.min()*1.01, 1e3, 10), *np.logspace(np.log10(1e3), np.log10(1e4), 10)]
    dist_selects = [distance < d for d in dist_bins]
    mb =  [compute_nmb (da[sel].values, da_ctl[sel].values) for sel in dist_selects]
    mae = [compute_nmae(da[sel].values, da_ctl[sel].values) for sel in dist_selects]
    rmse = [compute_nrmse(da[sel].values, da_ctl[sel].values) for sel in dist_selects]

    fig = plt.figure(figsize=(4.724,4.724))

    norm = plt.Normalize(level.min(), level.max())
    cmap = plt.get_cmap(args['cmap'])

    level_colors = cmap(norm(level))

    gs = matplotlib.gridspec.GridSpec(2, 5, fig, left=0.05, right=0.85, top=0.95, bottom=0.05, wspace=0, hspace=0.05, width_ratios=[2.5, 5, 5, 0.5, 0.5], height_ratios=[10, 1.5])

    xtext_ax =  fig.add_subplot(gs[1, 1:3])
    ytext_ax = fig.add_subplot(gs[0, 0])

    for textax in [xtext_ax, ytext_ax]:
        textax.spines['top'].set_visible(False)
        textax.spines['right'].set_visible(False)
        textax.spines['left'].set_visible(False)
        textax.spines['bottom'].set_visible(False)
        plt.setp(textax.get_xticklabels(), visible=False)
        plt.setp(textax.get_yticklabels(), visible=False)
        textax.get_yaxis().set_visible(False)
        textax.get_xaxis().set_visible(False)

    ytext_ax.annotate('Stretched-grid residual, [1]', xy=(0.0, 0.5), va='center', ha='left', rotation=90)
    xtext_ax.annotate('Grid-box distance to target, [km]', xy=(0.5, 0.0), va='bottom', ha='center')

    ax_linear = fig.add_subplot(gs[0, 1])
    ax_linear.set_xscale('linear')
    ax_linear.set_xlim((0, 1000))
    ax_linear.spines['right'].set_visible(False)
    ax_linear.set_ylim([-0.3, 0.3])
    ax_linear.set_xticks([0, 500])

    ax_log = fig.add_subplot(gs[0, 2], sharey=ax_linear)
    ax_log.set_xscale('log')
    ax_log.set_xlim((1000, 20e3))
    ax_log.set_xticks([1e3, 1e4])
    ax_log.spines['left'].set_visible(False)
    ax_log.yaxis.set_ticks_position('left')
    ax_log.yaxis.set_visible(False)
    plt.setp(ax_log .get_xticklabels(), visible=True)
    ax_log.set_ylim([-0.3, 0.3])


    ax_linear.scatter(distance[::-1], residual[::-1], marker='.', s=1, c=level_colors[::-1], alpha=0.7)
    ax_log.scatter(distance[::-1], residual[::-1], marker='.', s=1, c=level_colors[::-1], alpha=0.7)

    ax_linear.plot(dist_bins, mb, label='MB')
    ax_log.plot(dist_bins, mb, label='MB')
    ax_linear.plot(dist_bins, mae, label='MAE')
    ax_log.plot(dist_bins, mae, label='MAE')
    ax_linear.plot(dist_bins, rmse, label='RMSE')
    ax_log.plot(dist_bins, rmse, label='RMSE')

    ax_linear.legend(loc='upper left')
    ax_linear.axhline(0, color='k', linewidth=0.9)
    ax_log.axhline(0, color='k', linewidth=0.9)
    ax_linear.axvline(vline1, color='k', linestyle='--', linewidth=1.8)
    ax_linear.axvline(vline2, color='k', linestyle='--', linewidth=1.8)
    ax_log.axvline(vline1, color='k', linestyle='--', linewidth=1.8)
    ax_log.axvline(vline2, color='k', linestyle='--', linewidth=1.8)

    cbar_ax = fig.add_subplot(gs[0, 4])
    cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm, cmap), cax=cbar_ax)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('Model level', rotation=270)

    plt.savefig(args['o'], dpi=300)