import numpy as np
import cartopy.crs as ccrs
import pyproj
import matplotlib.pyplot as plt
import shapely.geometry
import argparse
import os.path
import ast
import matplotlib.gridspec
import matplotlib.ticker

import figures

from sg.compare_grids2 import determine_blocksize, get_minor_xy


def schmidt_transform(y, s):
    y = y * np.pi / 180
    D = (1 - s ** 2) / (1 + s ** 2)
    y = np.arcsin((D + np.sin(y)) / (1 + D * np.sin(y)))
    return y * 180 / np.pi


def invisible_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)


if __name__ == '__main__':
    import xarray as xr
    import yaml
    import  matplotlib.cm
    parser = argparse.ArgumentParser(description='Make a map')
    parser.add_argument('filein',
                        nargs='+',
                        metavar='FILEIN',
                        type=str,
                        help="input file")
    parser.add_argument('-v', '--var',
                        metavar='NAME',
                        type=str,
                        required=True,
                        help='path to the control\'s output directory')
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
    parser.add_argument('--lt',
                        metavar='EXPR',
                        type=str,
                        nargs='+',
                        default=[],
                        help='less than')
    parser.add_argument('--gt',
                        metavar='EXPR',
                        type=str,
                        nargs='+',
                        default=[],
                        help='greater than')
    parser.add_argument('--square-residuals',
                        action='store_true')
    parser.add_argument('-o',
                        metavar='O',
                        type=str,
                        default='output.png',
                        help='path to output')
    parser.add_argument('--cmap',
                        metavar='CMAP',
                        type=str,
                        default='Dark2',
                        help='color map')
    args = vars(parser.parse_args())


    nfiles = len(args['filein'])

    fig = plt.figure(figsize=figures.one_col_figsize(0.5))
    gs = fig.add_gridspec(nfiles+1, 1, height_ratios=[5, 5, 2], hspace=0.35, top=0.98, left=0.2, bottom=0)

    axes = [None]*nfiles

    # Read data
    for fileno, filein in enumerate(args['filein']):
        ds = xr.open_dataset(filein)

        axes[fileno] = fig.add_subplot(gs[fileno, 0])
        ax = axes[fileno]

        # Select data
        for sel_k, sel_v_str in zip(args['sel'][::2], args['sel'][1::2]):
            try:
                sel_v = ast.literal_eval(sel_v_str)
            except ValueError:
                sel_v = sel_v_str
            ds = ds.sel(**{sel_k: sel_v})

        for isel_k, isel_v in zip(args['isel'][::2], args['isel'][1::2]):
            ds = ds.isel(**{isel_k: eval(isel_v)})

        for where_key, where_lt in zip(args['lt'][::2], args['lt'][1::2]):
            where_arg = ds[where_key] < eval(where_lt)
            ds = ds.where(where_arg)

        for where_key, where_lt in zip(args['gt'][::2], args['gt'][1::2]):
            where_arg = ds[where_key] > eval(where_lt)
            ds = ds.where(where_arg)

        # Put data into 1D array
        stackable = ['nf', 'Ydim', 'Xdim', 'lev']
        stacked = [s for s in stackable if s in ds[args['var']].dims]

        ds = ds.stack(pts=stacked)

        da = ds[args['var']].squeeze()
        da_ctl = ds[f"{args['var']}_CTL"].squeeze()

        if args['square_residuals']:
            ds['residuals'] = (da - da_ctl)**2 / da_ctl**2
        else:
            ds['residuals'] = (da - da_ctl) / da_ctl

        ylim = [-0.3, 0.3]
        level = da.lev
        compute_nmae = lambda y_pred, y_true: np.sum(abs(y_pred - y_true)) / sum(y_true)
        compute_nmb = lambda y_pred, y_true: np.sum(y_pred - y_true) / np.sum(y_true)
        compute_nrmse = lambda y_pred, y_true: np.sqrt(np.sum((y_pred - y_true)**2 / y_pred.size)) / y_true.mean()

        drop = [v for v in ds.variables.keys() if v not in ['distance_from_target', 'max_intersect', 'residuals']]
        ds_new = ds.drop(drop)

        distance_bins = [
            (schmidt_transform(pos, ds.attrs['stretch_factor']) + 90) for pos in [-90, -67.5, -45, 0, 45, 67.5, 90]
        ]

        ds_new = ds_new.set_index({'pts': 'distance_from_target'}).rename({'pts': 'distance_from_target'})

        residual_bins = ds_new.residuals.groupby_bins('distance_from_target', distance_bins)

        intervals = [i.mid for i, _ in residual_bins]
        residuals = [np.random.choice(d, size=min(d.size, 10000), replace=False) for _, d in residual_bins]

        intervals, residuals = zip(*sorted(zip(intervals, residuals)))

        dy = 0.01
        gbl_div = np.array([np.diff(schmidt_transform(np.array([y-90, y+dy-90]), ds.attrs['stretch_factor'])).item() for y in intervals])/dy * ds.attrs['stretch_factor']

        boxplot = ax.boxplot(
            residuals,
            labels=[f'{dgbl:.2f}' for dgbl in gbl_div], #['TF1', 'TF2', 'NFs1', 'NFs2', 'ATF1', 'ATF2'],
            showfliers=False,
            patch_artist=True
        )

        cmap = plt.get_cmap(args['cmap'])
        colors = [cmap(0), cmap(0), cmap(1), cmap(1), cmap(2), cmap(2)] #[cmap(fileno), cmap(fileno), cmap(fileno), cmap(fileno), cmap(fileno), cmap(fileno)]

        for patch, color in zip(boxplot['boxes'], colors):
            patch.set_facecolor(color)
        plt.setp(boxplot['medians'], color='k')

        ax.axhline(0, color='k')

        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.3))
        ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))
        ax.yaxis.grid(True, which='both')

        ax.set_ylim(*[-0.3, 0.3])
        ax.set_ylabel('NOx residuals', labelpad=0)
        ax.set_xlabel("Median edge-enlargement")
        ax.text(0.05, 0.95, ds.attrs['short_name'], horizontalalignment='left', verticalalignment='top',  transform=ax.transAxes)

    ax = fig.add_subplot(gs[2,0])
    invisible_axes(ax)

    boxes = [boxplot["boxes"][i] for i in range(6)]
    label = ['TF Bin 1', 'TF Bin 2', 'NFs Bin 1', 'NFs Bin 2', 'ATF Bin 1', 'ATF Bin 2']

    ax.legend(
        boxes, label,
        loc='center',
        mode='expand', ncol=2,
        handlelength=1, handletextpad=0.3, columnspacing=1,
    )

    #plt.tight_layout()
    #plt.show()
    # figures.display_figure_instead=True
    figures.savefig(fig, 'NOx_residuals_bp_distance_bins.eps', pad_inches=0.01)

    print('here')


