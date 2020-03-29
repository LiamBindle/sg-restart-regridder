import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
import sklearn.metrics
from tqdm import tqdm


def plot_3stats(ax1, ax2, ax3, ax4, da, draw_ylabel=False, **sel):
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.setp(ax3.get_yticklabels(), visible=False)
    plt.setp(ax4.get_yticklabels(), visible=False)


    x1 = da.sel(metric='CTL_MEAN', **sel)*1e9
    x2 = da.sel(metric='EXP_MEAN', **sel)*1e9
    ax1.fill_betweenx(da.lev, x1, x2, where=x2 > x1, facecolor='red', label='EXP > CTL')
    ax1.fill_betweenx(da.lev, x1, x2, where=x2 < x1, facecolor='blue', label='EXP < CTL')
    ax1.scatter(x1, da.lev, s=0.01, color='gray', label='CTL')
    ax1.semilogx()
    ax1.set_xlabel('ppb')
    if draw_ylabel:
        ax1.set_ylabel('Model level')
    ax1.grid(True, 'major', 'both')

    x2 = da.sel(metric='MB', **sel)/da.sel(metric='CTL_MEAN', **sel)
    ax2.fill_betweenx(da.lev, 0, x2, where=x2 > 0, facecolor='red', label='EXP > CTL')
    ax2.fill_betweenx(da.lev, 0, x2, where=x2 < 0, facecolor='blue', label='EXP < CTL')
    ax2.axvline(0, color='k', linewidth=1)
    ax2.set_xlabel('NMB')
    ax2.set_xlim(-0.2, 0.2)
    ax2.grid(True, 'major', 'both')
    ax2.set_xticks([-0.1, 0.1])


    x2 = da.sel(metric='MAE', **sel)/da.sel(metric='CTL_MEAN', **sel)
    ax3.plot(da.sel(metric='RMSE', **sel) / da.sel(metric='CTL_MEAN', **sel), ds.lev, color='tab:orange', linewidth=0.6)
    ax3.barh(da.lev, x2, color='gray')
    ax3.set_xlabel('NMAE, RMSE')
    ax3.set_xlim(-0.05, 0.4)
    ax3.set_xticks([0, 0.3])
    ax3.grid(True, 'major', 'both')
    ax3.axvline(0, color='k', linewidth=1)


    c = plt.get_cmap('Purples')(plt.Normalize(vmin=0, vmax=0.2)(da.sel(metric='CTL_STD', **sel)/da.sel(metric='CTL_MEAN', **sel)))

    ax4.scatter(da.sel(metric='R2', **sel), da.lev, color=c, s=0.5)
    ax4.set_xlabel('R2')
    ax4.set_xlim(-0.2, 1.2)
    ax4.set_xticks([0, 1])
    ax4.plot(da.sel(metric='EXP_STD', **sel) / da.sel(metric='EXP_MEAN', **sel), ds.lev, color='tab:blue',
                  linewidth=0.6)
    ax4.plot(da.sel(metric='CTL_STD', **sel) / da.sel(metric='CTL_MEAN', **sel), ds.lev, color='tab:purple',
                  linewidth=0.6)
    ax4.grid(True, 'major', 'both')


def make_figure(fig, ds_total, data_var, title_fmt):
    nrows = 6
    ncols = 5

    page_wmargin = 0.1
    page_hmargin = 0.05

    pwidth = (1-2*page_wmargin) / ncols
    pheight = (1-2*page_hmargin) / nrows

    mwidth = 0.02
    mheight = 0.03

    hratio = [1, 8]

    da_total = ds_total[data_var]

    for i in range(nrows):
        for j in range(ncols):

            left = max(page_wmargin, j * pwidth + mwidth / 2 + page_wmargin)
            right = min(1-page_wmargin, (j + 1) * pwidth - mwidth / 2 + page_wmargin)
            bottom = max(page_hmargin, 1 - (i + 1) * pheight + mheight / 2 - page_hmargin)
            top = min(1-page_hmargin, 1 - i * pheight - mheight / 2 - page_hmargin)

            gs = fig.add_gridspec(
                2, 4,
                wspace=0.2,
                height_ratios=hratio,
                left=left, right=right,
                bottom=bottom, top=top,
            )

            index = i * ncols + j
            if index >= len(ds_total.i):
                continue

            ds_local = ds_total.isel(i=index)
            da_local = da_total.isel(i=index)

            sf = float(ds_local.stretch_factor)
            cs_res = float(ds_local.cs_res)
            ID = str(ds_local.i.values.item()[1])
            res = str(ds_local.i.values.item()[0])

            # Make title
            title_ax = fig.add_subplot(gs[0, :])
            title_ax.annotate(title_fmt.format(sf=sf, cs_res=cs_res, ID=ID, res=res), xy=(0.6, 0.05), va='center', ha='center')
            title_ax.spines['top'].set_visible(False)
            title_ax.spines['right'].set_visible(False)
            title_ax.spines['left'].set_visible(False)
            title_ax.spines['bottom'].set_visible(False)
            plt.setp(title_ax.get_xticklabels(), visible=False)
            plt.setp(title_ax.get_yticklabels(), visible=False)
            title_ax.get_yaxis().set_visible(False)
            title_ax.get_xaxis().set_visible(False)

            # Make plots
            ax1 = fig.add_subplot(gs[1, 0])
            ax2 = fig.add_subplot(gs[1, 1])
            ax3 = fig.add_subplot(gs[1, 2])
            ax4 = fig.add_subplot(gs[1, 3])

            plot_3stats(ax1, ax2, ax3, ax4, da_local, draw_ylabel=j == 0)  # , i=da_total.i[i*ncols+j])


if __name__ == '__main__':
    from matplotlib.backends.backend_pdf import PdfPages

    font = {'size': 4}

    matplotlib.rc('font', **font)

    # Load datasets
    ds = {res: xr.open_dataset(f'/home/liam/analysis/ensemble-2/old-1/c{res}e-processed.nc') for res in [180, 360, 720]}

    # Concatenate datasets
    ds = xr.concat([ds[180], ds[360], ds[720]], dim='res')  # concatenate datasets
    ds = ds.assign_coords({'res': [180, 360, 720]})   # add 'res' coordinate
    ds = ds.stack(i=('res', 'ID'))                    # stack 'res' and 'ID'

    sortbys = [None, 'stretch_factor', 'cs_res']
    title_fmts = [
        '\\textbf{{ID:{ID},c{res}e}}',
        '\\textbf{{SF:{sf:5.2f}}}; RES:{cs_res}',
        '\\textbf{{RES:{cs_res}}}; SF:{sf:5.2f}',
    ]
    suptitles = [
        'Not sorted',
        'Sorted by stretch-factor',
        'Sorted by base resolution',
    ]

    species = 'NOx'

    pp = PdfPages(f'{species}.pdf')
    for sortby, title, suptitle in zip(sortbys, title_fmts, suptitles):


        #width = 4.7 * 2
        #height = 6.1 * 2
        width = 8.5
        height = 11
        fig = plt.figure(figsize=(width, height), constrained_layout=False)


        # Sort datasets
        if sortby is not None:
            if sortby == 'cs_res':
                ds_total = ds.sortby(sortby, ascending=False)
            else:
                ds_total = ds.sortby(sortby)
        else:
            ds_total = ds.copy()

        # Drop incomplete simulations
        ds_total = ds_total.dropna('i')
        make_figure(fig, ds_total, species, title)

        #plt.suptitle(suptitle)
        #plt.tight_layout()
        pp.savefig(fig, papertype='letter', orientation='portrait')
    pp.close()
