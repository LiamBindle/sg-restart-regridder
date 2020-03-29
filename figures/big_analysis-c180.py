import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
import sklearn.metrics
from tqdm import tqdm
import matplotlib.ticker


def plot_3stats(ax1, ax2, ax3, ax4, da,  trop_l, pbl_l, draw_ylabel=False, **sel):
    if not draw_ylabel:
        plt.setp(ax1.get_yticklabels(), visible=False)
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
    #ax1.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(1))

    x2 = da.sel(metric='MB', **sel)/da.sel(metric='CTL_MEAN', **sel)
    ax2.fill_betweenx(da.lev, 0, x2, where=x2 > 0, facecolor='red', label='EXP > CTL')
    ax2.fill_betweenx(da.lev, 0, x2, where=x2 < 0, facecolor='blue', label='EXP < CTL')
    ax2.axvline(0, color='k', linewidth=1)
    ax2.set_xlabel('MB')
    ax2.set_xlim(-0.2, 0.2)
    ax2.grid(True, 'major', 'both')
    ax2.set_xticks([-0.1, 0.1])


    x2 = da.sel(metric='MAE', **sel)/da.sel(metric='CTL_MEAN', **sel)
    ax3.plot(da.sel(metric='RMSE', **sel) / da.sel(metric='CTL_MEAN', **sel), ds.lev, color='tab:orange', linewidth=0.6)
    ax3.barh(da.lev, x2, color='gray')
    ax3.set_xlabel('MAE,RMSE')
    ax3.set_xlim(-0.05, 0.4)
    ax3.set_xticks([0, 0.3])
    ax3.grid(True, 'major', 'both')
    ax3.axvline(0, color='k', linewidth=1)


    c = plt.get_cmap('Purples')(plt.Normalize(vmin=0, vmax=0.2)(da.sel(metric='CTL_STD', **sel)/da.sel(metric='CTL_MEAN', **sel)))

    ax4.scatter(da.sel(metric='R2', **sel), da.lev, color=c, s=0.5)
    ax4.set_xlabel('STD,R2')
    ax4.set_xlim(-0.2, 1.2)
    ax4.set_xticks([0, 1])
    ax4.plot(da.sel(metric='EXP_STD', **sel) / da.sel(metric='EXP_MEAN', **sel), ds.lev, color='tab:blue',
             linewidth=0.6)
    ax4.plot(da.sel(metric='CTL_STD', **sel) / da.sel(metric='CTL_MEAN', **sel), ds.lev, color='tab:purple',
                  linewidth=0.6)
    ax4.grid(True, 'major', 'both')


    ax1.axhline(trop_l, color='tab:blue', linewidth=0.4, linestyle='--')
    ax1.axhline(pbl_l, color='tab:blue', linewidth=0.4, linestyle='--')
    ax2.axhline(trop_l, color='tab:blue', linewidth=0.4, linestyle='--')
    ax2.axhline(pbl_l, color='tab:blue', linewidth=0.4, linestyle='--')
    ax3.axhline(trop_l, color='tab:blue', linewidth=0.4, linestyle='--')
    ax3.axhline(pbl_l, color='tab:blue', linewidth=0.4, linestyle='--')
    ax4.axhline(trop_l, color='tab:blue', linewidth=0.4, linestyle='--')
    ax4.axhline(pbl_l, color='tab:blue', linewidth=0.4, linestyle='--')


def make_figure(fig, ds_total, data_var, title_fmt):
    nrows = 5
    ncols = 3

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
            if index >= len(ds_total.ID):
                continue

            ds_local = ds_total.isel(ID=index)
            da_local = da_total.isel(ID=index)

            sf = float(ds_local.stretch_factor)
            cs_res = float(ds_local.cs_res)
            ID = str(ds_local.ID.item())

            # Make title
            title_ax = fig.add_subplot(gs[0, :])
            title_ax.annotate(title_fmt.format(sf=sf, cs_res=cs_res, ID=ID), xy=(0.6, 0.05), va='center', ha='center')
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

            plot_3stats(ax1, ax2, ax3, ax4, da_local, float(ds_local.Met_TropLev.mean()), float(ds_local.Met_PBLTOPL.mean()), draw_ylabel=j == 0)  # , i=da_total.i[i*ncols+j])


if __name__ == '__main__':
    from matplotlib.backends.backend_pdf import PdfPages

    #font = {'size': 8, 'labelsize': 'small'}

    matplotlib.rc('font', **dict(size=8))
    matplotlib.rc('axes', **dict(labelsize='small'))

    # Load datasets
    ds = xr.open_dataset(f'/home/liam/analysis/ensemble-2/c180e-processed.nc')

    # Concatenate datasets

    ds = ds.sortby('stretch_factor').sortby('cs_res', ascending=False).drop_sel(ID=['CTL', 'NA4']) #.dropna('ID')

    species = ['O3', 'NOx', 'OH', 'CO']


    pp = PdfPages(f'summary-C180.pdf')
    width = 8.5
    height = 11

    for s in species:
        fig = plt.figure(figsize=(width, height), constrained_layout=False)
        # Drop incomplete simulations
        make_figure(fig, ds, s, '\\textbf{{RES:C{cs_res}}}; SF:{sf:5.2f}; ID:{ID}')
        plt.suptitle(f'Species: {s}')
        pp.savefig(fig, papertype='letter', orientation='portrait')

    pp.close()
