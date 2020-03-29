import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
import sklearn.metrics
from tqdm import tqdm
import matplotlib.ticker
import figures


def plot_3stats(ax1, ax2, ax3, ax4, da,  trop_l, pbl_l, draw_ylabel=False, **sel):
    if not draw_ylabel:
        plt.setp(ax1.get_yticklabels(), visible=False)
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.setp(ax3.get_yticklabels(), visible=False)
    plt.setp(ax4.get_yticklabels(), visible=False)

    narrowline=0.3


    x1 = da.sel(metric='CTL_MEAN', **sel)*1e9
    x2 = da.sel(metric='EXP_MEAN', **sel)*1e9
    ax1.fill_betweenx(da.lev, x1, x2, where=x2 > x1, facecolor='red', label='EXP > CTL')
    ax1.fill_betweenx(da.lev, x1, x2, where=x2 < x1, facecolor='blue', label='EXP < CTL')
    ax1.plot(x1, da.lev, color='gray', linewidth=narrowline, linestyle=':',  label='CTL')
    ax1.semilogx()
    ax1.set_xlabel('ppb')
    if draw_ylabel:
        ax1.set_ylabel('Model level', labelpad=0.5)
    # ax1.grid(True, 'major', 'both')

    x2 = da.sel(metric='MB', **sel)/da.sel(metric='CTL_MEAN', **sel)
    ax2.fill_betweenx(da.lev, 0, x2, where=x2 > 0, facecolor='red', label='EXP > CTL')
    ax2.fill_betweenx(da.lev, 0, x2, where=x2 < 0, facecolor='blue', label='EXP < CTL')
    ax2.axvline(0, color='k', linewidth=narrowline)
    ax2.set_xlabel('MB')
    ax2.set_xlim(-0.2, 0.2)
    #ax2.grid(True, 'major', 'both')
    ax2.set_xticks([-0.1, 0.1])


    x2 = da.sel(metric='MAE', **sel)/da.sel(metric='CTL_MEAN', **sel)
    ax3.plot(da.sel(metric='RMSE', **sel) / da.sel(metric='CTL_MEAN', **sel), ds.lev, color='tab:orange', linewidth=1.5*narrowline)
    ax3.barh(da.lev, x2, color='gray', height=0.3)
    ax3.set_xlabel('MAE,RMSE')
    ax3.set_xlim(0, 0.4)
    ax3.set_xticks([0, 0.3])
    #ax3.grid(True, 'major', 'both')
    #ax3.axvline(0, color='k', linewidth=narrowline)


    c = plt.get_cmap('Purples')(plt.Normalize(vmin=0, vmax=0.2)(da.sel(metric='CTL_STD', **sel)/da.sel(metric='CTL_MEAN', **sel)))

    ax4.scatter(da.sel(metric='R2', **sel), da.lev, color=c, s=0.1)
    ax4.set_xlabel('R2')
    ax4.set_xlim(0, 1.2)
    ax4.set_xticks([0.2, 1])
    ax4.axvline(1, color='k', linewidth=narrowline)
    ax4_twin = ax4.twiny()


    x1 = da.sel(metric='CTL_STD', **sel) / da.sel(metric='CTL_MEAN', **sel)
    x2 = da.sel(metric='EXP_STD', **sel) / da.sel(metric='EXP_MEAN', **sel)
    ax4_twin.fill_betweenx(da.lev, x1, x2, where=x2 > x1, facecolor='palevioletred', label='EXP > CTL')
    ax4_twin.fill_betweenx(da.lev, x1, x2, where=x2 < x1, facecolor='cornflowerblue', label='EXP < CTL')
    ax4_twin.plot(x1, ds.lev, color='tab:purple',
                  linewidth=1.5*narrowline)
    ax4_twin.set_xlabel('STD', labelpad=5)
    ax4_twin.set_xlim(0, float(x1.max()))
    ax4_twin.axvline(0, color='k', linewidth=narrowline)


    ax1.axhline(trop_l, color='k', linewidth=narrowline, linestyle='--')
    ax1.axhline(pbl_l, color='k', linewidth=narrowline, linestyle='--')
    ax2.axhline(trop_l, color='k', linewidth=narrowline, linestyle='--')
    ax2.axhline(pbl_l, color='k', linewidth=narrowline, linestyle='--')
    ax3.axhline(trop_l, color='k', linewidth=narrowline, linestyle='--')
    ax3.axhline(pbl_l, color='k', linewidth=narrowline, linestyle='--')
    ax4.axhline(trop_l, color='k', linewidth=narrowline, linestyle='--')
    ax4.axhline(pbl_l, color='k', linewidth=narrowline, linestyle='--')

    tick_params = {
        'length': 2,
        'width': 0.5,
        'pad': 0.5
    }

    for ax in [ax1, ax2, ax3, ax4, ax4_twin]:
        ax.tick_params(**tick_params)
        ax.xaxis.labelpad = 0.5
        ax.set_ylim(0, 72)
        plt.setp(ax.spines.values(), linewidth=0.5)


def make_figure(fig, ds_total, data_var, title_fmt):
    nrows = 4
    ncols = 4

    page_wmargin = 0.1
    page_hmargin = 0.05

    pwidth = (1-2*page_wmargin) / ncols
    pheight = (1-2*page_hmargin) / nrows

    mwidth = 0.02
    mheight = 0.05

    hratio = [1, 8]

    da_total = ds_total[data_var]


    row_sel = ['NA{column}', 'EU{column}', 'IN{column}', 'SE{column}']

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

            col = j+1
            if i == 2 and j > 0:
                col = j
            if i == 2 and j == 1:
                filled_axes = fig.add_subplot(gs[1:, :])
                filled_axes.fill([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], 'gray', alpha=0.3)
                filled_axes.spines['top'].set_visible(False)
                filled_axes.spines['right'].set_visible(False)
                filled_axes.spines['left'].set_visible(False)
                filled_axes.spines['bottom'].set_visible(False)
                plt.setp(filled_axes.get_xticklabels(), visible=False)
                plt.setp(filled_axes.get_yticklabels(), visible=False)
                filled_axes.get_yaxis().set_visible(False)
                filled_axes.get_xaxis().set_visible(False)
                continue

            index = row_sel[i].format(column=col)

            if index == 'NA4':
                title_ax = fig.add_subplot(gs[:, :])
                title_ax.annotate('NA4 crashed', xy=(0.5, 0.5), va='center', ha='center')
                title_ax.spines['top'].set_visible(False)
                title_ax.spines['right'].set_visible(False)
                title_ax.spines['left'].set_visible(False)
                title_ax.spines['bottom'].set_visible(False)
                plt.setp(title_ax.get_xticklabels(), visible=False)
                plt.setp(title_ax.get_yticklabels(), visible=False)
                title_ax.get_yaxis().set_visible(False)
                title_ax.get_xaxis().set_visible(False)


            if not index in ds_total.ID:
                continue

            if np.isnan(ds_total.sel(ID=index).stretch_factor.item()):
                filled_axes = fig.add_subplot(gs[1:, :])
                filled_axes.fill([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], 'gray', alpha=0.3)
                filled_axes.spines['top'].set_visible(False)
                filled_axes.spines['right'].set_visible(False)
                filled_axes.spines['left'].set_visible(False)
                filled_axes.spines['bottom'].set_visible(False)
                plt.setp(filled_axes.get_xticklabels(), visible=False)
                plt.setp(filled_axes.get_yticklabels(), visible=False)
                filled_axes.get_yaxis().set_visible(False)
                filled_axes.get_xaxis().set_visible(False)
                continue

            ds_local = ds_total.sel(ID=index)
            da_local = da_total.sel(ID=index)

            sf = float(ds_local.stretch_factor)
            cs_res = int(ds_local.cs_res)
            ID = str(ds_local.ID.item())

            # Make title
            title_ax = fig.add_subplot(gs[0, :])
            title_ax.annotate(title_fmt.format(sf=sf, cs_res=cs_res, ID=ID), xy=(0.0, 0.05), va='center', ha='left')
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

            plot_3stats(ax1, ax2, ax3, ax4, da_local, float(ds_local.Met_TropLev.mean())-1, float(ds_local.Met_PBLTOPL.mean())-1, draw_ylabel=j == 0)  # , i=da_total.i[i*ncols+j])


if __name__ == '__main__':
    from matplotlib.backends.backend_pdf import PdfPages

    #font = {'size': 8, 'labelsize': 'small'}

    matplotlib.rc('font', **dict(size=4))
    matplotlib.rc('axes', **dict(labelsize='small'))

    res = 720

    # Load datasets
    ds = xr.open_dataset(f'/home/liam/analysis/ensemble-2/c{res}e-processed.nc')

    # Concatenate datasets

    ds = ds.sortby('stretch_factor').sortby('cs_res', ascending=False).drop_sel(ID=['CTL', 'NA4']) #.dropna('ID')

    species = ['O3', 'NOx']# , 'NOx', 'OH', 'CO']


    # pp = PdfPages(f'summary-C180.pdf')
    # width = 4#8.5
    # height = 6#11

    figures.display_figure_instead = False

    for s in species:
        fig = plt.figure(figsize=figures.two_col_figsize(4/3), constrained_layout=False)
        # Drop incomplete simulations
        make_figure(fig, ds, s, '\\textbf{{{ID}}}: C{cs_res}, SF:{sf:5.2f}')
        #plt.suptitle(f'Species: {s}')
        figures.savefig(fig, f'{s}-summary-{res}.eps')
        # pp.savefig(fig, papertype='letter', orientation='portrait')

    # pp.close()
