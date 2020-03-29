import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
from tqdm import tqdm

import figures

from sg.grids import CubeSphere, StretchedGrid

from pcolormesh2.pcolormesh2 import pcolormesh2


def draw_major_grid_boxes_naive(ax, xx, yy, **kwargs):
    xx_majors = [xx[0, :], xx[-1, :], xx[:, 0], xx[:, -1]]
    yy_majors = [yy[0, :], yy[-1, :], yy[:, 0], yy[:, -1]]
    for xm, ym in zip(xx_majors, yy_majors):
        ax.plot(xm, ym, transform=ccrs.PlateCarree(), **kwargs, linestyle='-')

def get_native(ds, ID, name):
    cs_res = int(ds.sel(ID=ID).cs_res)
    ID_sel = {f'ID_native_C{cs_res}': ID}
    return ds[f'{name}_native_C{cs_res}'].sel(**ID_sel).squeeze()


filename = '/home/liam/analysis/ensemble-2/c180e.nc'
layer = 0
IDs = ['NA1', 'EU1', 'IN1', 'SE1']
cmap = plt.get_cmap('Dark2')
colors = [cmap(0), cmap(1), cmap(2), cmap(3)]
species = 'O3'
bbox_linewidth=1.5
narrowline=0.3

ds = xr.open_dataset(filename)
grid = CubeSphere(180)

fig = plt.figure(figsize=figures.two_col_figsize(1))

gs = plt.GridSpec(3, 2, height_ratios=[10,10,1], width_ratios=[1,10], left=0.05, right=0.95, top=0.98, bottom=0.07, wspace=0.01, hspace=0.1)

names_ax = [fig.add_subplot(gs[0,0]), fig.add_subplot(gs[1,0])]
names = [f'C180 control simulation', f'Target faces of NA1, EU1, \n IN1, and SE1 simulations']
for ax, row_title in zip(names_ax, names):
    ax.annotate(row_title, xy=(0.5, 0.5), rotation=90, va='center', ha='center', fontsize=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)

ax = fig.add_subplot(gs[0,1],  projection=ccrs.EqualEarth())
ax.coastlines(linewidth=narrowline)
ax.outline_patch.set_linewidth(narrowline)
ax.set_global()


da_ctl = ds[species].isel(lev=layer).sel(ID='CTL').squeeze()
cmap = 'cividis' if species == 'NOx' else 'viridis'

lower_norm = 0 if species == 'NOx' else float(da_ctl.quantile(0.05))*1e9

norm = plt.Normalize(lower_norm, float(da_ctl.quantile(0.95))*1e9)
for face in tqdm(range(6)):
    pcolormesh2(grid.xe(face), grid.ye(face), da_ctl.isel(face=face)*1e9, 180 if face != 2 else 20, norm, cmap=cmap)

da_sg = {}
sg = {}

for ID, c in zip(IDs, colors):
    ds_sg = ds.sel(ID=ID)
    da_sg[ID] = get_native(ds, ID, species).isel(lev=layer).squeeze() #ds_sg[species].isel(lev=layer).squeeze()
    sg[ID] = StretchedGrid(int(ds_sg.cs_res), float(ds_sg.stretch_factor), float(ds_sg.target_lat),
                                   float(ds_sg.target_lon))
    draw_major_grid_boxes_naive(plt.gca(), sg[ID].xe(5), sg[ID].ye(5), color=c, linewidth=bbox_linewidth)

ax = fig.add_subplot(gs[1,1],  projection=ccrs.EqualEarth())
ax.coastlines(linewidth=narrowline)
ax.outline_patch.set_linewidth(narrowline)
ax.set_extent([-133, 135, -17, 78], crs=ccrs.PlateCarree())
ax.set_global()

for ID, c in zip(IDs, colors):
    pc = pcolormesh2(sg[ID].xe(5), sg[ID].ye(5), da_sg[ID]*1e9, int(da_sg[ID].shape[-1]), norm, cmap=cmap)
    draw_major_grid_boxes_naive(plt.gca(), sg[ID].xe(5), sg[ID].ye(5), color=c, linewidth=bbox_linewidth, label=ID)

ax = fig.add_subplot(gs[2,1])
cbar = plt.colorbar(pc, orientation='horizontal', cax=ax)
cbar.set_label(f'Mean {species} concentration (July 2016, level {layer}), [ppb]')
print('Saving to file...')
figures.savefig(plt.gcf(), f'{species}-lev{layer}-map.png')
print('Done!')