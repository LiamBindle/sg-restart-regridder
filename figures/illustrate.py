import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colorbar
import cartopy.crs as ccrs
import pyproj
import shapely
import figures

from sg.grids import StretchedGrid
from sg.compare_grids import get_am_and_pm_masks_and_polygons_outline


def draw_major_grid_boxes_naive(ax, xx, yy, **kwargs):
    kwargs.setdefault('linewidth', 0.3)
    kwargs.setdefault('color', '#151515')

    center_idx = int(xx.shape[0]/2)
    center_x = xx[center_idx, center_idx]
    center_y = yy[center_idx, center_idx]

    gnomonic = ccrs.Gnomonic(center_y, center_x)
    proj = pyproj.Proj(gnomonic.proj4_init)

    xx_majors = [xx[0, :], xx[-1, :], xx[:, 0], xx[:, -1]]
    yy_majors = [yy[0, :], yy[-1, :], yy[:, 0], yy[:, -1]]
    for xm, ym in zip(xx_majors, yy_majors):
        ax.plot(*proj(xm, ym), transform=gnomonic, color='k', linewidth=0.5)


def draw_minor_grid_boxes_naive(ax, xx, yy, **kwargs):
    kwargs.setdefault('linewidth', 0.3)
    kwargs.setdefault('color', '#151515')

    center_idx = int(xx.shape[0]/2)
    center_x = xx[center_idx, center_idx]
    center_y = yy[center_idx, center_idx]

    gnomonic = ccrs.Gnomonic(center_y, center_x)
    proj = pyproj.Proj(gnomonic.proj4_init)

    for x, y in zip(xx[1:, :], yy[1:, :]):
        x, y = proj(x, y)
        ax. plot(x, y, transform=gnomonic, **kwargs)
    for x, y in zip(xx[:, 1:].transpose(), yy[:, 1:].transpose()):
        x, y = proj(x, y)
        ax. plot(x, y, transform=gnomonic, **kwargs)


def grid_box_length(ax, xx_sg, yy_sg, xx_cs, yy_cs, **kwargs):

    areas = [None, None]
    outlines = [None, None]

    for i, (xx, yy) in enumerate(((xx_sg, yy_sg), (xx_cs, yy_cs))):
        #xx[xx > 180] -= 360

        center_idx = int(xx.shape[0] / 2)
        center_x = xx[center_idx, center_idx]
        center_y = yy[center_idx, center_idx]
        x_shape = int(xx.shape[0]-1)

        _, _, boxes = get_am_and_pm_masks_and_polygons_outline(xx, yy)
        equal_area_meters = pyproj.Proj(f'+proj=laea +lon_0={center_x} +lat_0={center_y} +units=m')

        ea_boxes = equal_area_meters(boxes[...,0], boxes[...,1])
        ea_boxes = np.moveaxis(ea_boxes, 0, -1).reshape((x_shape*x_shape, 4, 2))
        outlines[i] = ea_boxes.copy()
        ea_boxes = np.array([shapely.geometry.Polygon(outline) for outline in ea_boxes])

        areas[i] = np.reshape([box.area for box in ea_boxes], (x_shape, x_shape))

    dl = np.sqrt(areas[0]) / np.sqrt(areas[1])

    center_idx = int(xx_sg.shape[0] / 2)
    center_x = xx_sg[center_idx, center_idx]
    center_y = yy_sg[center_idx, center_idx]

    gnomonic = ccrs.Gnomonic(center_y, center_x)
    proj = pyproj.Proj(gnomonic.proj4_init)

    if 'draw_polygons' not in kwargs:
        ax.pcolormesh(
            *proj(xx_sg, yy_sg),
            np.log2(dl),
            cmap='RdBu_r',
            vmin=-4, vmax=4,
            transform=gnomonic,
            **kwargs
        )
    else:
        _, _, boxes = get_am_and_pm_masks_and_polygons_outline(xx_sg, yy_sg)
        boxes = proj(boxes[..., 0], boxes[..., 1])
        boxes = np.moveaxis(boxes, 0, -1).reshape((x_shape*x_shape, 4, 2))
        boxes = [shapely.geometry.Polygon(xy) for xy in boxes]

        cmap = plt.get_cmap('RdBu_r')
        norm = plt.Normalize(vmin=-4, vmax=4)
        for box, value in zip(boxes, np.log2(dl).flatten()):
            c = cmap(norm(value))
            ax.add_geometries([box], gnomonic, edgecolor='None', facecolor=c, linewidth=0)


if __name__ == '__main__':
    fig = plt.figure(figsize=figures.two_col_figsize(2/1))
    gs = plt.GridSpec(2, 5, figure=fig, wspace=0, hspace=0.05, left=0, right=1, bottom=0, top=1, width_ratios=[2,10,10,10,1])

    front_projection = ccrs.Orthographic(0+20, 20)
    back_projection = ccrs.Orthographic(-180+20, 0+20)

    grids = [
        StretchedGrid(16, 1, 0, 0),
        StretchedGrid(16, 2, 0, 0),
        StretchedGrid(16, 6, 0, 0)
    ]

    for i, grid in enumerate(grids):
        ax_front = fig.add_subplot(gs[0, i+1], projection=front_projection)
        ax_back = fig.add_subplot(gs[1, i+1], projection=back_projection)

        ax_front.set_global()
        ax_back.set_global()

        ax_front.outline_patch.set_linewidth(0.2)
        ax_back.outline_patch.set_linewidth(0.2)

        ax_front.set_title(f'SF: {grid.sf}')

        for face in range(6):
            stepsize = 2
            for xs in [slice(s*stepsize, (s+1)*stepsize+1) for s in range(grid.cs//stepsize)]:
                for ys in [slice(s*stepsize, (s+1)*stepsize+1) for s in range(grid.cs//stepsize)]:
                    draw_minor_grid_boxes_naive(ax_front, grid.xe(face)[xs, ys], grid.ye(face)[xs, ys], linewidth=0.1, color='k')
                    draw_minor_grid_boxes_naive(ax_back, grid.xe(face)[xs, ys], grid.ye(face)[xs, ys], linewidth=0.1, color='k')

                    kwargs = {'draw_polygons': True} if i ==2 and face==2 else {}
                    grid_box_length(ax_front, grid.xe(face)[xs, ys], grid.ye(face)[xs, ys], grids[0].xe(face)[xs, ys],
                                    grids[0].ye(face)[xs, ys], **kwargs)
                    grid_box_length(ax_back, grid.xe(face)[xs, ys], grid.ye(face)[xs, ys], grids[0].xe(face)[xs, ys],
                                    grids[0].ye(face)[xs, ys], **kwargs)
            if face != 2:
                draw_major_grid_boxes_naive(ax_front, grid.xe(face), grid.ye(face))
                draw_major_grid_boxes_naive(ax_back, grid.xe(face), grid.ye(face))

    title_axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])]
    for title_ax in title_axes:
        title_ax.spines['top'].set_visible(False)
        title_ax.spines['right'].set_visible(False)
        title_ax.spines['left'].set_visible(False)
        title_ax.spines['bottom'].set_visible(False)
        plt.setp(title_ax.get_xticklabels(), visible=False)
        plt.setp(title_ax.get_yticklabels(), visible=False)
        title_ax.get_yaxis().set_visible(False)
        title_ax.get_xaxis().set_visible(False)
    title_axes[0].annotate('Front', xy=(0.0, 0.5), va='center', ha='left')
    title_axes[1].annotate('Back', xy=(0.0, 0.5), va='center', ha='left')

    cbar_ax = fig.add_subplot(gs[:,-1])
    cb = matplotlib.colorbar.ColorbarBase(
        ax=cbar_ax,
        cmap=plt.get_cmap('RdBu_r'),
        norm=plt.Normalize(-4, 4),
        ticks=np.log2([1/6, 1/2, 1, 2, 6])
    )
    cb.set_ticklabels(['1/6', '1/2', '1', '2', '6'])
    #plt.gcf().colorbar(cb, ax=[ax_front, ax_back])
    plt.tight_layout()
    # plt.show()
    figures.savefig(fig, 'sg-illustrate.png', pad_inches=0.02)
    #plt.savefig('/home/liam/Copernicus_LaTeX_Package/figures/sg-illustrate.png')
