import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from sg.grids import CubeSphere, StretchedGrid


def draw_minor_grid_boxes(ax, xx, yy, **kwargs):
    kwargs.setdefault('linewidth', 0.3)
    kwargs.setdefault('color', '#151515')
    for x, y in zip(xx, yy):
        idx = np.argwhere(np.diff(np.sign(x % 360 - 180))).flatten()
        x360 = x % 360
        idx = idx[(x360[idx] > 10) & (x360[idx] < 350)]
        start = [0, *(idx + 1)]
        end = [*(idx + 1), len(x)]
        for s, e in zip(start, end):
            ax.plot(x[s:e], y[s:e], transform=ccrs.PlateCarree(), **kwargs)
    for x, y in zip(xx.transpose(), yy.transpose()):
        idx = np.argwhere(np.diff(np.sign(x % 360 - 180))).flatten()
        x360 = x % 360
        idx = idx[(x360[idx] > 10) & (x360[idx] < 350)]
        start = [0, *(idx + 1)]
        end = [*(idx + 1), len(x)]
        for s, e in zip(start, end):
            ax.plot(x[s:e], y[s:e], transform=ccrs.PlateCarree(), **kwargs)

def draw_grid_boxes(ax, grid):
    for i in range(6):
        xx = grid.xe(i)
        yy = grid.ye(i)

        xx[xx > 180] -= 360

        draw_minor_grid_boxes(ax, xx, yy, linewidth=0.3)

        xx_majors = [xx[0, :], xx[-1, :], xx[:, 0], xx[:, -1]]
        yy_majors = [yy[0, :], yy[-1, :], yy[:, 0], yy[:, -1]]
        for xm, ym in zip(xx_majors, yy_majors):
            ax.plot(xm, ym, transform=ccrs.PlateCarree(), color='k', linewidth=0.5)

if __name__ == '__main__':
    fig = plt.figure(constrained_layout=True)
    gs = plt.GridSpec(3, 3, figure=fig)

    front_projection = ccrs.Orthographic(0+20, 20)
    back_projection = ccrs.Orthographic(-180+20, 0+20)

    grids = [
        StretchedGrid(24, 1, 0, 0),
        StretchedGrid(24, 3, 0, 0),
        StretchedGrid(24, 12, 0, 0)
    ]

    for i, grid in enumerate(grids):
        ax_front = fig.add_subplot(gs[0, i], projection=front_projection)
        ax_back = fig.add_subplot(gs[1, i], projection=back_projection)

        draw_grid_boxes(ax_front, grid)
        draw_grid_boxes(ax_back, grid)

    plt.show()
