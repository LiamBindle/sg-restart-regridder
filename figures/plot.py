import numpy as np
import matplotlib.pyplot as plt


import matplotlib.colors as colors


def schmidt_transform(y, s):
    y = y * np.pi / 180
    D = (1 - s ** 2) / (1 + s ** 2)
    y = np.arcsin((D + np.sin(y)) / (1 + D * np.sin(y)))
    return y * 180 / np.pi


if __name__ == '__main__':

    npts = 1500
    y = np.linspace(-90, 90, npts)
    s = np.linspace(1, 15, npts)
    s = s[1:]
    ss, yy = np.meshgrid(s, y, indexing='ij')

    dyp_m_dy = np.diff(schmidt_transform(yy, ss)) - np.diff(yy)
    yy2 = schmidt_transform(yy, ss)


    plt.figure()

    geoseries = np.cumsum(0.5 ** np.arange(5)) - 1
    spacing = [*-geoseries[::-1], *geoseries[1:]]

    cmap = plt.get_cmap('RdBu')
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap.N,
    )
    norm = colors.BoundaryNorm(spacing, cmap.N)

    r_earth = 6378.1  # km

    Z = dyp_m_dy/np.diff(yy)
    pcm = plt.pcolormesh(
        ss[:, :-1],
        (yy2[:, :-1] + 90)*np.pi/180 * r_earth,
        Z,
        norm=norm,
        cmap=cmap,
        antialiased=True
    )

    plt.plot(s, (schmidt_transform(-45, s) + 90)*np.pi/180 * r_earth, 'k--', linewidth=0.5)
    plt.plot(s, (schmidt_transform(45, s) + 90)*np.pi/180 * r_earth, 'k--', linewidth=0.5)

    cbar = plt.colorbar(pcm)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_yticklabels([f'{2**g}' if g>=0 else f'1/{2**-g}' for g in np.arange(-4, 5)])


    # What's being plotted: relative change in grid-box-length divergence resulting from Schmidt transform

    plt.yscale('log')
    # plt.grid(
    #     True,
    #     which="both",
    #     axis='y',
    # )

    # plt.text(12.4, 748, '1/8', horizontalalignment='center', verticalalignment='center')
    # plt.text(9.82, 1559, '1/4', horizontalalignment='center', verticalalignment='center')
    # plt.text(5.1, 3140, '1/2', horizontalalignment='center', verticalalignment='center')
    # plt.text(12.89, 3402, '1', horizontalalignment='center', verticalalignment='center')
    # plt.text(10.52, 14288, '> 8', horizontalalignment='center', verticalalignment='center')

    plt.xlabel('Stretch Factor')
    plt.ylabel("Distance from target, [km]")

    plt.show()


