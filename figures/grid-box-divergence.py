import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker

import labellines

import matplotlib.colors as colors

import figures


def schmidt_transform(y, s):
    y = y * np.pi / 180
    D = (1 - s ** 2) / (1 + s ** 2)
    y = np.arcsin((D + np.sin(y)) / (1 + D * np.sin(y)))
    return y * 180 / np.pi


if __name__ == '__main__':
    r_earth = 6378.1  # km

    npts = 800
    #y = np.linspace(-90, 90, npts)
    y = np.logspace(1, np.log10(20e3), 2000)/r_earth*180/np.pi - 90
    s = np.linspace(1, 10, npts)
    s = s[1:]
    ss, yy = np.meshgrid(s, y, indexing='ij')

    dydy = np.diff(schmidt_transform(yy, ss)) / np.diff(yy)
    yy2 = schmidt_transform(yy, ss)
    yy2[:,-2] = 90


    plt.figure(figsize=(4, 7))#figures.one_col_figsize(1))



    pcm = plt.pcolormesh(
        ss[:, :-1],
        (yy2[:, :-1] + 90)*np.pi/180 * r_earth,
        np.log2(dydy),
        cmap='RdBu_r',
        vmin=-3.322, vmax=3.322,
        antialiased=True,
        rasterized=True
    )

    cs = plt.contour(
        ss[:, :-1],
        (yy2[:, :-1] + 90)*np.pi/180 * r_earth,
        np.log2(dydy),
        colors=['k', 'k', 'k', 'k', 'k', 'k', 'k'],
        vmin=-3.322, vmax=3.322,
        antialiased=True,
        rasterized=True,
        levels=[-3, -2, -1, 0, 1, 2, 3],
        linewidths=0.1,
    )

    def str_fmt(f):
        if f < 0:
            v = f'{int(2**-f):d}'
            return f'1/{v}'
        else:
            return f'{int(2**f):d}'

    plt.gca().clabel(cs, inline=1, fontsize=8, manual=True, fmt=str_fmt)

    plt.xlim([1, 10])
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # contour = plt.contour(
    #     ss[:, :-1],
    #     (yy2[:, :-1] + 90) * np.pi / 180 * r_earth,
    #     np.log2(dydy),
    #     levels=[-3, -2, -1, 0, 1, 2, 3],
    #     colors='gray',
    #     vmin=-4, vmax=4,
    #     linewidths=0.2,
    #     linestyles='solid'
    # )
    #
    # plt.gca().clabel(
    #     contour,
    #     inline=1,
    #     fontsize='x-small',
    #     inline_spacing=15,
    #     fmt="%1.0f"
    # )

    plt.plot(s, (schmidt_transform(-45, s) + 90)*np.pi/180 * r_earth, 'k--', linewidth=0.5)
    plt.plot(s, (schmidt_transform(45, s) + 90)*np.pi/180 * r_earth, 'k--', linewidth=0.5)

    cbar = plt.colorbar(pcm, ticks=np.linspace(-4, 4, 9))
    cbar.ax.set_yticklabels([f'{2**e}' if e>=0 else f'1/{2**-e}' for e in np.linspace(-4, 4, 9, dtype=int)])

    cbar.ax.get_yaxis().labelpad = 10
    cbar.ax.set_ylabel('Local scaling (unitless)')


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

    plt.xlabel('Stretch-factor parameter, $S$ (unitless)')
    plt.ylabel("Distance from target (km)")

    plt.ylim([100, 20000])

    plt.tight_layout()
    # plt.show()

    figures.savefig(plt.gcf(), 'div-thesis.png', pad_inches=0.05)
    #plt.savefig('/home/liam/Copernicus_LaTeX_Package/figures/grid-box-divergence.png')

