import numpy as np


# def auxiliary_X(xi):
#     return np.tan(xi)
#
#
# def auxiliary_Y(eta):
#     return np.tan(eta)


def auxiliary_delta(X, Y):
    return 1 + X**2 + Y**2


def auxiliary_C(X):
    return np.sqrt(1 + X**2)


def auxiliary_D(Y):
    return np.sqrt(1 + Y**2)


def spherical_to_local_equatorial(X, Y):
    delta = auxiliary_delta(X, Y)
    C = auxiliary_C(X)
    D = auxiliary_D(Y)
    return np.array([[0, C*D/np.sqrt(delta)], [-1, X*Y/np.sqrt(delta)]])


def spherical_to_local_north_pole(X, Y):
    delta = auxiliary_delta(X, Y)
    C = auxiliary_C(X)
    D = auxiliary_D(Y)
    return np.array([[D*X, -D*Y/np.sqrt(delta)], [C*Y, C*X/np.sqrt(delta)]]) / np.sqrt(delta - 1)


def spherical_to_local_south_pole(X, Y):
    delta = auxiliary_delta(X, Y)
    C = auxiliary_C(X)
    D = auxiliary_D(Y)
    return np.array([[-D*X, D*Y/np.sqrt(delta)], [-C*Y, -C*X/np.sqrt(delta)]]) / np.sqrt(delta - 1)


def XY_I(phi, theta):
    X = np.tan(phi)
    Y = 1/(np.tan(theta) * np.cos(phi))
    return X, Y


def XY_II(phi, theta):
    X = -1/np.tan(phi)
    Y = 1/(np.tan(theta) * np.sin(phi))
    return X, Y


def XY_III(phi, theta):
    X = np.tan(phi)
    Y = -1/(np.tan(theta) * np.cos(phi))
    return X, Y


def XY_IV(phi, theta):
    X = -1/np.tan(phi)
    Y = -1/(np.tan(theta) * np.sin(phi))
    return X, Y


def XY_V(phi, theta):
    X = np.tan(theta) * np.sin(phi)
    Y = -np.tan(theta) * np.cos(phi)
    return X, Y


def XY_VI(phi, theta):
    X = -np.tan(theta) * np.sin(phi)
    Y = -np.tan(theta) * np.cos(phi)
    return X, Y


def change_basis_gmao_1(lon, lat):
    X, Y = XY_I(lon, lat)
    M = spherical_to_local_equatorial(X, Y)
    return M


def rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


if __name__ == '__main__':
    import matplotlib as plt
    import cartopy.crs as ccrs
    from sg.grids import CubeSphere, StretchedGrid
    from sg.figure_axes import FigureAxes
    from sg.plot import *
    from tqdm import tqdm

    f = plt.figure()
    grid = CubeSphere(48)

    proj = ccrs.PlateCarree()
    # proj = ccrs.NearsidePerspective(360 - 78, 36)

    ax = plt.subplot(1, 1, 1, projection=proj)
    figax = FigureAxes(ax, proj)

    ax.set_global()
    ax.coastlines(linewidth=0.8)

    lon = -53.42
    lat = 33.76
    east = np.array([[0], [1]])
    north = np.array([[1], [0]])

    # Get U
    X, Y = XY_I(lon+10, 90-lat)
    M = spherical_to_local_equatorial(X, Y)
    U = rotation_matrix(np.pi/2) @ M @ east
    V = rotation_matrix(np.pi/2) @ M @ north

    for i in range(6):
        draw_minor_grid_boxes(figax, *figax.transform_xy(grid.xe(i), grid.ye(i)))
        draw_major_grid_boxes(figax, *figax.transform_xy(grid.xe(i), grid.ye(i)))
        draw_face_number(figax, *figax.transform_xy(grid.xe(i), grid.ye(i)), face=i+1)

    for xx, yy in tqdm(zip(grid.xe(0)[::4,::4].flatten(), grid.ye(0)[::4,::4].flatten())):
        X, Y = XY_I(np.deg2rad(xx+10), np.deg2rad(90-yy))
        M = rotation_matrix(np.pi/2) @ spherical_to_local_equatorial(X, Y)
        U = M @ east
        V = M @ north
        plt.quiver(xx, yy, U[1], U[0], pivot='tail', angles='xy', scale_units='xy', scale=0.5, color='g')
        plt.quiver(xx, yy, V[1], V[0], pivot='tail', angles='xy', scale_units='xy', scale=0.5, color='m')

    # for xx, yy in tqdm(zip(grid.xe(0)[::4,::4].flatten(), grid.ye(0)[::4,::4].flatten())):
    #     X, Y = XY_I(xx+10, yy)
    #     M = rotation_matrix(np.pi/2) @ spherical_to_local_equatorial(X, Y)
    #     U = M @ east
    #     plt.quiver(xx, yy, U[1], U[0], pivot='tail', angles='xy', scale_units='xy', scale=0.5, color='g')

    plt.quiver(lon, lat, east[1], east[0], pivot='tail', angles='xy', scale_units='xy', scale=0.5, color='blue')
    plt.quiver(lon, lat, north[1], north[0], pivot='tail', angles='xy', scale_units='xy', scale=0.5, color='red')
    # plt.quiver(lon, lat, U[1], U[0], pivot='tail', angles='xy', scale_units='xy', scale=0.5, color='g')
    # plt.quiver(lon, lat, V[1], V[0], pivot='tail', angles='xy', scale_units='xy', scale=0.5, color='m')
    plt.tight_layout()
    plt.show()
