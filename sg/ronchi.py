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
    M = np.array([[np.zeros_like(X), C*D/np.sqrt(delta)], [-np.ones_like(X), X*Y/np.sqrt(delta)]])
    M = np.moveaxis(M, 0, -1)   # roll back once
    M = np.moveaxis(M, 0, -1)   # roll back again
    return M


def spherical_to_local_north_pole(X, Y):
    delta = auxiliary_delta(X, Y)
    C = auxiliary_C(X)
    D = auxiliary_D(Y)
    M = np.array([[D*X, -D*Y/np.sqrt(delta)], [C*Y, C*X/np.sqrt(delta)]]) / np.sqrt(delta - 1)
    M = np.moveaxis(M, 0, -1)   # roll back once
    M = np.moveaxis(M, 0, -1)   # roll back again
    return M


def spherical_to_local_south_pole(X, Y):
    delta = auxiliary_delta(X, Y)
    C = auxiliary_C(X)
    D = auxiliary_D(Y)
    M = np.array([[-D*X, D*Y/np.sqrt(delta)], [-C*Y, -C*X/np.sqrt(delta)]]) / np.sqrt(delta - 1)
    M = np.moveaxis(M, 0, -1)   # roll back once
    M = np.moveaxis(M, 0, -1)   # roll back again
    return M


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


def spherical_to_ronchi(phi, theta, ronchi_face: int):
    # Ronchi basis to lon,lat
    if ronchi_face < 5:
        if ronchi_face == 1:
            X, Y = XY_I(phi, theta)
        elif ronchi_face == 2:
            X, Y = XY_II(phi, theta)
        elif ronchi_face == 3:
            X, Y = XY_III(phi, theta)
        elif ronchi_face == 4:
            X, Y = XY_IV(phi, theta)
        R = spherical_to_local_equatorial(X, Y)
    elif ronchi_face == 5:
        X, Y = XY_V(phi, theta)
        R = spherical_to_local_north_pole(X, Y)
    elif ronchi_face == 6:
        X, Y = XY_VI(phi, theta)
        R = spherical_to_local_south_pole(X, Y)
    R = np.flip(R, axis=-1)                 # RH: colat,lon to lon,colat
    Rm = R @ np.array([[1, 0], [0, -1]])    # RH: lon,colat to lon,lat
    return Rm

def spherical_to_gmao(phi, theta, gmao_face: int):
    if gmao_face == 1:
        ronchi_face = 1
        gmao_to_ronchi = np.array([
            [1, 0],
            [0, 1]
        ]) #    ^---- psi in ronchi
           # ^------- zeta in ronchi
    if gmao_face == 2:
        ronchi_face = 2
        gmao_to_ronchi = np.array([
            [1, 0],
            [0, 1]
        ])
    if gmao_face == 3:
        ronchi_face = 5
        gmao_to_ronchi = np.array([
            [0, -1],
            [1,  0]
        ])
    if gmao_face == 4:
        ronchi_face = 3
        gmao_to_ronchi = np.array([
            [ 0, 1],
            [-1, 0]
        ])
    if gmao_face == 5:
        ronchi_face = 4
        gmao_to_ronchi = np.array([
            [ 0, 1],
            [-1, 0]
        ])
    if gmao_face == 6:
        ronchi_face = 6
        gmao_to_ronchi = np.array([
            [1, 0],
            [0, 1]
        ])
    Rm = spherical_to_ronchi(phi, theta, ronchi_face=ronchi_face)
    G  = np.linalg.inv(gmao_to_ronchi) @ Rm
    return G



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

    U = np.array([1, 0])
    V = np.array([0, 1])

    for i in range(6):
        draw_minor_grid_boxes(figax, *figax.transform_xy(grid.xe(i), grid.ye(i)))
        draw_major_grid_boxes(figax, *figax.transform_xy(grid.xe(i), grid.ye(i)))
        draw_face_number(figax, *figax.transform_xy(grid.xe(i), grid.ye(i)), face=i+1)

    # GMAO Face 1
    face = 6
    xx = grid.xe(face - 1)[::4, ::4]
    yy = grid.ye(face - 1)[::4, ::4]
    G = spherical_to_gmao(np.deg2rad(xx+10), np.deg2rad(90-yy), gmao_face=face)
    # xi = np.inner(np.linalg.inv(G), U)
    # eta = np.inner(np.linalg.inv(G), V)
    # xi = np.linalg.inv(G) @ U
    # eta = np.linalg.inv(G) @ V

    G_inv = np.linalg.inv(G)
    xi = np.tensordot(G_inv, U, (3, 0))
    eta = np.tensordot(G_inv, V, (3, 0))

    ax.quiver(xx, yy, xi[:, :, 0], xi[:, :,1], pivot='tail', angles='xy', scale_units='xy', scale=0.5, color='g')
    ax.quiver(xx, yy, eta[:, :, 0], eta[:, :, 1], pivot='tail', angles='xy', scale_units='xy', scale=0.5, color='m')

    # for xx, yy in tqdm(zip(grid.xe(face-1)[::4,::4].flatten(), grid.ye(face-1)[::4,::4].flatten())):
    #     # Rm = lon_lat_to_xi_eta(np.deg2rad(xx+10), np.deg2rad(90-yy), ronchi_face=5)
    #     G = spherical_to_gmao(np.deg2rad(xx+10), np.deg2rad(90-yy), gmao_face=face)
    #
    #     xi = np.linalg.inv(G) @ U
    #     eta = np.linalg.inv(G) @ V
    #
    #     ax.quiver(xx, yy, *xi, pivot='tail', angles='xy', scale_units='xy', scale=0.5, color='g')
    #     ax.quiver(xx, yy, *eta, pivot='tail', angles='xy', scale_units='xy', scale=0.5, color='m')

    # # GMAO Face 2
    # for xx, yy in tqdm(zip(grid.xe(1)[::4,::4].flatten(), grid.ye(1)[::4,::4].flatten())):
    #     M = change_of_basis_matrix(np.deg2rad(xx+10), np.deg2rad(90-yy), 2, 0)
    #     xi = M @ U
    #     eta = M @ V
    #     plt.quiver(xx, yy, *xi, pivot='tail', angles='xy', scale_units='xy', scale=0.5, color='g')
    #     plt.quiver(xx, yy, *eta, pivot='tail', angles='xy', scale_units='xy', scale=0.5, color='m')

    # # GMAO Face 4
    # for xx, yy in tqdm(zip(grid.xe(3)[::4, ::4].flatten(), grid.ye(3)[::4, ::4].flatten())):
    #     M = change_of_basis_matrix(np.deg2rad(xx + 10), np.deg2rad(90 - yy), 3, -np.pi/2)
    #     xi = M @ U
    #     eta = M @ V
    #     plt.quiver(xx, yy, *xi, pivot='tail', angles='xy', scale_units='xy', scale=0.5, color='g')
    #     plt.quiver(xx, yy, *eta, pivot='tail', angles='xy', scale_units='xy', scale=0.5, color='m')

    # # GMAO Face 5
    # for xx, yy in tqdm(zip(grid.xe(4)[::4, ::4].flatten(), grid.ye(4)[::4, ::4].flatten())):
    #     M = change_of_basis_matrix(np.deg2rad(xx + 10), np.deg2rad(90 - yy), 4, -np.pi/2)
    #     xi = M @ U
    #     eta = M @ V
    #     plt.quiver(xx, yy, *xi, pivot='tail', angles='xy', scale_units='xy', scale=0.5, color='g')
    #     plt.quiver(xx, yy, *eta, pivot='tail', angles='xy', scale_units='xy', scale=0.5, color='m')

    # # GMAO Face 3 (direction looks bad)
    # for xx, yy in tqdm(zip(grid.xe(2)[::4, ::4].flatten(), grid.ye(2)[::4, ::4].flatten())):
    #     M = change_of_basis_matrix(np.deg2rad(xx + 10), np.deg2rad(90 - yy), 5, -np.pi/2)
    #     M = np.flip(M, axis=0)
    #     eta = M @ U
    #     xi = M @ V
    #     plt.quiver(xx, yy, *xi, pivot='tail', angles='xy', scale_units='xy', scale=0.5, color='g')
    #     plt.quiver(xx, yy, *eta, pivot='tail', angles='xy', scale_units='xy', scale=0.5, color='m')

    # # GMAO Face 6 (direction looks bad)
    # for xx, yy in tqdm(zip(grid.xe(5)[::4, ::4].flatten(), grid.ye(5)[::4, ::4].flatten())):
    #     M = change_of_basis_matrix(np.deg2rad(xx + 10), np.deg2rad(90 - yy), 6, 0)
    #     xi = M @ U
    #     eta = M @ V
    #     plt.quiver(xx, yy, *xi, pivot='tail', angles='xy', scale_units='xy', scale=0.5, color='g')
    #     plt.quiver(xx, yy, *eta, pivot='tail', angles='xy', scale_units='xy', scale=0.5, color='m')

    # for xx, yy in tqdm(zip(grid.xe(0)[::4,::4].flatten(), grid.ye(0)[::4,::4].flatten())):
    #     X, Y = XY_I(xx+10, yy)
    #     M = rotation_matrix(np.pi/2) @ spherical_to_local_equatorial(X, Y)
    #     U = M @ east
    #     plt.quiver(xx, yy, U[1], U[0], pivot='tail', angles='xy', scale_units='xy', scale=0.5, color='g')

    # plt.quiver(lon, lat, east[1], east[0], pivot='tail', angles='xy', scale_units='xy', scale=0.5, color='blue')
    # plt.quiver(lon, lat, north[1], north[0], pivot='tail', angles='xy', scale_units='xy', scale=0.5, color='red')
    # plt.quiver(lon, lat, U[1], U[0], pivot='tail', angles='xy', scale_units='xy', scale=0.5, color='g')
    # plt.quiver(lon, lat, V[1], V[0], pivot='tail', angles='xy', scale_units='xy', scale=0.5, color='m')
    plt.tight_layout()
    plt.show()
