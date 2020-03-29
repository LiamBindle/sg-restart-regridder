import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == '__main__':

    cmap = plt.get_cmap('Dark2')
    dtfal = 4*np.pi/180

    grids = [
        {'SF': 2, 'Lat0': 35, 'Lon0': 264, 'Region': 'NA', 'ID': 'NA1'},
        {'SF': 3.6, 'Lat0': 38, 'Lon0': 252, 'Region': 'NA', 'ID': 'NA2'},
        {'SF': 6.8, 'Lat0': 37, 'Lon0': 244, 'Region': 'NA', 'ID': 'NA3'},
        {'SF': 12.5, 'Lat0': 36, 'Lon0': 241, 'Region': 'NA', 'ID': 'NA4'},

        {'SF': 2, 'Lat0': 48, 'Lon0': 14, 'Region': 'EU', 'ID': 'EU1'},
        {'SF': 3.4, 'Lat0': 47, 'Lon0': 5, 'Region': 'EU', 'ID': 'EU2'},
        {'SF': 6.8, 'Lat0': 42.5, 'Lon0': 12.5, 'Region': 'EU', 'ID': 'EU3'},
        {'SF': 15, 'Lat0': 45, 'Lon0': 10.5, 'Region': 'EU', 'ID': 'EU4'},

        {'SF': 2.8, 'Lat0': 21.5, 'Lon0': 79, 'Region': 'IN', 'ID': 'IN1'},
        {'SF': 6, 'Lat0': 25, 'Lon0': 81, 'Region': 'IN', 'ID': 'IN2'},
        {'SF': 14, 'Lat0': 27.5, 'Lon0': 78.5, 'Region': 'IN', 'ID': 'IN3'},

        {'SF': 2.7, 'Lat0': 5, 'Lon0': 111, 'Region': 'SE', 'ID': 'SE1'},
        {'SF': 4, 'Lat0': 0, 'Lon0': 108, 'Region': 'SE', 'ID': 'SE2'},
        {'SF': 7, 'Lat0': -5, 'Lon0': 110, 'Region': 'SE', 'ID': 'SE3'},
        {'SF': 15, 'Lat0': -7, 'Lon0': 108, 'Region': 'SE', 'ID': 'SE4'},
    ]

    print(grids.__len__())

    df = pd.DataFrame(grids).set_index(['Region', 'ID'])

    df['SF'] = df['SF'].map('{:> 4.1f}'.format)
    df['Lat0'] = df['Lat0'].map('{: 4.1f}'.format)
    df['Lon0'] = df['Lon0'].map('{: 5.1f}'.format)

    table = df.to_latex(
        index_names=True,
        multirow=True,
        col_space=0
    )
    print(table)




