import sys

import sklearn.metrics

import numpy as np
import pandas as pd
import xarray as xr


def mean_bias(y_true, y_pred):
    return (y_pred.mean() - y_true.mean()).item()

def schmidt_transform(y, s):
    y = y * np.pi / 180
    D = (1 - s ** 2) / (1 + s ** 2)
    y = np.arcsin((D + np.sin(y)) / (1 + D * np.sin(y)))
    return y * 180 / np.pi

if __name__ == '__main__':
    fnames = sys.argv[1:]

    species = ['O3', 'NOx', 'OH', 'CO']

    min_trop_lev = 25     # CTL min Tropopause level was 27.88 (1-based level-index)
    # area_ratio_tol = 2.3  # Normal CSs have max/min grid-box area of ~2.3
    intersect_qtol = 0.7  # Intersect tolerance quantile

    mi = pd.MultiIndex.from_product([['O3', 'NOx', 'CO', 'OH'], ['MB', 'MAE', 'RMSE', 'R2']])
    df = pd.DataFrame(columns=mi)


    for fname in fnames:
        ds = xr.open_dataset(fname)

        short_name = ds.attrs['short_name']
        sf = ds.attrs['stretch_factor']
        tf_edge = schmidt_transform(-45, sf) + 90

        ds = ds.isel(lev=slice(0, min_trop_lev+1))
        ds = ds.where(ds.distance_from_target < tf_edge)

        intersect_tol = ds.max_intersect.quantile(intersect_qtol).item()
        ds = ds.where(ds.max_intersect > intersect_tol)

        ds = ds.stack(pts=['lev', 'nf', 'Ydim', 'Xdim']).squeeze()
        ds = ds.dropna('pts')

        def score_species(df, species):
            mean = ds[f'{species}_CTL'].mean().item()
            df[species, 'MB'] = f"{mean_bias(ds[f'{species}_CTL'], ds[species]) / mean * 100:.1f}"
            df[species, 'MAE'] = f"{sklearn.metrics.mean_absolute_error(ds[f'{species}_CTL'], ds[species]) / mean * 100:.1f}"
            df[species, 'RMSE'] = f"{np.sqrt(sklearn.metrics.mean_squared_error(ds[f'{species}_CTL'], ds[species])) / mean * 100:.1f}"
            df[species, 'R2'] = f"{sklearn.metrics.r2_score(ds[f'{species}_CTL'], ds[species]):.2f}"
            return df

        df_entry = pd.DataFrame(index=[short_name], columns=mi)
        df_entry = score_species(df_entry, 'O3')
        df_entry = score_species(df_entry, 'NOx')
        df_entry = score_species(df_entry, 'CO')
        df_entry = score_species(df_entry, 'OH')

        df = df.append(df_entry)

    print(df.to_latex(multicolumn=True))













