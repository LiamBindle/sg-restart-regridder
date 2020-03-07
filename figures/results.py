import os.path
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.legend_handler import HandlerTuple, HandlerLine2D
import matplotlib.colors as colors


import pandas as pd


import sklearn.metrics
import sklearn.linear_model



def eval_metric(df: pd.DataFrame, callable_metric):
    scores = df.copy()
    del scores['fnames']
    is_first = True
    for lineno, fname in zip(df.index, df['fnames']):
        if os.path.exists(fname):
            data = pd.read_csv(fname)
            species = list(set([cname.split(':')[0] for cname in data.columns]))
            layers = list(set([int(cname.split(':')[2]) for cname in data.columns]))
            species.sort()
            layers.sort()

            if is_first:
                for s in species:
                    for l in layers:
                        scores[f'{s}:{l}'] = np.ones_like(df['fnames']) * np.nan
                is_first=False

            for s in species:
                for l in layers:
                    x_cname = f'{s}:CTL:{l}'
                    y_cname = f'{s}:EXP:{l}'
                    new_cname = f'{s}:{l}'
                    scores[new_cname].at[lineno] = callable_metric(y_true=data[x_cname], y_pred=data[y_cname])
    return scores


def load_results(control_res: int, target_res: int):
    basedir = '/extra-space/line-7'
    sims_fname = f'{basedir}/c{target_res}e_sims.csv'
    simulations = pd.read_csv(
        sims_fname,
        usecols=[3,5,6,7],
        names=['sf', 'lat0', 'lon0', 'region']
    )
    simulations.index += 1
    fnames = [f'{basedir}/c{control_res}-c{target_res}e-{i}.csv' for i in simulations.index]
    simulations['fnames'] = pd.Series(fnames, index=simulations.index)
    return simulations

def make_table(df: pd.DataFrame, select, **metrics):
    output = pd.DataFrame(
        {
            'sf': df['sf'],
            'lat0': df['lat0'],
            'lon0': df['lon0'],
            'region': df['region'],
        },
        index=df.index
    )
    for name, scorer in metrics.items():
        output[name] = eval_metric(df, scorer)[select]
    return output.set_index(['region', 'sf', 'lat0', 'lon0'])


def make_scoring_table(df, what, level):

    score_r2 = sklearn.metrics.r2_score
    score_nrmse = lambda y_true, y_pred: np.sqrt(sklearn.metrics.mean_squared_error(y_true, y_pred)) / np.mean(y_true)
    score_nmae = lambda y_true, y_pred: sklearn.metrics.mean_absolute_error(y_true, y_pred) / np.mean(y_true)
    score_nmb = lambda y_true, y_pred: np.sum(y_pred - y_true) / np.sum(y_true)

    output = make_table(df, f'{what}:{level}', R2=score_r2, NRMSE=score_nrmse, NMAE=score_nmae, NMB=score_nmb)

    output = output.dropna()
    output['NRMSE'] *= 100
    output['NMAE'] *= 100
    output['NMB'] *= 100
    # output['SF'] = output['SF'].map('{:.1f}'.format)
    output['R2'] = output['R2'].map('{:.3f}'.format)
    output['NRMSE'] = output['NRMSE'].map('{:.1f}'.format)
    output['NMAE'] = output['NMAE'].map('{:.1f}'.format)
    output['NMB'] = output['NMB'].map('{:.1f}'.format)

    output = output[['R2', 'NRMSE', 'NMAE', 'NMB']]
    return output

def scoring_table(what, level, control_res, target_res):
    simulations = pd.read_csv(
        '/home/liam/temp2/sg-restart-regridder/figures/experiments.csv',
        names=['sf', 'lat0', 'lon0', 'region']
    )
    simulations.set_index(['region', 'sf', 'lat0', 'lon0'], inplace=True)

    mc = [[f'c{r}e' for r in target_res], ['R2', 'NRMSE', 'NMAE', 'NMB']]
    mc = pd.MultiIndex.from_product(mc, names=['target_res', 'metric'])
    simulations = pd.DataFrame(index=simulations.index, columns=mc)

    for res in target_res:
        data_table = load_results(control_res=control_res, target_res=res)
        simulations[f'c{res}e'] = make_scoring_table(data_table, what, level)
    simulations.index = simulations.index.map(
        lambda t: (t[0], f'{t[1]:.1f}', f'{t[2]:.1f}', f'{t[3] if t[3] < 180 else t[3] - 360:.1f}')
    )

    what = what.replace('SpeciesConc_', '')

    table = simulations.to_latex(
        index_names=True,
        multirow=True,
        multicolumn=True,
        na_rep='---',
        caption=f'Scoring metrics for {what} in model layer {level}',
        col_space=0
    )

    table = table.replace('\\begin{tabular}', '\\scriptsize\n\\begin{tabular}')

    return table



if __name__ == '__main__':


    table = scoring_table('SpeciesConc_O3', 0, 180, [180, 360, 720])
    print(table)
    table = scoring_table('SpeciesConc_O3', 10, 180, [180, 360, 720])
    print(table)


    # normalized_rmse = eval_metric(df, score_nrmse)
    # normalized_rmae = eval_metric(df, score_nmae)
    # normalized_nb = eval_metric(df, score_nmb)
    # slope = eval_metric(df, score_slope)
    # intercept = eval_metric(df, score_intercept)

    # plt.figure()
    #
    # lines = []
    # clabels = []
    # markers = []
    # marker_names = ['NA', 'EU', 'IN', 'SE']
    # for c1, control_res in enumerate([90, 180]):
    #     for c2, target_res in enumerate([180, 360, 720]):
    #         df = load_results(control_res=control_res, target_res=target_res)
    #         r2_scores = eval_metric(df, sklearn.metrics.r2_score)
    #
    #         NA = r2_scores.loc[r2_scores['region'].str.contains('NA')]
    #         EU = r2_scores.loc[r2_scores['region'].str.contains('EU')]
    #         IN = r2_scores.loc[r2_scores['region'].str.contains('IN')]
    #         SE = r2_scores.loc[r2_scores['region'].str.contains('SE')]
    #
    #         color = plt.get_cmap('tab10').colors[3*c1 + c2]
    #         clabels.append(f'S{target_res} vs C{control_res}')
    #
    #         p, = plt.plot(NA['sf'], NA['SpeciesConc_O3:0'], color=color)
    #         plt.plot(EU['sf'], EU['SpeciesConc_O3:0'], color=color)
    #         plt.plot(IN['sf'], IN['SpeciesConc_O3:0'], color=color)
    #         plt.plot(SE['sf'], SE['SpeciesConc_O3:0'], color=color)
    #         m1, = plt.plot(NA['sf'], NA['SpeciesConc_O3:0'], color=color, marker='o', linestyle="")
    #         m2, = plt.plot(EU['sf'], EU['SpeciesConc_O3:0'], color=color, marker='v', linestyle="")
    #         m3, = plt.plot(IN['sf'], IN['SpeciesConc_O3:0'], color=color, marker='X', linestyle="")
    #         m4, = plt.plot(SE['sf'], SE['SpeciesConc_O3:0'], color=color, marker='d', linestyle="")
    #         markers.append([m1, m2, m3, m4])
    #         lines.append(p)
    #
    #
    # plt.legend([*lines, *markers], [*clabels, *marker_names],
    #            handler_map={tuple: HandlerTuple(ndivide=None), p: HandlerLine2D()})
    # plt.xlabel('Stretch factor')
    # plt.ylabel('$r^2$ score')
    #
    # plt.xlim([0, max(r2_scores['sf'])+1])
    # plt.ylim([0.4, 1.2])
    #
    # plt.show()


