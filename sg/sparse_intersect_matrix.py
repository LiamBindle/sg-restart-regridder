import argparse
import yaml
import os.path

import numpy as np
import scipy.sparse

from sg.compare_grids2 import ciwam2, revist, normalize, look_for_missing_intersections
from sg.grids import StretchedGrid, CubeSphere


def print_matrix_stats(M, desc):
    total_weights = M.sum(axis=1)
    print(f'Intersect matrix ({desc})')
    print(f'    Mean:  {total_weights.mean()}')
    print(f'    P1Err: {np.quantile(1-abs(1-total_weights), 0.01)}')
    print(f'    P2Err: {np.quantile(1-abs(1-total_weights), 0.02)}')
    print(f'    P5Err: {np.quantile(1-abs(1-total_weights), 0.05)}')
    print(f'    <0.98: {np.count_nonzero(total_weights < 0.98)}/{total_weights.size}')
    print(f'    >1.02: {np.count_nonzero(total_weights > 1.02)}/{total_weights.size}')
    print(f'    Zeros: {np.count_nonzero(total_weights == 0.0)}/{total_weights.size}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates a sparse intersect matrix')
    parser.add_argument('-c',
                        metavar='CTL_PATH',
                        type=str,
                        required=True,
                        help='path to the control run directory')
    parser.add_argument('-e',
                        metavar='EXP_PATH',
                        type=str,
                        required=True,
                        help='path to the experiment run directory')
    parser.add_argument('-f', action='store_true',
                        help='force all stages')
    args = vars(parser.parse_args())

    with open(os.path.join(args['c'], 'conf.yml'), 'r') as f:
        ctl_conf = yaml.safe_load(f)
    with open(os.path.join(args['e'], 'conf.yml'), 'r') as f:
        exp_conf = yaml.safe_load(f)

    if 'stretch_factor' in ctl_conf['grid']:
        ctl_grid = StretchedGrid(
            cs=ctl_conf['grid']['cs_res'],
            sf=ctl_conf['grid']['stretch_factor'],
            target_lat=ctl_conf['grid']['target_lat'],
            target_lon= ctl_conf['grid']['target_lon']
        )
    else:
        ctl_grid = CubeSphere(
            cs=ctl_conf['grid']['cs_res'],
        )
    if 'stretch_factor' in exp_conf['grid']:
        exp_grid = StretchedGrid(
            cs=exp_conf['grid']['cs_res'],
            sf=exp_conf['grid']['stretch_factor'],
            target_lat=exp_conf['grid']['target_lat'],
            target_lon= exp_conf['grid']['target_lon']
        )
    else:
        exp_grid = CubeSphere(
            cs=exp_conf['grid']['cs_res'],
        )

    init_fname = 'sparse_intersect-init.npz'
    init_path = os.path.join(args['e'], init_fname)
    if os.path.exists(init_path) and not args['f']:
        print(f'Loading {init_path}')
        M = scipy.sparse.load_npz(init_path)
        print_matrix_stats(M, 'initial')
    else:
        print('Generating sparse intersect matrix')
        M = ciwam2(exp_grid, ctl_grid)
        print_matrix_stats(M, 'initial')
        print(f'Saving to {init_fname}\n')
        scipy.sparse.save_npz(init_path, M)

    print('Recalculating intersects with enhanced gridbox edges')
    M = revist(M, exp_grid, ctl_grid, 0.99)
    print_matrix_stats(M, 'post-revisit1')
    revisted_fname = 'sparse_intersect-revisit1.npz'
    print(f'Saving to {revisted_fname}\n')
    scipy.sparse.save_npz(os.path.join(args['e'], revisted_fname), M)

    print('Looking for intersections that might have been missed')
    M = look_for_missing_intersections(M, exp_grid, ctl_grid, tol=0.98)
    print_matrix_stats(M, 'post-revisit2')
    revisted_fname = 'sparse_intersect-revisit2.npz'
    print(f'Saving to {revisted_fname}\n')
    scipy.sparse.save_npz(os.path.join(args['e'], revisted_fname), M)

    print('Normalizing rows (intersect-weighted average)')
    M = normalize(M)
    print_matrix_stats(M, 'post-normalize')
    normalized_fname = 'sparse_intersect.npz'
    print(f'Saving to {normalized_fname}\n')
    scipy.sparse.save_npz(os.path.join(args['e'], normalized_fname), M)

    print('Done!')

