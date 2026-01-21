from exp_stability import exp_stability

from read_auc import read_auc

import os
import random
import argparse
import numpy as np

import matplotlib.pyplot as plt


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stability measured by Euclidean distance')
    parser.add_argument('-d', '--data', default='svmguide3', help='name of the dataset (default: svmguide3)')
    parser.add_argument('-algo', '--algorithm', default='SGDA', help='name of the algorithm')
    args = parser.parse_args()

    set_seed(42)

    for eta_p in [1e-2, 3e-2, 5e-2]:
        exp_stability(args.data, args.algorithm, eta_p)