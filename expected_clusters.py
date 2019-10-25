import os
import argparse

import numpy as np
import scipy.special as special

import matplotlib.pyplot as plt
from matplotlib import cm

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

cmap = viridis = cm.get_cmap('viridis', 100)

parser = argparse.ArgumentParser()

parser.add_argument("-o", "--outdir", default="results", type=str)
parser.add_argument("-a", "--alpha", default=1, type=float)
parser.add_argument("-n", "--num_samples", default=100000, type=int)
parser.add_argument('-ms', "--max_clusters", default=[5,10,50,100,1000,10000], nargs='+', type=int)
parser.add_argument('-s', '--show', action='store_true')

args = parser.parse_args()

alpha = args.alpha
n = args.num_samples
ms = args.max_clusters
outdir = args.outdir

def expected_clusters_finite_exact(M, n, alpha):
    expected = np.zeros(n)
    expected[0] = 1

    for i in range(2, n+1):
        expected[i-1] = (1 - (alpha/M)/(alpha + i - 1)) * expected[i-2] + (alpha / (alpha + i - 1))

    return expected

def expected_clusters_infinite_exact(n, alpha):
    # expected = np.zeros(n)
    # expected[0] = 1

    # for i in range(2, n+1):
    #     expected[i-1] = expected[i-2] + (alpha / (alpha + i - 1))

    expected = np.arange(n) + 1
    expected = alpha * (special.digamma(alpha + expected) - special.digamma(alpha))

    return expected


def expected_clusters_infinite_approx(n, alpha):
    expected = np.arange(n) + 1
    return alpha * np.log(1 + (expected / alpha))


expected_clusters_exact = expected_clusters_infinite_exact(n, alpha)
expected_clusters_approx = expected_clusters_infinite_approx(n, alpha)

expected_clusters_finite_exact = [expected_clusters_finite_exact(m, n, alpha) for m in ms]

os.makedirs(outdir, exist_ok=True)

plt.figure(figsize=(5, 3.5))

for i, (l, m) in enumerate(zip(expected_clusters_finite_exact, ms)):
    plt.plot(l, label= f'Finite DP, M ={m}', color = cmap(1 - (i) / (len(ms) + 1)))

plt.semilogx(expected_clusters_exact, label='Infinite DP', color=cmap(0.0))

plt.legend()

plt.xlabel('Number of observed data points')
plt.ylabel('Expected number of clusters')
plt.title(f' Expected number of clusters for $\\alpha$ = {alpha}')

plt.savefig(f'{outdir}/expected_alpha_{int(alpha)}.pdf')

if args.show:
    plt.show()