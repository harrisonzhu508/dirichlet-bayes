import os
import pickle
import argparse

import numpy as np

from src.plot import *

parser = argparse.ArgumentParser()

parser.add_argument('-r', '--results_dir', required=True)
parser.add_argument('-b', '--burn_in', default=100, type=int)
parser.add_argument('-t', '--thinning', default=10, type=int)
parser.add_argument('-s', '--show', action='store_true')
parser.add_argument("-d", "--data", default="data/galaxy.txt", type=str)

args = parser.parse_args()
results_dir = args.results_dir 
# results_dir = "/Users/harrisonzhu/Documents/work/code/dirichlet-bayes/results/finite/galaxy_N_10000_M_3_alpha_1.000_m0_20.000_s0_10.000_a0_2.000_b0_0.111"
# Load data from file
data = np.loadtxt(args.data)
assignments = np.load(os.path.join(results_dir, 'assignments.npy'))
chain = pickle.load(open(os.path.join(results_dir, 'chain_iter.pkl'), 'rb'))
mus = chain["mu"]
sigmas =  chain["sigma"]
weights = chain["weights"]

# Thin the samples to reduce correlation
burn_in = args.burn_in
thin_factor = args.thinning
# burn_in = 1000
# thin_factor = 100

assignments = assignments[burn_in::thin_factor]
mus = mus[burn_in::thin_factor]
sigmas = sigmas[burn_in::thin_factor]
weights = weights[burn_in::thin_factor]

Nsamp = assignments.shape[0]
nd = assignments.shape[1]

print('Making graphs for {}'.format(args))
print(' using {} samples'.format(Nsamp))

cluster_sizes = [np.unique(assignments[i, :], return_counts=True)[1] for i in range(Nsamp)]

num_clusters = np.array(list(map(lambda x: len(set(x)), assignments)))

print(' plotting co-occurance matrix')
plot_co_occurrence_matrix(assignments, file_dir=results_dir)

print(' plotting cluster size histogram')
plot_cluster_size_hist(assignments, file_dir=results_dir)

print(' plotting cluster parameters')
plot_cluster_params(mus, sigmas, weights, file_dir=results_dir)

print(' plotting posterior predictive')
plot_posterior_predictive(
    data,
    mus,
    sigmas,
    weights,
    assignments,
    file_dir=results_dir
)

if args.show:
    plt.show()