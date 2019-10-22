import pickle

import numpy as np

from src.plot import *

# Load data from file
assignments = np.load('results/assignments.npy')
mus, sigmas, weights = pickle.load(open('results/chain_iter.pkl', 'rb'))

# Thin the samples to reduce correlation
thin_factor = 10

assignments = assignments[0::thin_factor]
mus = mus[0::thin_factor]
sigmas = sigmas[0::thin_factor]
weights = sigmas[0::thin_factor]

Nsamp = assignments.shape[0]
nd = assignments.shape[1]

cluster_sizes = [np.unique(assignments[i, :], return_counts=True)[1] for i in range(Nsamp)]

num_clusters = np.array(list(map(lambda x: len(set(x)), assignments)))

plot_co_occurance_matirx(assignments)

plot_cluster_size_hist(assignments)

# plot_cluster_params(mus, weights)

plt.show()
