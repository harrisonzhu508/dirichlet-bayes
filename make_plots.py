import pickle

import numpy as np

from src.plot import *

assignments = np.load('results/assignments.npy')
mus, sigmas, weights = pickle.load(open('results/chain_iter.pkl', 'rb'))

Nsamp = assignments.shape[0]
nd = assignments.shape[1]

cluster_sizes = [np.unique(assignments[i, :], return_counts=True)[1] for i in range(Nsamp)]

num_clusters = np.array(list(map(lambda x: len(set(x)), assignments)))

plot_co_occurance_matirx(assignments)

plot_cluster_size_hist(assignments)

# plot_cluster_params(mus, weights)

plt.show()
