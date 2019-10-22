import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from util.distributions import normal_mixture_likelihood

# J - num samples
# SS - subsample iterations
# https://link.springer.com/content/pdf/10.1007%2Fs11634-018-0329-y.pdf


def plot_co_occurrence_matrix(
    assignment_matrix, file_dir="../results", filename="co_occurrence_matrix"
):
    """plot the co-occurrence matrix i.e. p(i,j) = prob i,j in same cluster 
    i, j being the ith and jth datapoints in the dataset

    Very slow! 
    T(n) = T(Nsamp*n) + O(1) complexity

    """
    Nsamp = assignment_matrix.shape[0]  # Number of samples taken
    nd = assignment_matrix.shape[1]  # Number of datapoints
    com = np.zeros([nd, nd])  # empty co-occurrence matrix

    for i in range(nd):
        for j in range(i + 1):
            for m in range(Nsamp):
                com[i, j] = com[i, j] + (
                    assignment_matrix[m, i] == assignment_matrix[m, j]
                )

    for i in range(nd):
        for j in range(i + 1):
            com[j, i] = com[i, j]

    com = com / Nsamp

    plt.figure()
    ax = sns.heatmap(com, square=False)
    ax.invert_yaxis()
    plt.title("Co-occurrence matrix i.e. p(i,j) = prob i,j in same cluster")
    plt.ylabel("Datapoints")
    plt.xlabel("Datapoints")
    plt.savefig("{}/{}.png".format(file_dir, filename))


def plot_cluster_size_hist(
    assignments, file_dir="../results", filename="cluster_size_hist"
):
    """

    """
    Nsamp = assignments.shape[0]
    nd = assignments.shape[1]

    num_clusters = np.array(list(map(lambda x: len(set(x)), assignments)))

    plt.figure()
    sns.distplot(
        num_clusters,
        bins=np.array(list(range(np.max(num_clusters) + 2))) - 0.5,
        kde=False,
        hist_kws={"ec": "k"},
    )
    plt.title("Histogram of the cluster occurrence")
    plt.xlabel("Cluster")
    plt.ylabel("Frequency")
    plt.savefig("{}/{}.png".format(file_dir, filename))


def plot_cluster_params(
    mu_chain, sigma_chain, weights_chain, file_dir="../results", filename="cluster_para"
):
    """Plots x-y graph with x being the parameters and y being the weights
    we plot it with the colour representing the number of clusters
    
    """

    mu_flat = np.concatenate(mu_chain)
    sigma_flat = np.concatenate(sigma_chain)
    weights_flat = np.concatenate(weights_chain)

    num_clusters = [len(mu_iter) for mu_iter in mu_chain]
    num_clusters = [np.ones(n) * n for n in num_clusters]
    num_clusters = np.concatenate(num_clusters)

    plt.figure()
    plt.scatter(
        mu_flat, weights_flat, c=num_clusters, s=10, marker=".", edgecolors=None
    )
    plt.colorbar()
    plt.title("weights again mu")
    plt.xlabel("mu paramter")
    plt.ylabel("weights")
    plt.savefig("{}/mu-{}.eps".format(file_dir, filename), format="eps")

    plt.figure()
    plt.scatter(
        sigma_flat, weights_flat, c=num_clusters, s=10, marker=".", edgecolors=None
    )
    plt.colorbar()
    plt.title("weights again sigma")
    plt.xlabel("sigma paramter")
    plt.ylabel("weights")
    plt.savefig("{}/sigma-{}.eps".format(file_dir, filename), format="eps")


def plot_posterior_predictive(
    mu_chain,
    sigma_chain,
    weights,
    assignments,
    file_dir="../results",
    filename="cluster_size_hist",
):
    """Plots the posterior predictive distribution density

    0. For each datapoint, we compute the average of all the paramters
    1. For each datapoint, we compute the probability density
    """
    density = np.zeros(assignments.shape[1])
    for i in range(assignments.shape[0]):
        # get the parameters and weights
        mu = mu_chain[i]
        sigma = sigma_chain[i]
        weight = weights[i]

        normal_mixture_likelihood()

