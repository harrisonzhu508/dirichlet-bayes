import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from src.util.distributions import normal_mixture_likelihood
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

from matplotlib import cm
cmap = viridis = cm.get_cmap('viridis', 100)

# from src.util.distributions import normal_mixture_likelihood

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
    plt.title("weights against mu")
    plt.xlabel("mu parameter")
    plt.ylabel("weights")
    plt.savefig("{}/mu-{}.eps".format(file_dir, filename), format="eps")

    plt.figure()
    plt.scatter(
        sigma_flat, weights_flat, c=num_clusters, s=10, marker=".", edgecolors=None
    )
    plt.colorbar()
    plt.title("weights against sigma")
    plt.xlabel("sigma parameter")
    plt.ylabel("weights")
    plt.savefig("{}/sigma-{}.eps".format(file_dir, filename), format="eps")


def plot_posterior_predictive(
    data,
    mu_chain,
    sigma_chain,
    weights,
    assignments,
    file_dir="../results",
    filename="posterior_predictive",
):
    """Plots the posterior predictive distribution density

    0. For each datapoint, we compute the average of all the parameters
    1. For each datapoint, we compute the probability density

    TODO: this is super slow!!
    """
    num_points = 400
    x = np.linspace(0, 40, num_points)
    num_clust_in_chain = np.array(list(map(lambda row: len(row)-1, mu_chain)))
    active_cluster_nums = np.max(num_clust_in_chain) - np.min(num_clust_in_chain) + 1
    density = np.zeros([active_cluster_nums, num_points])

    # we also perform thinning
    for i in range(assignments.shape[0]):
        if i % 100 == 0:
            print(i)
        # get the parameters and weights
        ind = num_clust_in_chain[i]-  np.min(num_clust_in_chain)
        apply_row = lambda x_elt,: normal_mixture_likelihood(
            x_elt, weights[i], mu_chain[i], sigma_chain[i]
        )
        density[ind, :] = density[ind, :] + np.exp(list(map(apply_row, x)))

    counts = np.unique(num_clust_in_chain, return_counts=True)[1]
    cumulative_density = density / np.atleast_2d(counts).T
    post_density = np.sum(density, axis=0) / assignments.shape[0]
    
    plt.figure()

    for i, cluster in enumerate(range(np.min(num_clust_in_chain), np.max(num_clust_in_chain) + 1)):
        plt.plot(x, cumulative_density[i, :], label='Density given K={}'.format(cluster))

    plt.plot(x, post_density, label='Total density', color='k')

    plt.hist(
        data,
        bins=np.linspace(0, 40, num_points/4),
        density=1,
        color='w',
        edgecolor='black', 
        linewidth=0.8
    )
    
    plt.xlabel("Data point")
    plt.ylabel("Posterior density")
    plt.title("Posterior Predictive Plot")
    plt.legend()
    plt.savefig("{}/{}.eps".format(file_dir, filename), format="eps")

if __name__ == "__main__":
    pass