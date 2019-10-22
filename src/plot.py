import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# J - num samples
# SS - subsample iterations


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


def plot_cluster_params(mu_chain, weights_chain, file_dir="../results", filename="cluster_para"):
    """Plots x-y graph with x being the parameters and y being the weights
    we plot it with the colour representing the number of clusters
    
    """

    mu_flat = np.concatenate(mu_chain).ravel()
    weights_flat = np.concatenate(weights_chain)

    num_clusters = [len(mu) for mu in mus]
    num_clusters = [np.ones(n) * n for n in num_clusters]

    sns.scatterplot(vec_mus, vec_weights, hue=num_clusters)
    plt.savefig("{}/{}.png".format(file_dir, filename))


def plot_posterior_predictive(
    mus, sigmas, weights, file_dir="../results", filename="cluster_size_hist"
):
    pass
