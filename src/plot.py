import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# J - num samples
# SS - subsample iterations

def plot_co_occurance_matirx(assignment_matrix):
    Nsamp = assignment_matrix.shape[0] # Number of samples taken
    nd = assignment_matrix.shape[1] # Number of datapoints
    com = np.zeros([nd, nd]) # empty co-occurance matrix

    for i in range(nd):
        for j in range(i+1):
            for m in range(Nsamp):
                com[i,j] = com[i,j] + (assignment_matrix[m, i] == assignment_matrix[m, j])

    for i in range(nd):
        for j in range(i+1):
            com[j,i] = com[i,j]

    com = com / Nsamp
    
    plt.figure()
    ax = sns.heatmap(com, square=True)
    ax.invert_yaxis()

def plot_cluster_size_hist(assignments):
    Nsamp = assignments.shape[0]
    nd = assignments.shape[1]

    num_clusters = np.array(list(map(lambda x: len(set(x)), assignments)))

    plt.figure()
    sns.distplot(num_clusters, bins = np.array(list(range(np.max(num_clusters)+2))) - 0.5, kde=False, hist_kws={'ec': 'k'})

def plot_cluster_params(mus, weights):
    print(mus)

    vec_mus = np.concatenate(mus)
    vec_weights = np.concatenate(weights)

    num_clusters = [len(mu) for mu in mus]
    num_clusters = [np.ones(n) * n for n in num_clusters]
    vec_num_clusters = np.concatenate(num_clusters)

    sns.scatterplot(vec_mus, vec_weights, hue=num_clusters)

def plot_posterior_predictive(mus, sigmas, weights):
    pass