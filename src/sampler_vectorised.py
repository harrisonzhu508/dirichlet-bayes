import numpy as np
from numpy.random import normal, gamma, dirichlet
from scipy.stats import norm
import logging
import pickle
import json
from os import path

logging.basicConfig(filename="logs/sampler_test.log", level=logging.DEBUG)


class InfiniteNormalDirichlet:
    """FiniteDirichlet class
    Args:

        - params: dictionary containing the DP model parameters
        - prior_init: 
        - data: our dataset
    """

    def __str__(self):
        print_str = "Finite Dirichlet Mixture model with {} and initial parameters {}".format(
            self.params, self.hyperparam
        )
        return print_str

    def __init__(self, params, hyperparam, data):

        self.params = params
        self.hyperparam = hyperparam
        self.n = data.shape[0]
        self.data = data

        # create storage space of size O(num_samples) for chain
        mu_chain = []
        sigma_chain = []
        weights = []
        ## some extra storage for assignment at each time step
        ## we index from 0! So with K clusters we have 0...K-1
        ## this has storage O(num_datapoints * num_samples)

        # place initial parameters values into chain
        # take crude averages and standard deviation!
        mu_chain.append([np.mean(data)])
        sigma_chain.append([np.std(data)])
        weights.append([1])  # initially we have 1 cluster!

        self.chain = {"mu": mu_chain, "sigma": sigma_chain, "weights": weights}

    def run_chain(self, steps=1):
        """Run a Gibbs sampler
        """
        z_chain = np.zeros((steps + 1, self.n), dtype=int)
        self.assignments = z_chain

        for i in range(1, steps):
            if i % 50 == 0: print("MCMC Chain: {}".format(i))
            # find the number of points in each clusters
            old_unique, old_counts = np.unique(
                self.assignments[i - 1, :], return_counts=True)
            num_pts_clusters = dict(zip(old_unique, old_counts))
            # old_unique = list(old_unique)

            # initialise the arrays of the chain as the array lengths differ
            # as we increase the number of clusters
            mu_new = np.zeros(old_unique.shape)
            sigma_new = np.zeros(old_unique.shape)
            weights_new = np.zeros(old_unique.shape)

            mu_old = self.chain["mu"][i - 1]
            sigma_old = self.chain["sigma"][i - 1]
            weights_old = self.chain["weights"][i - 1]

            for k in old_unique:
                num_pts_cluster = num_pts_clusters[k]
                data_cluster = self.data[np.where(
                    self.assignments[i - 1, :] == k)[0]]
                if num_pts_cluster > 0:
                    # now sample mu[k] given mu[-k], sigma and the partition
                    # see lecture 15 last slide calculations N(a, b) trick
                    sigma_tmp = sigma_old[k]
                    b = np.sqrt(
                        1
                        / (
                            num_pts_cluster / sigma_tmp ** 2
                            + 1 / self.hyperparam["sigma_0"] ** 2
                        )
                    )
                    a = b ** 2 * (
                        sum(data_cluster) / sigma_tmp ** 2
                        + self.hyperparam["mu_0"] /
                        self.hyperparam["sigma_0"] ** 2
                    )
                    # update mu
                    mu_new[k] = normal(loc=a, scale=b)

                    # now sample sigma[k] given sigma[-k], mu and the partition
                    c = self.hyperparam["alpha_0"] + num_pts_cluster / 2
                    d = self.hyperparam["beta_0"] + 0.5 * sum(
                        (data_cluster - mu_new[k]) ** 2
                    )

                    # update sigma
                    sigma_new[k] = 1. / np.sqrt(gamma(shape=c, scale=1 / d))

            self.assignments[i, :] = self.assignments[i - 1, :].copy()
            # now, loop through all the datapoints to compute the new cluster probabilities
            for j in range(self.n):

                # TODO: this bit could definitely be taken out in the future, but i
                # will just leave it for now
                old_unique, old_counts = np.unique(
                    self.assignments[i, :], return_counts=True)

                old_cluster_assigned = int(self.assignments[i, j])
                old_counts[old_cluster_assigned] -= 1

                K = len(old_unique)

                # probability for each existing k cluster -> gives a vector of probabilities
                p_old_cluster = old_counts * norm(
                    mu_new, sigma_new
                ).pdf(self.data[j])

                mu_update = normal(
                    loc=self.hyperparam["mu_0"], scale=self.hyperparam["sigma_0"]
                )
                sigma_update = 1 / np.sqrt(
                    gamma(
                        shape=self.hyperparam["alpha_0"],
                        scale=1 / self.hyperparam["beta_0"],
                    )
                )
                p_new_cluster = self.params["alpha"] * norm(
                    mu_update, sigma_update
                ).pdf(self.data[j])
                # logging.debug(p_old_cluster)
                # logging.debug(p_new_cluster)
                p_new_cluster = np.atleast_1d(p_new_cluster)
                # normlise the probabilities
                prob_clusters = np.concatenate((p_old_cluster, p_new_cluster))
                prob_clusters = prob_clusters / sum(prob_clusters)
                # select a new cluster!
                # if we get K then new cluster!
                # try:
                cluster_pick = np.random.choice(
                    K+1, p=prob_clusters)
                # except Exception as e:
                #     logging.info("Iteration {}, datapoint {}".format(i, j))
                #     logging.info("{}".format(sigma_new),
                #                  "nan probabilities occurring due to values "
                #                  "mu_new and sigma_new: \n\n"
                #                  "mu_new: {} \n sigma_new: {}".format(
                #                      mu_new, sigma_new),
                #                  "\n with probabilities: {}".format(prob_clusters))

                if cluster_pick == K:
                    self.assignments[i, j] = cluster_pick
                    # update the indices and shift the parameters up the list
                    mu_new = np.append(mu_new, mu_update)
                    sigma_new = np.append(sigma_new, sigma_update)

                else:
                    self.assignments[i, j] = cluster_pick

                # obtain the number of members in the cluster belonging to the ith element, with
                # it removed!
                # find the number of points in each clusters as it will change with each iteration
                # unique, counts = np.unique(
                #     self.assignments[i, :], return_counts=True)
                # num_pts_clusters = dict(zip(unique, counts))

                # remove empty clusters and their parameters
                # now, sample the cluster weights!
                if (old_counts[old_cluster_assigned] == 0) & (cluster_pick != old_cluster_assigned):
                    mu_new = np.delete(mu_new, old_cluster_assigned)
                    sigma_new = np.delete(sigma_new, old_cluster_assigned)

                    ind = self.assignments[i, :] > (old_cluster_assigned)
                    self.assignments[i, ind] = self.assignments[i, ind] - 1

            # update the weights
            weights_update = dirichlet(
                alpha=self.params["alpha"]
                + np.array(list(num_pts_clusters.values()))
            )
            weights_new = weights_update.copy()

            self.chain["mu"].append(mu_new)
            self.chain["sigma"].append(sigma_new)
            self.chain["weights"].append(weights_new)

        print("Complete sampling")

        return self.chain, self.assignments


if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.path.abspath("."))

    import numpy as np
    from scipy.stats import norm
    from config.config import INFINITE_NORMAL_PARAMS, INFINITE_NORMAL_HYPERPARAMETER

    np.random.seed(0)

    data = np.loadtxt("data/galaxy.txt")
    finiteDirichlet = InfiniteNormalDirichlet(
        INFINITE_NORMAL_PARAMS, INFINITE_NORMAL_HYPERPARAMETER, data
    )
    finiteDirichlet.run_chain(10000)
