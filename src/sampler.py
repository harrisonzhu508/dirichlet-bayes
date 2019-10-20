import numpy as np
from numpy.random import normal, gamma, dirichlet
from scipy.stats import norm
import logging

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
        z_chain = np.zeros((params["num_samples"] + 1, self.n), dtype=int)

        # place initial parameters values into chain
        # take crude averages and standard deviation!
        # TODO: this bit is tricky because the number of clusters if varying...
        # we might get stackoverflow as the dimensions of cluster parameters grow
        # this could be resolved through caching throughout the MCMC procedure
        mu_chain.append(np.array([np.mean(data)]))
        sigma_chain.append(np.array([np.std(data)]))
        weights.append([np.array(self.params["alpha"])])  # initially we have 1 cluster!

        self.chain = {
            "mu": mu_chain,
            "sigma": sigma_chain,
            "assignments": z_chain,
            "weights": weights,
        }

    def run_chain(self):
        """Run a Gibbs sampler

        """
        for i in range(1, self.params["num_samples"] + 1):
            logging.info("MCMC Chain: {}".format(i))
            # find the number of points in each clusters
            unique, counts = np.unique(
                self.chain["assignments"][i, :], return_counts=True
            )
            num_pts_clusters = dict(zip(unique, counts))
            num_clusters = max(self.chain["assignments"][i, :]) + 1

            # initialise the arrays of the chain as the array lengths differ
            # as we increase the number of clusters
            self.chain["mu"].append(np.zeros(num_clusters))
            self.chain["sigma"].append(np.zeros(num_clusters))

            for k in range(num_clusters):
                logging.info("MCMC Chain: {}, Cluster loop: {}".format(i, k))
                num_pts_cluster = num_pts_clusters[k]
                data_cluster = self.data[
                    np.where(self.chain["assignments"][i - 1, :] == k)[0]
                ]
                if num_pts_cluster > 0:
                    # now sample mu[k] given mu[-k], sigma and the partition
                    # see lecture 15 last slide calculations N(a, b) trick
                    sigma_tmp = self.chain["sigma"][i - 1][k]
                    b = np.sqrt(
                        1
                        / (
                            num_pts_cluster / sigma_tmp ** 2
                            + 1 / self.hyperparam["sigma_0"] ** 2
                        )
                    )
                    a = b ** 2 * (
                        sum(data_cluster) / sigma_tmp ** 2
                        + self.hyperparam["mu_0"] / self.hyperparam["sigma_0"] ** 2
                    )
                    # update mu
                    self.chain["mu"][i][k] = normal(loc=a, scale=b)

                    # now sample sigma[k] given sigma[-k], mu and the partition
                    c = self.hyperparam["alpha_0"] + num_pts_cluster / 2
                    d = self.hyperparam["beta_0"] + 0.5 * sum(
                        (data_cluster - self.chain["mu"][i - 1][k])
                    )

                    # update sigma
                    self.chain["sigma"][i][k] = 1 / np.sqrt(gamma(shape=c, scale=d))

            self.chain["assignments"][i, :] = self.chain["assignments"][i - 1, :]
            # now, loop through all the datapoints to compute the new cluster probabilities
            for j in range(self.n):
                logging.info("MCMC Chain: {}, Dataset index: {}".format(i, j))
                num_pts_cluster_tmp = self.chain["assignments"].copy()
                cluster_assigned = self.chain["assignments"][i - 1, j].copy()
                num_pts_cluster_tmp[cluster_assigned] = num_pts_clusters[
                    cluster_assigned
                ]

                # probability for each existing k cluster -> gives a vector of probabilities
                p_old_cluster = norm(self.chain["mu"][i], self.chain["sigma"][i]).pdf(
                    self.data[j]
                )
                mu_new = normal(
                    loc=self.hyperparam["mu_0"], scale=self.hyperparam["sigma_0"]
                )
                sigma_new = 1 / np.sqrt(
                    gamma(
                        shape=self.hyperparam["alpha_0"],
                        scale=self.hyperparam["beta_0"],
                    )
                )
                p_new_cluster = self.params["alpha"] * norm(mu_new, sigma_new).pdf(
                    self.data[j]
                )
                p_new_cluster = np.array([p_new_cluster])
                # normlise the probabilities
                prob_clusters = np.concatenate((p_new_cluster, p_old_cluster))
                prob_clusters = prob_clusters / sum(prob_clusters)

                # select a new cluster!
                # if we get 0 then new cluster!
                cluster_pick = np.random.choice(
                    [num for num in range(num_clusters + 1)], p=prob_clusters
                )
                if cluster_pick == 0:
                    num_clusters = num_clusters + 1
                    self.chain["assignments"][i, j] = num_clusters - 1
                    self.chain["mu"][i] = np.concatenate(
                        (self.chain["mu"][i], np.array([mu_new]))
                    )
                    self.chain["sigma"][i] = np.concatenate(
                        (self.chain["sigma"][i], np.array([sigma_new]))
                    )
                    # obtain the number of members in the cluster belonging to the ith element, with
                    # it removed!
                    # find the number of points in each clusters as it will change with each iteration
                    unique, counts = np.unique(
                        self.chain["assignments"][i, :], return_counts=True
                    )
                    num_pts_clusters = dict(zip(unique, counts))

                else:
                    self.chain["assignments"][i, j] = cluster_pick

            # now, sample the cluster weights!
            weights_new = dirichlet(
                alpha=self.params["alpha"] + np.array(list(num_pts_clusters.values()))
            )
            self.chain["weights"].append(weights_new)

        print("Complete sampling")

        return self.chain


def test():
    pass


if __name__ == "__main__":
    test()
