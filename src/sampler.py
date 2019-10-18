import numpy as np
from numpy.random import normal
from config.config import INFINITE_NORMAL_PARAMS


class InfiniteNormalDirichlet:
    """FiniteDirichlet class

    Args:
        
        - params: dictionary containing the DP model parameters
        - prior_init: 
        - likelihood:
        - data: our dataset
    """

    def __str__(self):
        print_str = "Finite Dirichlet Mixture model with {} and initial parameters {}".format(
            self.params, self.prior_init
        )
        return print_str

    def __init__(self, params, prior_init, data):

        self.params = params
        self.prior_init = prior_init
        self.n = data.shape[0]

        # create storage space of size O(num_samples) for chain
        mu_chain = np.zeros(params["num_samples"] + 1)
        sigma_chain = np.zeros(params["num_samples"] + 1)
        ## some extra storage for assignment at each time step
        ## we index from 0! So with K clusters we have 0...K-1
        ## this has storage O(num_datapoints * num_samples)
        z_chain = np.zeros((self.n, params["num_samples"] + 1))

        # place initial parameters values into chain
        # take crude averages and standard deviation!
        mu_chain[0] = np.mean(data)
        sigma_chain[0] = np.std(data)

        self.chain = {"mu": mu_chain, "sigma": sigma_chain, "assignments": z_chain}

    def run_chain(self):
        """Run a Gibbs sampler

        """
        for i in self.params["num_samples"]:

            # find the number of points in each clusters
            unique, counts = np.unique(
                self.chain["assignments"][i, :], return_counts=True
            )
            num_pts_cluster = dict(zip(unique, counts))
            num_clusters = max(self.chain["assignments"])

            




def test():
    pass


if __name__ == "__main__":
    test()
