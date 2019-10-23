import numpy as np
from scipy.stats import norm


def normal_mixture_likelihood(response, weights, mu, sigma):

    loglik = np.log(np.sum(weights * norm(loc=mu, scale=sigma).pdf(response)))

    return loglik

