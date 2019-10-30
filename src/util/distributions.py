import numpy as np
from scipy.stats import norm

norm_C = np.sqrt(2 * np.pi)

def normal_mixture_likelihood(response, weights, mu, sigma):

    loglik = np.log(np.sum(weights * norm(loc=mu, scale=sigma).pdf(response)))

    return loglik


def normal_mixture_likelihood_vector(response, weights, mu, sigma):

    mu = np.atleast_2d(mu).T
    sigma = np.atleast_2d(sigma).T
    weights = np.atleast_2d(weights).T
    response = np.atleast_2d(response)

    z = (response - mu) / sigma

    pdf = np.exp(-z**2/2) / (norm_C * sigma)

    weighted_pdf = np.matmul(weights.T, pdf)

    loglik = np.log(weighted_pdf)

    return np.squeeze(loglik)
