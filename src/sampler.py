import numpy as np
from numpy.random import normal

class FiniteDirichlet:
    """FiniteDirichlet class

    Args:
        
        - parameters: dictionary containing alpha, M
        - prior:
        - likelihood:
    """
    def __str__(self):
        print_str = "Finite Dirichlet Mixture model with self.parameters:"
        return print_str

    def __init__(self, parameters, prior, likelihood):

        self.parameters = parameters
        self.prior = prior
        self.likelihood = likelihood

    def run_chain(self):
        pass