from src.sampler import InfiniteNormalDirichlet
import numpy as np
from scipy.stats import norm
from config.config import INFINITE_NORMAL_PARAMS, INFINITE_NORMAL_HYPERPARAMETER

data = np.loadtxt("data/galaxy.txt")
finiteDirichlet = InfiniteNormalDirichlet(INFINITE_NORMAL_PARAMS, INFINITE_NORMAL_HYPERPARAMETER, data)
finiteDirichlet.run_chain()
