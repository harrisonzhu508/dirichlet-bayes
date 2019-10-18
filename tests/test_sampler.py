import unittest
import numpy as np
from config.config import INFINITE_NORMAL_PARAMS, INFINITE_NORMAL_PRIOR
from src.sampler import InfiniteNormalDirichlet

data = np.loadtxt("data/galaxy.txt")
finiteDirichlet = InfiniteNormalDirichlet(INFINITE_NORMAL_PARAMS, INFINITE_NORMAL_PRIOR, data)


class test_sampler(unittest.TestCase):

    def test_init(self):
        self.assertEqual(np.ndarray, type(finiteDirichlet.data))




