from pathlib import Path
from os import path
ROOT = Path(__file__).resolve().parents[1]
out_dir = path.join(ROOT, "results")

INFINITE_NORMAL_PARAMS = {
    "num_samples": 10,
    "alpha": 1,
    "sample_freq": 10,
    "out_dir": out_dir
}

INFINITE_NORMAL_HYPERPARAMETER = {
        "mu_0": 20,
        "sigma_0": 10,
        "alpha_0": 2,
        "beta_0": 1/9
    }
