import os
import pickle
import argparse

import numpy as np

from src.sampler_vectorised_finite import FiniteNormalDirichlet

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--data", default="data/galaxy.txt", type=str)
parser.add_argument("-o", "--outdir", default="results", type=str)

parser.add_argument("-n", "--num_samples", default=10000, type=int)
parser.add_argument("-m", "--M", default=1e6, type=int)
parser.add_argument("-a", "--alpha", default=1.0, type=float)

parser.add_argument("-m0", "--mu_0", default=20.0, type=float)
parser.add_argument("-s0", "--sigma_0", default=10, type=float)
parser.add_argument("-a0", "--alpha_0", default=2, type=float)
parser.add_argument("-b0", "--beta_0", default=1.0 / 9.0, type=float)

parser.add_argument("-s", "--seed", default=0, type=int)

args = parser.parse_args()

parameters = {"alpha": args.alpha, "M": args.M}

hyperparameters = {
    "mu_0": args.mu_0,
    "sigma_0": args.sigma_0,
    "alpha_0": args.alpha_0,
    "beta_0": args.beta_0,
}

np.random.seed(args.seed)

data = np.loadtxt("data/galaxy.txt")
finiteDirichlet = FiniteNormalDirichlet(parameters, hyperparameters, data)
chain, assignments = finiteDirichlet.run_chain(args.num_samples)

# file_name = f"{os.path.splitext(os.path.basename(args.data))[0]}_N_{args.num_samples}_M_{args.M}_alpha_{args.alpha:0.3f}_m0_{args.mu_0:0.3f}_s0_{args.sigma_0:0.3f}_a0_{args.alpha_0:0.3f}_b0_{args.beta_0:0.3f}"

file_name = "{}_N_{}_M_{}_alpha_{:0.3f}_m0_{:0.3f}_s0_{:0.3f}_a0_{:0.3f}_b0_{:0.3f}".format(
    os.path.splitext(os.path.basename(args.data))[0],
    args.num_samples,
    args.M,
    args.alpha,
    args.mu_0,
    args.sigma_0,
    args.alpha_0,
    args.beta_0,
)

os.makedirs(os.path.join(args.outdir, file_name), exist_ok=True)

with open(os.path.join(args.outdir, file_name, "chain_iter.pkl"), "wb") as f:
    pickle.dump(chain, f)

np.save(os.path.join(args.outdir, file_name, "assignments.npy"), assignments)
