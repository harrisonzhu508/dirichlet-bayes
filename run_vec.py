import os
import pickle
import argparse

import numpy as np

from src.sampler_vectorised import InfiniteNormalDirichlet

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--data', default='data/galaxy.txt', type=str)
parser.add_argument('-o', '--outdir', default='results', type=str)

parser.add_argument('-n', '--num_samples', default=1000, type=int)
parser.add_argument('-a', '--alpha', default=1., type=float)

parser.add_argument('-m0', '--mu_0', default=20., type=float)
parser.add_argument('-s0', '--sigma_0', default=10, type=float)
parser.add_argument('-a0', '--alpha_0', default=2, type=float)
parser.add_argument('-b0', '--beta_0', default=1./9., type=float)

args = parser.parse_args()

parameters = {
    "alpha": args.alpha,
}

hyperparameters = {
        "mu_0": args.mu_0,
        "sigma_0": args.sigma_0,
        "alpha_0": args.alpha_0,
        "beta_0": args.beta_0
    }

data = np.loadtxt("data/galaxy.txt")
finiteDirichlet = InfiniteNormalDirichlet(parameters, hyperparameters, data)
chain, assignments = finiteDirichlet.run_chain(args.num_samples)

file_name = f'{os.path.splitext(os.path.basename(args.data))[0]}_N_{args.num_samples}_alpha_{args.alpha:0.3f}_m0_{args.mu_0:0.3f}_s0_{args.sigma_0:0.3f}_a0_{args.alpha_0:0.3f}_b0_{args.beta_0:0.3f}'

os.makedirs(os.path.join(args.outdir, file_name), exist_ok=True)

with open(os.path.join(args.outdir, file_name, "chain_iter.pkl"), "wb") as f:
    pickle.dump(chain, f)

np.save(os.path.join(args.outdir, file_name, "assignments.npy"), assignments)