import os

jobs_dir = 'jobs'

base_job = ['#!/bin/bash',
            '#PBS -N dirichlet-bayes',
            '#PBS -m be',
            '#PBS -q standard',
            '',
            'cd ${HOME}/dirichlet-bayes',
            'python run_vec_finite.py ']

for i, M in enumerate([3,6,10,20,50,1e2,1e3,1e4,1e5,1e10]):
    job = base_job.copy()
    job[-1] = job[-1] + f'-n 10000 -m {int(M)}'
    job[1] = job[1] + f'_{i}'

    with open(os.path.join(jobs_dir, f'job_{i}'), 'w') as f:
        for line in job:
            f.write(line + '\n')