import os

jobs_dir = 'jobs'
cluster_storage = '/home/clustor2/ma/h/hbz15/results'


base_job = ['#!/bin/bash',
            '#PBS -N dirichlet-bayes',
            '#PBS -m be',
            '#PBS -q standard',
            '',
            'cd ${HOME}/dirichlet-bayes',
            'python3 run_vec_finite.py ']

for i, M in enumerate([3,6,10,20,50,1e2,1e3,1e4,1e5,1e10]):
    job = base_job.copy()
    job[-1] = job[-1] + '-n 10000 -m {} -o {}'.format(M, cluster_storage)
    job[1] = job[1] + '_{}'.format(i)

    with open(os.path.join(jobs_dir, 'job_{}'.format(i)), 'w') as f:
        for line in job:
            f.write(line + '\n')