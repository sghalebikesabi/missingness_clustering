'''
This file creates/processes the data input for the algorithm
'''

import numpy as np
from scipy.stats import invgamma
from scipy.stats import invwishart

# set these parameters
n = 20
k = 6
m = 5


np.random.seed(1234)

# cluster
pC = np.random.uniform(size=k)
pC = pC/np.sum(pC)
C = np.random.choice(np.arange(k), n, list(pC))

# mask matrix
pi = np.random.uniform(size=(k,m))
pM = np.random.uniform(size=(n,m))
M = [pM[i,:] < pi[C[i],:] for i in range(n)]    

# covariates as mixture of multivariate normals
prior_mu = np.random.normal(np.random.normal(scale=5, size=m))
prior_sigma = np.random.uniform(size=(m*m))**2
mu = np.random.multivariate_normal(prior_mu, cov=np.identity(m), size=k)
sigma = invwishart.rvs(k+3, scale=prior_sigma, size=k, random_state=1234)

for l in range(k):


def simulate(n, m, k, pC, pi):
    C = np.random.choice(np.arange(k),n,pC)

    for i in range(n):
        for l in range(k):

