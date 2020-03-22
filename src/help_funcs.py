'''
This file contains help functions for testing
'''

class test:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

args = test
args.input = '/home/ghalebik/Projects/missingness_clustering/data/X.simulated'
args.k = 3
args.tol = 10**6


args.input='/home/ghalebik/Projects/missingness_clustering/data/'##='Input data frame path')
    
args.n=20#='Number of observations')

args.m=5#='Dimension of covariates')

args.k=3#='Number of mixture components')

args.seed=1234#='Random seed')

args.pC=0#='Array of size k specifying the probability of each mixture component')

args.pi=0#='Matrix of size k*m specifying the probability of a covariate being observed')

args.mu=0#='List of k arrays of length m with means for each covariate of each mixture component')

args.sigma=0#='List of k positive semidefinite matrices as covariance matrices for each mixture distribution')
