'''
This file creates/processes the data input for the algorithm
run with:
    python src/data_input.py --n 100000 --m 150 --k 2
'''

import argparse
import numpy as np
from scipy.stats import invwishart
import pandas as pd
import pickle as pkl

def parse_args():
    '''
    Parses the arguments for simulating data.
    '''
    parser = argparse.ArgumentParser(description="Run node2vec.")
    
    parser.add_argument('--input', nargs='?', default='/home/ghalebik/Projects/missingness_clustering/data/', help='Input data frame path')
    
    parser.add_argument('--n', type=int, nargs='?', default=20, help='Number of observations')

    parser.add_argument('--m', type=int, nargs='?', default=5, help='Dimension of covariates')

    parser.add_argument('--k', type=int, nargs='?', default=3, help='Number of mixture components')

    parser.add_argument('--seed', type=int, nargs='?', default=1234, help='Random seed')

    parser.add_argument('--pC', nargs='?', default=0, help='Array of size k specifying the probability of each mixture component')

    parser.add_argument('--pi', nargs='?', default=0, help='Matrix of size k*m specifying the probability of a covariate being observed')

    parser.add_argument('--mu', nargs='?', default=0, help='List of k arrays of length m with means for each covariate of each mixture component')

    parser.add_argument('--sigma', nargs='?', default=0, help='List of k positive semidefinite matrices as covariance matrices for each mixture distribution')
    
    return parser.parse_args()


def simulate(n, m, k, seed, pC=0, pi=0, mu=0, sigma=0):
    ''' 
    This function simulates incomplete data from a multivariate normal mixture distribution. 
    Each subpopulation has its own missingness pattern.

    Arguments:
    n -- number of observations
    m -- number of covariates
    k -- number of mixture components
    seed -- random seed
    pC -- array of size k specifying the probability of each mixture component. If 0 (default), pC is  sampled randomly from a uniform distribution.
    pi -- matrix of size k*m specifying the probability of a covariate being observed for all covariates and all mixture components. If 0 (default), the elemnts are sampled from a uniform distribution.
    mu -- list of k arrays of length m with means for each covariate of each mixture component. If 0, drawn from a normal prior with scale 5.
    sigma -- list of k positive semidefinite matrices as covariance matrices for each mixture distribution. If 0, drawn from an inverse Wishart distribution with identity scale matrix.

    Returns:
    X -- a dataframe of incomplete data with additional attributes
    X.M -- mask matrix
    X.pC -- pC
    X.pi -- pi
    X.C -- C
    '''
    np.random.seed(seed)

    # cluster
    if pC==0:
        pC = np.random.uniform(size=k)
    pC = pC/np.sum(pC)
    C = np.random.choice(np.arange(k), n, list(pC))

    # mask matrix
    if pi==0:
        pi = np.random.uniform(size=(k,m))
    pM = np.random.uniform(size=(n,m))
    M = np.array([list(pM[i,:] < pi[C[i],:]) for i in range(n)], dtype=bool)

    # covariates as mixture of multivariate normals
    if mu==0:
        prior_mu = np.random.normal(np.random.normal(scale=5, size=m))
        mu = np.random.multivariate_normal(prior_mu, cov=np.identity(m), size=k)
    if sigma==0:
        prior_sigma = np.linalg.matrix_power(np.random.uniform(size=(m,m)),2)
        sigma = invwishart.rvs(m+3, scale=np.identity(m), size=k, random_state=seed)  # prior sigma is not positive definite

    # draw sample matrix
    X = [np.random.multivariate_normal(mu[C[i]], sigma[C[i]]) for i in range(n)]
    X = pd.DataFrame.from_records(X)
    
    # apply mask
    X = X.mask(False==M)

    return({'X': X, 'M': M, 'pC': pC, 'pi': pi, 'C': C})


def main(args):
    with open(args.input + 'X_n' + str(args.n) + '_m' + str(args.m) + '_k' + str(args.k) + '.simulated', 'wb') as file: 
        pkl.dump(simulate(n=args.n, m=args.m, k=args.k, seed=args.seed, pC=args.pC, pi=args.pi, mu=args.mu, sigma=args.sigma), file)

if __name__ == "__main__":
	args = parse_args()
	main(args)
