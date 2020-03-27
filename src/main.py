'''
This file computes the latent features according to the method presented in report
'''
import argparse
import numpy as np
import pickle as pkl
import warnings

from functions import EM_clustering

def parse_args():
    '''
    Parses the arguments for the latent feature extraction.
    '''
    parser = argparse.ArgumentParser(description="Run node2vec.")
    
    parser.add_argument('--input', nargs='?', default='/home/ghalebik/Projects/missingness_clustering/data/X.simulated', help='Input data frame path')
    
    parser.add_argument('--k', type=int, nargs='?', default=2, help='Number of missigness clusters')

    parser.add_argument('--tol', type=float, nargs='?', default=10**(-6), help='Tolerance for EM algorithm')
    
    return parser.parse_args()


def main(args):
    '''
    Pipeline for the presented latent feature model
    '''
    '''
    class test:
        def __init__(self, **kwds):
            self.__dict__.update(kwds)

    args = test
    args.input = '/home/ghalebik/Projects/missingness_clustering/data/X_n10000_m5_k2.simulated'
    args.k = 3
    args.tol = 10**(-6)
    '''

    # load data
    with open(args.input, 'rb') as file: 
        X = pkl.load(file)
    
    # determine if data is simulated
    simulated = False
    if args.input.split('.')[-1] == 'simulated':
        simulated = True

    # run EM algorithm
    pi, pC, gamma, C, niter = EM_clustering(X, args.k, args.tol, simulated)
    
    # train VAEs
    Z = clustered_VAE(X, C)

if __name__ == "__main__":
	args = parse_args()
	main(args)
    #main()