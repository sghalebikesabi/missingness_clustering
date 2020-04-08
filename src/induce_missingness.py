'''
This file induces missingness into the data input:
    python src/data_input.py --n 100000 --m 150 --k 2
'''

import argparse
import gzip
import numpy as np
import os
import random
from scipy.stats import invwishart
from sklearn import preprocessing
import pandas as pd
import pickle as pkl


def parse_args():
    '''
    Parses the arguments for simulating data.
    '''
    parser = argparse.ArgumentParser(description="Run node2vec.")
    
    parser.add_argument('--input', nargs='?', default='mnist', help='Dataset input frame path')
    
    parser.add_argument('--k', type=int, nargs='?', default=3, help='Number of mixture components')

    parser.add_argument('--seed', type=int, nargs='?', default=1234, help='Random seed')

    parser.add_argument('--n', type=int, nargs='?', default=20, help='Number of observations')

    parser.add_argument('--m', type=int, nargs='?', default=5, help='Dimension of covariates')

    parser.add_argument('--pC', nargs='?', default=0, help='Array of size k specifying the probability of each mixture component')

    parser.add_argument('--pi', nargs='?', default=0, help='Matrix of size k*m specifying the probability of a covariate being observed')
    
    return parser.parse_args()


def read_in_mnist():
    '''adapted from https://stackoverflow.com/questions/40427435/extract-images-from-idx3-ubyte-file-or-gzip-via-python'''
    f = gzip.open('/home/ghalebik/Data/MNIST/train-images-idx3-ubyte.gz','rb')
    image_size = 28
    buf = f.read()#60000*image_size*image_size)
    data = np.frombuffer(buf, dtype=np.uint8, offset=16).astype(np.float32)
    data = data.reshape(-1, image_size * image_size)
    return data


def simulate_data(k, n, m, pC=0, mu=0, sigma=0):
    np.random.seed(args.seed)

    # cluster
    if pC==0:
        pC = np.random.uniform(size=k)
    pC = pC/np.sum(pC)
    C = np.random.choice(np.arange(k), n, list(pC))

    # covariates as mixture of multivariate normals
    if mu==0:
        prior_mu = np.random.normal(np.random.normal(scale=5, size=m))
        mu = np.random.multivariate_normal(prior_mu, cov=np.identity(m), size=k)
    if sigma==0:
        prior_sigma = np.linalg.matrix_power(np.random.uniform(size=(m,m)),2)
        sigma = invwishart.rvs(m+3, scale=np.identity(m), size=k, random_state=args.seed)  # prior sigma is not positive definite

    # draw sample matrix
    if k > 1:
        X = [np.random.multivariate_normal(mu[C[i]], sigma[C[i]]) for i in range(n)]
    else:
        X = [np.random.multivariate_normal(mu[C[i]], sigma) for i in range(n)]
    X = pd.DataFrame.from_records(X)

    return({'X': X, 'C': C})


def simulate_missingness(data, missingness='MNAR', overall_completeness=0.8, uniform=True, k=1, pC=0, pi=0, C=0):
    ''' 
    This function simulates missingness for the data. 
    Each subpopulation has its own missingness pattern.

    Arguments:
    data -- data
    pC -- array of size k specifying the probability of each mixture component. If 0 (default), pC is  sampled randomly from a uniform distribution.
    pi -- matrix of size k*m specifying the probability of a covariate being observed for all covariates and all mixture components. If 0 (default), the elemnts are sampled from a uniform distribution.
    uniform -- if True, probability for all features to be missing is the same; else probability for the features is different

    Returns:
    X -- a dataframe of incomplete data with additional attributes
    X.M -- mask matrix
    X.pC -- pC
    X.pi -- pi
    X.C -- C
    '''
    np.random.seed(args.seed)

    n, m = data.shape

    # cluster
    if missingness == 'MCAR': 
        pC = np.zeros(k)
        pC[0] = 1
        C = np.zeros(n, int)
    elif missingness == 'clustered':
        if isinstance(C, int):
            pC = np.random.dirichlet([1]*k)
            C = np.random.choice(np.arange(k), n, p = list(pC))
            
    else:
        while True:
            x1, x2 = np.random.choice(range(m), 2)
            med_x1 = np.median(data[:,x1])
            med_x2 = np.median(data[:,x2])
            if np.mean(((data[:,x1] < med_x1) + (data[:,x2] > med_x2))>0) > 0.1 and np.mean(((data[:,x1] < med_x1) + (data[:,x2] > med_x2))>0) < 0.9:
                break

        C = np.array(((data[:,x1] < med_x1) + (data[:,x2] > med_x2)) > 0 , dtype = int)
        k = 2
        pC = np.array([np.mean(C), 1-np.mean(C)])

    # random prob of features being observed
    if missingness == 'MNAR':
        desired_completeness = overall_completeness + 0.25 * 0.25 / m
    else:
        desired_completeness = overall_completeness
    

    if pi==0:
        if not uniform:
            while True:
                spread = max(0.95 - desired_completeness, 0.3)
                pi = np.random.uniform(low = desired_completeness - spread, high = desired_completeness + spread, size=(k,m))
                if missingness == 'MAR':
                    pi[:,x1] = 1
                    pi[:,x2] = 1
                pi /= np.sum(pC*np.mean(pi,axis=1))
                pi *= desired_completeness
                pi_larger_1 = np.where(pi > 1)
                if np.sum(pi > 1) > 0:           
                    pi[pi_larger_1] = np.random.uniform(low=0.65, high=0.95, size=(len(pi_larger_1[0]))) 
                else: 
                    break
        else:
            while True:
                pi = np.random.uniform(low=0.65, high=0.95, size=(k))
                pi /= np.sum(pC*pi)
                pi *= desired_completeness
                pi = np.array([[i] * m for i in pi])
                if missingness == 'MAR':
                    pi[:,x1] = 1
                    pi[:,x2] = 1
                if np.sum(pi > 1) == 0: 
                    break

    # adjust for overall_completeness rate
    if overall_completeness != 0:
        while True:
            if missingness == 'MNAR':
                avr_completeness = 0.25 * np.sum(pC*np.mean([[pi[l][j] if j not in [x1,x2] else 1 for j in range(m)] for l in range(k)], axis=1)) + 0.75 * np.sum(pC*np.mean(pi, axis=1))
            else:
                avr_completeness = np.sum(pC*np.mean(pi, axis=1))
            if ((avr_completeness - overall_completeness)**2 < 0.01):
                break
            else:
                pi = pi / avr_completeness * desired_completeness
                pi = np.array([[min(j,1) for j in i] for i in pi])
                pi = np.array([[max(j,0) for j in i] for i in pi])

    # mask matrix
    pM = np.random.uniform(size=(n,m))

    # compute mask
    M = np.array([(pM[i,:] < pi[C[i],:]) for i in range(n)], dtype=bool)

    # MAR and MNAR
    if missingness == 'MNAR':
        M = np.array([
            [M[i,j]==False if j not in [x1, x2] else True for j in range(m)] if (data[i,x1] < med_x1) or (data[i,x2] > med_x2) 
            else M[i]==False for i in range(n)
            ], dtype=bool) == False

    # apply mask
    X = pd.DataFrame(data).mask(False==M)

    return({'X': X, 'M': M, 'pC': pC, 'pi': pi, 'C': C})


def main(args):
    for k in [1,2,5]:
        for uniform in [True, False]:
            for missingness in ['MCAR', 'MAR', 'MNAR', 'clustered']:
                if 'MNIST' in args.input:
                    data = read_in_mnist()
                    missing_data = simulate_missingness(data, missingness=missingness, k=k, uniform=uniform)
                    with open(os.getcwd() + '/data/' + 'MNIST' + '_k' + str(k) + '_' + missingness + '_' + 
                                'not' * (1-uniform) + 'uniform' + '.simulated', 'wb') as file: 
                        pkl.dump({'incomplete': missing_data['X'], 'complete': data, 'C': missing_data['C']}, file)

                elif args.input == 'simulate':                        
                        for n, m in [[100, 5], [100, 150], [1000, 150], [10000, 150]]:
                            if k == 1 and uniform == True and missingness != 'clustered':
                                break
                            data = simulate_data(k, n, m)
                            try:
                                missing_data = simulate_missingness(np.array(data['X']), C=data['C'], missingness=missingness, k=k, uniform=uniform)
                            except:
                                raise Exception("Error: " + 'X_n' + str(n) + '_m' + str(m) + '_k' + str(k) + '_' +
                                        missingness + '_' + 'not' * (1-uniform) + 'uniform' + '.simulated') 
                            with open(os.getcwd() + '/data/' + 'X_n' + str(n) + '_m' + str(m) + '_k' + str(k) + '_' +
                                        missingness + '_' + 'not' * (1-uniform) + 'uniform' + '.simulated', 'wb') as file: 
                                pkl.dump({'incomplete': missing_data['X'], 'complete': data['X'], 'C': missing_data['C']}, file)


if __name__ == "__main__":
	args = parse_args()
	main(args)
