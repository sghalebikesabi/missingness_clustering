'''
This file contains all the functions for the missigness clustering embedding approach.
'''
import itertools
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


def EM_clustering(X, k, tol, simulated=False, pC=0, pi=0, test=False, seed=3434):

    # model parameters
    if not simulated:
        M = np.array(X.notnull())
    else: 
        M = np.array(X['M'])
        C_true = X['C']
        X = X['X']

    n, m = M.shape
    np.random.seed(seed)

    # k=1
    if k == 1:
        if not test:
            return(0, 0, 0, np.zeros(n), M, 0)
        else:
            return(0, np.zeros(n), M)

    # initialization
    if isinstance(pC, int):
        pC_unnorm = list(np.random.uniform(size=k))
        pC = [pC_unnorm/np.sum(pC_unnorm)]
        pi = [np.random.uniform(size=(k,m))]
        loss = []

        niter = 0
        while True:
            niter += 1

            # E step
            numerator = [[np.log(pC[-1][l]) + np.sum(np.log(pi[-1][l]**M[i,:]*(1-pi[-1][l])**(1-M[i,:]))) for l in range(k)] for i in range(n)]
            tr_gamma = [[1/(1+np.sum([np.exp(not_l-numerator[i][l]) for not_l in numerator[i] if not_l!=numerator[i][l]])) for l in range(k)] for i in range(n)]
            gamma = list(map(list, zip(*tr_gamma))) # transpose gamma

            # M step
            sum_gamma_l = [np.sum(gamma[l]) for l in range(k)]
            sum_gamma = np.sum(sum_gamma_l)
            pC.append(np.array([sum_gamma_l[l]/sum_gamma for l in range(k)]))

            sum_gamma_l_j = [[np.sum(M[:,j]*gamma[l]) for j in range(m)] for l in range(k)]
            pi.append(np.array([sum_gamma_l_j[l]/sum_gamma_l[l] for l in range(k)]))

            if simulated:
                loss.append(np.mean(C_true != [np.argmax(tr_gamma[i]) for i in range(n)]))

            if np.linalg.norm(pi[-1]-pi[-2]) + np.linalg.norm(pC[-1]-pC[-2]) < tol:
                    break

        C = [np.argmax(tr_gamma[i]) for i in range(n)]

        if simulated:
            permutations = list(itertools.permutations(range(k)))
            losses = []
            for perm in permutations:
                losses.append(np.mean([perm[c] for c in C_true] != [np.argmax(tr_gamma[i]) for i in range(n)]))
            loss.append(np.min(losses))
            current_permutation = permutations[np.argmin(losses)]

            pi.append([list(pi[-1][current_permutation[i]]) for i in range(k)])
            pC.append([pC[-1][current_permutation[i]] for i in range(k)])

        return(loss[-1], np.array(pi[-1]), np.array(pC[-1]), gamma, C, M, niter)

    else:

        # E step
        numerator = [[np.log(pC[l]) + np.sum(np.log(pi[l]**M[i,:]*(1-pi[l])**(1-M[i,:]))) for l in range(k)] for i in range(n)]
        tr_gamma = [[1/(1+np.sum([np.exp(not_l-numerator[i][l]) for not_l in numerator[i] if not_l!=numerator[i][l]])) for l in range(k)] for i in range(n)]

        C = [np.argmax(tr_gamma[i]) for i in range(n)]
        gamma = list(map(list, zip(*tr_gamma))) # transpose gamma

        loss = np.mean(C_true != [np.argmax(tr_gamma[i]) for i in range(n)])

        return(loss, gamma, C, M)

