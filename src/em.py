'''
This file contains all the functions for the missigness clustering embedding approach.
'''
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


def EM_clustering(X, k, tol, simulated=False, pC=0, pi=0, seed=3434):

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
        return(0, 0, 0, np.zeros(n), M, 0)

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
                permutations = list(itertools.permutations(range(3)))
                loss.append(np.mean(C_true != [np.argmax(tr_gamma[i]) for i in range(n)]))

            if np.linalg.norm(pi[-1]-pi[-2]) + np.linalg.norm(pC[-1]-pC[-2]) < tol:
                    break

        C = [np.argmax(tr_gamma[i]) for i in range(n)]

        if simulated:
            plt.plot(loss)
            plt.show()
            if loss[-1] > 0.5:
                C = [1-c for c in C]

        return(pi[-1], pC[-1], gamma, C, M, niter)

    else:

        # E step
        numerator = [[np.log(pC[l]) + np.sum(np.log(pi[l]**M[i,:]*(1-pi[l])**(1-M[i,:]))) for l in range(k)] for i in range(n)]
        # denominator = [np.log(np.sum(np.exp(numerator[i]))) for i in range(n)]
        tr_gamma = [[1/(1+np.sum([np.exp(not_l-numerator[i][l]) for not_l in numerator[i] if not_l!=numerator[i][l]])) for l in range(k)] for i in range(n)]
        # tr_gamma = [numerator[i]/denominator[i] for i in range(n)]

        C = [np.argmax(tr_gamma[i]) for i in range(n)]
        gamma = list(map(list, zip(*tr_gamma))) # transpose gamma

        if np.mean(C_true != [np.argmax(tr_gamma[i]) for i in range(n)]) > 0.5:
            C = [1-c for c in C]

        return(gamma, C, M)

