'''
This file contains all the functions for the missigness clustering embedding approach.
'''
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


def EM_clustering(X, k, tol, simulated=False, seed=3434):

    # model parameters
    if simulated==False:
        M = np.array(X.notnull())
    else: 
        M = np.array(X['M'])
        C_true = X['C']
        X = X['X']

    n, m = M.shape
    np.random.seed(seed)


    # initialization
    pC_unnorm = list(np.random.uniform(size=k))
    pC = [pC_unnorm/np.sum(pC_unnorm)]
    pi = [np.random.uniform(size=(k,m))]
    loss = []

    niter = 0
    while True:
        niter += 1

        # E step
        numerator = [[pC[-1][l] * np.prod((pi[-1][l]**M[i,:]*(1-pi[-1][l])**(1-M[i,:]))) for l in range(k)] for i in range(n)]
        denominator = [np.sum(numerator[i]) for i in range(n)]
        tr_gamma = [numerator[i]/denominator[i] for i in range(n)]
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

    plt.plot(loss)
    plt.show()

    return(pi[-1], pC[-1], gamma, C, niter)

