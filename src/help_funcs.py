'''
This file contains help functions for testing
'''


class test:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

args = test
args.cuda = False
args.no_cuda = True
args.input = '/home/ghalebik/Projects/missingness_clustering/data/X_n10000_m5_k2.simulated'
args.k = 2
args.tol = 10**6
args.train_percentages = 0.8
args.batch_size = 128
args.epochs = 10
args.log_interval = 10

args.distinct_hdim=[[400],[400]]
args.commonencoder_hdim=[[20,20]]
args.decoder_hdim=[400]

#args.input='/home/ghalebik/Projects/missingness_clustering/data/'##='Input data frame path')
    
args.n=20#='Number of observations')

args.m=5#='Dimension of covariates')

args.seed=1234#='Random seed')

args.pC=0#='Array of size k specifying the probability of each mixture component')

args.pi=0#='Matrix of size k*m specifying the probability of a covariate being observed')

args.mu=0#='List of k arrays of length m with means for each covariate of each mixture component')

args.sigma=0#='List of k positive semidefinite matrices as covariance matrices for each mixture distribution')



class self:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)
self.k = args.k
