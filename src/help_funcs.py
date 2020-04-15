'''
This file contains help functions for testing
'''

model.encode(data_C['x'][0].view(-1, model.input_dim), torch.tensor([1]))#data_C['C'][0].reshape((1)))

for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

from torch import autograd
with autograd.detect_anomaly():
    model(data_C)

#args.input = '/home/ghalebik/Data/MNIST/t10k-images-idx3-ubyte'


class test:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)


args.input = 'MNIST_k10_clustered_notuniform.simulate' #MNIST_k10_clustered_notuniform.simulate
args.output = '/home/ghalebik/Projects/missingness_clustering/data/'
args.missingness_pattern = 'clustered'


args = test
args.cuda = False
args.no_cuda = True
args.k = 2
args.tol = 10**(-3)
args.train_percentages = 0.8
args.batch_size = 128
args.epochs = 10
args.log_interval = 10
args.seed=1234#='Random seed')

args.distinct_hdim=[[400], [400]]
args.commonencoder_hdim=[[20,20]]
args.decoder_hdim=[400]

args.goal = 'embedding'
args.images = True
args.images_dim = [28, 28]

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
self.input_dim = 150
input_dim = 150


import inspect
lines = inspect.getsource(F.binary_cross_entropy)
print(lines)


x = torch.randn(3, 4)
mask = x.ge(0.5)
x.masked_fill_(mask,0)


====> Epoch: 10 Average loss: 1.5664
====> Test set loss: 1.5507


def encode(x, C):
    # create list of list of tensors for saving results of each distinct layer in a list 
    y = [[x[[c==l for c in C]].float()] for l in range(self.k) if len(x[[c==l for c in C]])>0]
    y_C = [l for l in range(self.k) if len(x[[c==l for c in C]])>0]
    z = []
    for i, m in enumerate(self.layers):
        # distinct encoder layers
        if i < self.cum_distinctlayers[-1]:
        #if ((i//(self.cum_distinctlayers[-1]-1)) < self.k): 
            # determine which cluster layer belongs to
            masked_layers = [i < cum_distinctlayers_iter for cum_distinctlayers_iter in self.cum_distinctlayers]
            l = np.min(np.nonzero(masked_layers))
            if l in y_C:
                l_index = y_C.index(l)
                y[l_index].append(F.relu(self.layers[i](y[l_index][-1])))
            if l==(self.k-1):
                y_com = torch.cat([y_l[-1] for y_l in y])
        # common encoder layers
        elif i < (self.n_encoderlayers-1):
            y_com = F.relu(self.layers[i](y_com))
        # last encoder layer
        elif (i == self.n_encoderlayers-1) or (i == self.n_encoderlayers):
            z.append(self.layers[i](y_com))
    return z

    def reparameterize(mu, logvar):

        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std


    def decode(z):
        for i, l in enumerate(self.layers):
            if i > self.n_encoderlayers:
                if i==(len(self.layers)-1):
                    z = torch.sigmoid(self.layers[i](z))
                else:
                    z = F.relu(self.layers[i](z))

        return z



            if missingness == 'MAR' or missingness=='MNAR':
                # compute mask
                M = np.array([(pM[i,:] < pi[C[i],:]) for i in range(n)], dtype=bool)
                # MNAR
                if missingness == 'MAR':
                    M = np.array([[False]*m if (data[i,x1] < med_x1) and (data[i,x2] > med_x2) else M[i]==False for i in range(n)], dtype=bool) == False
                else:
                    M = np.array([
                        [M[i,j]==False if j not in [x1, x2] else True for j in range(m)] if (data[i,x1] < med_x1) and (data[i,x2] > med_x2) 
                        else M[i]==False for i in range(n)
                        ], dtype=bool) == False
                avr_completeness = np.mean(M)
            else:
                avr_completeness = np.sum(pC * np.mean(pi, axis=1))


for batch_idx, (data_iter, batched_C_train, missingness) in enumerate(train_loader):
