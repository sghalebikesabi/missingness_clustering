'''
This file builds a VAE model which allows for different encoder structures for different clusters
This code makes use of the repo https://github.com/pytorch/examples/tree/master/vae
'''

import numpy as np
import torch
import torch.utils.data
from torch import optim, nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


class clustered_VAE(nn.Module):
    def __init__(self, input_dim, k=2, distinct_hdim=[[400],[400]], commonencoder_hdim=[[20,20]], decoder_hdim=[400]):
        super(clustered_VAE, self).__init__()
        
        # create attributes needed for the encoder
        self.n_distinctlayers = [len(distinct_hdim[i]) for i in range(len(distinct_hdim))]
        self.cum_distinctlayers = [np.sum(self.n_distinctlayers[:(i+1)]) for i in range(len(self.n_distinctlayers))]
        self.distinct_hdim = distinct_hdim
        self.n_encoderlayers = np.sum(self.n_distinctlayers) + len(commonencoder_hdim)
        self.k = k
        self.input_dim = input_dim
        self.layers = nn.ModuleList()
        
        # add layers distinctive to each cluster
        for l in range(self.k):
            current_dim = input_dim
            for hdim in distinct_hdim[l]:
                self.layers.append(nn.Linear(current_dim, hdim))
                current_dim = hdim

        # add layers common to all clusters
        for hdim in (commonencoder_hdim + decoder_hdim):
            if isinstance(hdim, int):
                self.layers.append(nn.Linear(current_dim, hdim))
                current_dim = hdim
            else:
                for hhdim in hdim:
                    self.layers.append(nn.Linear(current_dim, hhdim))
                current_dim = hhdim    
                    
        # add output layer
        self.layers.append(nn.Linear(current_dim, input_dim))

    def encode(self, x, C):
        # create list of list of tensors for saving results of each distinct layer in a list 
        y = [[x[C==l,:].float()] for l in range(self.k) if len(x[C==l,:])>0]
        y_C = [l for l in range(self.k) if len(x[C==l,:])>0]
        z = []
        for i, m in enumerate(self.layers):
            # distinct encoder layers
            if ((i//(self.cum_distinctlayers[-1]-1)) < self.k): 
                # determine which cluster layer belongs to
                masked_layers = [i < self.cum_distinctlayers[j] for j in range(len(self.cum_distinctlayers))]
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

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        for i, l in enumerate(self.layers):
            if i > self.n_encoderlayers:
                if i==(len(self.layers)-1):
                    z = torch.sigmoid(self.layers[i](z))
                else:
                    z = F.relu(self.layers[i](z))

        return z

    def forward(self, x):
        mu, logvar = self.encode(x['x'].view(-1, self.input_dim), x['C'])
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, x.shape[1]), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(model, C_train, train_loader, C_test, test_loader, device, args):
    torch.manual_seed(args.seed)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0
        for batch_idx, (data_iter) in enumerate(train_loader):
            # load data
            data_iter = data_iter.to(device)
            batched_C_train = C_train[(batch_idx*args.batch_size):(batch_idx*args.batch_size+len(data_iter))]
            optimizer.zero_grad()
            # sort input obserations according to clusters
            argsort_batched_C_train = np.argsort(batched_C_train)
            batched_C_train = torch.tensor([batched_C_train[argsort_batched_C_train[i]] for i in range(len(argsort_batched_C_train))])
            data_iter = data_iter[argsort_batched_C_train]
            data_C = {'x': data_iter, 'C': batched_C_train}
            # train model
            recon_batch, mu, logvar = model(data_C)
            loss = loss_function(recon_batch.float(), data_iter.float(), mu.float(), logvar.float())
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data_iter), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item() / len(data_iter)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_loader.dataset)))
        test(model, epoch, C_test, test_loader, device, args)
        if args.images == True:
            with torch.no_grad():
                sample = torch.randn(64, 20).to(device)
                sample = model.decode(sample).cpu()
                save_image(sample.view(64, 1, args.images_dim[0], args.image_dim[1]), 'results/sample_' + str(epoch) + '.png')

    return(model)


def test(model, epoch, C_test, test_loader, device, args):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data) in enumerate(test_loader): # change to (data, _) for labelled data
            # load data 
            if i==0:
                old_len_data = len(data)
            data = data.to(device)
            # sort data observations according to cluster
            batched_C_test = C_test[(i*old_len_data):(i*old_len_data+len(data))]
            argsort_batched_C_test = np.argsort(batched_C_test)
            batched_C_test = torch.tensor([batched_C_test[argsort_batched_C_test[i]] for i in range(len(argsort_batched_C_test))])
            data = data[argsort_batched_C_test]
            data_C = {'x': data, 'C': batched_C_test}
            recon_batch, mu, logvar = model(data_C)
            test_loss += loss_function(recon_batch.float(), data.float(), mu.float(), logvar.float()).item()
            if args.images == True:
                if i == 0:
                    n = min(data.size(0), 8)
                    comparison = torch.cat([data[:n],
                                        recon_batch.view(args.batch_size, 1, args.images_dim[0], args.images_dim[1])[:n]])
                    save_image(comparison.cpu(),
                            'results/reconstruction_' + str(epoch) + '.png', nrow=n)
            old_len_data = len(data)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
