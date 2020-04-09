'''
This file builds a VAE model which allows for different encoder structures for different clusters
This code makes use of the repo https://github.com/pytorch/examples/tree/master/vae
'''

import numpy as np
import os
import torch
import torch.utils.data
from torch import optim, nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


class missingness_dataset(torch.utils.data.Dataset):
    """Missingness dataset class."""

    def __init__(self, data, C, M):
        """
        Args:
            data: Imputed dataset.
            C: Onedimensional tensor of length equal to length of data .
            missingness: Tensor of same shape as data.
        """
        self.data = data
        self.C = C
        self.M = M

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data[idx], self.C[idx], self.M[idx]

        return sample


class clustered_VAE(nn.Module):
    def __init__(self, input_dim, k=2, distinct_hdim=[[400],[400]], commonencoder_hdim=[[20,20]], decoder_hdim=[400]):
        super(clustered_VAE, self).__init__()
        
        # create attributes needed for the encoder
        self.n_distinctlayers = [len(distinct_hdim_iter) for distinct_hdim_iter in distinct_hdim]
        self.cum_distinctlayers = [np.sum(self.n_distinctlayers[:(i+1)]) for i in range(len(self.n_distinctlayers))]
        self.distinct_hdim = distinct_hdim
        self.n_encoderlayers = self.cum_distinctlayers[-1] + len(commonencoder_hdim)
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
def masked_loss_function(recon_x, x, mu, logvar, mask):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, x.shape[1]), reduction='sum', weight=mask.float())

    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(model, train_loader, test_loader, device, output_path, args):
    torch.manual_seed(args.seed)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0
        for batch_idx, (data_iter, batched_C_train, M) in enumerate(train_loader):
            # load data
            data_iter = data_iter.to(device)
            optimizer.zero_grad()
            # sort input obserations according to clusters
            argsort_batched_C_train = np.argsort(batched_C_train)
            batched_C_train = torch.tensor([batched_C_train[argsort_batched_C_train[i]] for i in range(len(argsort_batched_C_train))])
            data_iter = data_iter[argsort_batched_C_train]
            data_C = {'x': data_iter, 'C': batched_C_train}
            # train model
            recon_batch, mu, logvar = model(data_C)
            if args.goal != 'imputation':
                loss = masked_loss_function(recon_batch.float(), data_iter.float(), mu.float(), logvar.float(), M)
            else:
                loss = masked_loss_function(recon_batch.float(), data_iter.float(), mu.float(), logvar.float(), M == False)
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
        test(model, epoch, test_loader, device, output_path, args)

        if args.images == True:
            with torch.no_grad():
                sample = torch.randn(64, 20).to(device)
                sample = model.decode(sample).cpu()
                save_image(sample.view(64, 1, args.images_dim[0], args.images_dim[1]), output_path + '/sample_' + str(epoch) + '.png')

    return(model)


def test(model, epoch, test_loader, device, output_path, args):
    model.eval() 
    test_loss = 0
    with torch.no_grad():
        for i, (data_iter, batched_C_test, missingness) in enumerate(test_loader):
            # load data 
            if i==0:
                old_len_data = len(data_iter)
            data_iter = data_iter.to(device)
            # sort data observations according to cluster
            argsort_batched_C_test = np.argsort(batched_C_test)
            batched_C_test = torch.tensor([batched_C_test[argsort_batched_C_test[i]] for i in range(len(argsort_batched_C_test))])
            data_iter = data_iter[argsort_batched_C_test]
            data_C = {'x': data_iter, 'C': batched_C_test}
            recon_batch, mu, logvar = model(data_C)
            test_loss += masked_loss_function(recon_batch.float(), data_iter.float(), mu.float(), logvar.float(), missingness).item()
            if args.images == True:
                if i == 0:
                    n = min(data_iter.size(0), 8)
                    comparison = torch.cat([data_iter[:n].view(n, 1, args.images_dim[0], args.images_dim[1]),
                                        recon_batch.view(args.batch_size, 1, args.images_dim[0], args.images_dim[1])[:n]])
                    save_image(comparison.cpu(), output_path + '/reconstruction_' + str(epoch) + '.png', nrow=n)
            old_len_data = len(data_iter)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
