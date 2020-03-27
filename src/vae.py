from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--k', type=int, default=2, metavar='N',
                    help='number of clusters')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

n_train = len(train_loader)*args.batch_size
n_test = len(test_loader)*args.batch_size
pC = np.random.uniform(size=args.k)
pC = pC/np.sum(pC)
C_train = torch.from_numpy(np.random.choice(np.arange(args.k), n_train, list(pC)))
C_test = torch.from_numpy(np.random.choice(np.arange(args.k), n_test, list(pC)))

class VAE(nn.Module):
    def __init__(self, input_dim, k=2, distinct_hdim=[[400],[400]], commonencoder_hdim=[[20,20]], decoder_hdim=[400]):
        super(VAE, self).__init__()
        
        # create attributes needed for the encoder
        self.n_distinctlayers = [len(distinct_hdim[i]) for i in range(len(distinct_hdim))]
        self.cum_distinctlayers = [np.sum(self.n_distinctlayers[:(i+1)]) for i in range(len(self.n_distinctlayers))]
        self.distinct_hdim = distinct_hdim
        self.n_encoderlayers = np.sum(self.n_distinctlayers) + len(commonencoder_hdim)
        self.k = k
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
        y = [[x[C==l,:]] for l in range(self.k) if len(x[C==l,:])>0]
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
        mu, logvar = self.encode(x['x'].view(-1, 784), x['C'])
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

model = VAE(input_dim=784,k=args.k).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        # load data
        data = data.to(device)
        batched_C_train = C_train[(batch_idx*args.batch_size):(batch_idx*args.batch_size+len(data))]
        optimizer.zero_grad()
        # sort input obserations according to clusters
        argsort_batched_C_train = np.argsort(batched_C_train)
        batched_C_train = batched_C_train[argsort_batched_C_train]
        data = data[argsort_batched_C_train]
        data_C = {'x': data, 'C': batched_C_train}
        # train model
        recon_batch, mu, logvar = model(data_C)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            # load data 
            if i==0:
                old_len_data = len(data)
            data = data.to(device)
            # sort data observations according to cluster
            batched_C_test = C_test[(i*old_len_data):(i*old_len_data+len(data))]
            argsort_batched_C_test = np.argsort(batched_C_test)
            batched_C_test = batched_C_test[argsort_batched_C_test]
            data = data[argsort_batched_C_test]
            data_C = {'x': data, 'C': batched_C_test}
            recon_batch, mu, logvar = model(data_C)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)
            old_len_data = len(data)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')
