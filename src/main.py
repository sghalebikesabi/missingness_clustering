'''
This file computes the latent features according to the method presented in report
'''
import argparse
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle as pkl
from sklearn import preprocessing
import torch
import warnings

from em import EM_clustering
from vae import clustered_VAE, train, test, missingness_dataset


def parse_args():
    '''
    Parses the arguments for the latent feature extraction.
    '''
    parser = argparse.ArgumentParser(description="Run missingness clustering based latent factor model.")
    
    parser.add_argument('--input', nargs='?', default='X_n10000_m5_k2.simulated', help='Input data frame path')
    
    parser.add_argument('--goal', nargs='?', default='embedding', help='Is the goal to embed the information available in a latent space (embedding) or to test the imputation (imputation)?')
    
    parser.add_argument('--k', type=int, nargs='?', default=1, help='Number of missigness clusters')

    parser.add_argument('--tol', type=float, nargs='?', default=10**(-3), help='Tolerance for EM algorithm')

    parser.add_argument('--distinct_hdim', type=int, nargs='+', default=[[75], [75]], help='Distinct encoder layers of VAE')
    parser.add_argument('--commonencoder_hdim', nargs='+', default=[[10,10]], help='Common encoder layers of VAE')
    parser.add_argument('--decoder_hdim', nargs='+', default=[75], help='Decoder layers of VAE')

    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='Input batch size for training (default: 128)')

    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='Number of epochs to train (default: 10)')

    parser.add_argument('--no-cuda', action='store_true', default=True, help='Enables CUDA training')

    parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed (default: 1)')

    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='How many batches to wait before logging training status')
    
    parser.add_argument('--train-percentages', type=float, default=0.8, metavar='N',
                        help='Percentage of data set that represents test data set')

    parser.add_argument('--images', type=bool, default=False, metavar='N',
                        help='True if input is image data')

    parser.add_argument('--images-dim', type=list, default=[28, 28], metavar='N',
                        help='If input is image data, dimension of image data')

    return parser.parse_args()


def imputation_loss(recon_x, x, mask):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, x.shape[1]), reduction='sum', weight=mask.float())
    return(BCE)    


def show_image(x, shape=[28,28]):
    plt.imshow(np.array(x).reshape(shape), cmap='gray')
    plt.show()


def main(args):
    '''
    Pipeline for the presented latent feature model
    '''
    
    # determine if data is simulated
    simulated = 'simulated' in args.input

    # load data
    args.input = 'MNIST_k2_clustered_uniform.simulated'
    with open(os.getcwd() + '/data/' + args.input, 'rb') as file: 
        X = pkl.load(file)
    
    # resample order
    if simulated:
        data = X['X'].sample(frac=1)
    else:
        data = pd.DataFrame(X).sample(frac=1)
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    # train test split data
    data_train = data.iloc[:int(args.train_percentages*len(data))]
    data_test = data.iloc[int(args.train_percentages*len(data)):]

    # save old indices to preserve order of original order
    old_indices_train = data_train.index
    old_indices_test = data_test.index

    # run EM algorithm
    pi, pC, gamma, C_train, M_train, niter = EM_clustering(data_train, args.k, args.tol, simulated)
    pi, pC, gamma, C_test, M_test, niter = EM_clustering(data_test, args.k, args.tol, simulated, pi=pi, pC=pC)

    # mean imputation within clusters
    data2_train = pd.DataFrame(columns=data_train.columns)
    for l in range(args.k):
        data2_train = data2_train.append(data_train.iloc[np.where(C_train==l)].apply(lambda x: x.fillna(x.mean()), axis=0))
    data_train = data2_train.loc[old_indices_train]

    data2_test = pd.DataFrame(columns=data_test.columns)
    for l in range(args.k):
        data2_test = data2_test.append(data_test.iloc[np.where(C_test==l)].apply(lambda x: x.fillna(x.mean()), axis=0))
    data_test = data2_test.loc[old_indices_test]

    # normalize values
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit(data_train)
    data_train = torch.tensor(min_max_scaler.transform(data_train))
    data_test = torch.tensor(min_max_scaler.transform(data_test))

    # transform data into DataLoader
    train_loader = torch.utils.data.DataLoader(missingness_dataset(data_train, C_train, M_train), 
                        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(missingness_dataset(data_test, C_test, M_test),
                        batch_size=args.batch_size, shuffle=True, **kwargs)

    # create VAE model
    device = torch.device("cuda" if args.cuda else "cpu")
   
    model = clustered_VAE(input_dim=data_train.shape[1], k=args.k, distinct_hdim=args.distinct_hdim, 
                            commonencoder_hdim=args.commonencoder_hdim, decoder_hdim=args.decoder_hdim).to(device)

    # train and test VAE
    model = train(model, train_loader, test_loader, device, args)

    # reconstruction loss
    print('The masked reconstruction loss is ', masked_reconstruction_loss(recon_x, x, mask))

if __name__ == "__main__":
    args = parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    main(args)
