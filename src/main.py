'''
This file computes the latent features according to the method presented in report
'''
import argparse
import numpy as np
import torch
import pandas as pd
import pickle as pkl
import warnings

from em import EM_clustering
from vae import clustered_VAE, train, test


def parse_args():
    '''
    Parses the arguments for the latent feature extraction.
    '''
    parser = argparse.ArgumentParser(description="Run missingness clustering based latent factor model.")
    
    parser.add_argument('--input', nargs='?', 
                        default='/home/ghalebik/Projects/missingness_clustering/data/X_n10000_m5_k2.simulated', 
                        help='Input data frame path')
    
    parser.add_argument('--k', type=int, nargs='?', default=2, help='Number of missigness clusters')

    parser.add_argument('--tol', type=float, nargs='?', default=10**(-3), help='Tolerance for EM algorithm')

    parser.add_argument('--distinct_hdim', type=list, nargs='?', default=[[400],[400]], help='Distinct encoder layers of VAE')
    parser.add_argument('--commonencoder_hdim', type=list, nargs='?', default=[[20,20]], help='Common encoder layers of VAE')
    parser.add_argument('--decoder_hdim', type=list, nargs='?', default=[400], help='Decoder layers of VAE')

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


def main(args):
    '''
    Pipeline for the presented latent feature model
    '''

    # determine if data is simulated
    simulated = False
    if args.input.split('.')[-1] == 'simulated':
        simulated = True

    # load data
    with open(args.input, 'rb') as file: 
        X = pkl.load(file)
    if simulated == True:
        data = X['X'].sample(frac=1)
    else:
        data = X.sample(frac=1)
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    # save old indices to preserve order of original order
    old_indices = data.index

    # run EM algorithm
    pi, pC, gamma, C, niter = EM_clustering(X, args.k, args.tol, simulated)

    # mean imputation within clusters
    data2 = pd.DataFrame(columns=data.columns)
    for l in range(args.k):
        data2 = data2.append(data.loc[np.where([c==l for c in C])].apply(lambda x: x.fillna(x.mean()), axis=0))

    data = data2.iloc[old_indices]


    # create VAE model
    device = torch.device("cuda" if args.cuda else "cpu")

    train_loader = torch.utils.data.DataLoader(torch.tensor(data.iloc[:int(args.train_percentages*len(data))].values), 
                        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(torch.tensor(data.iloc[int(args.train_percentages*len(data)):].values),
                        batch_size=args.batch_size, shuffle=True, **kwargs)

    C_train = C[:int(args.train_percentages*len(data))]
    C_test = C[int(args.train_percentages*len(data)):]
    
    model = clustered_VAE(input_dim=data.shape[1], k=args.k, distinct_hdim=args.distinct_hdim, 
                            commonencoder_hdim=args.commonencoder_hdim, decoder_hdim=args.decoder_hdim).to(device)

    # train and test VAE
    model = train(model, C_train, train_loader, C_test, test_loader, device, args)

if __name__ == "__main__":
    args = parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    main(args)
