'''
# EEG Classification - PyTorch

        updated: Sep. 01, 2018

        Data: https://www.physionet.org/pn4/eegmmidb/

        # 1. Data Downloads

        Warning: Executing these blocks will automatically create directories and download datasets.
'''

# System
import requests
import re
import os
from io import StringIO
import shutil
import pathlib
import urllib

from collections import defaultdict
from sklearn.preprocessing import scale, StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

# Essential Data Handling
import numpy as np
import pandas as pd
from math import ceil, floor

# Get Paths
from glob import glob

# EEG package
from mne import pick_types, events_from_annotations
from mne.io import read_raw_edf

import pickle
import sys
import json

from datetime import datetime

# PyThorch 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset

# To use Tensorflow with PyTorch
from torchbearer import Trial
from torchbearer.callbacks import TensorBoard, Best, ReduceLROnPlateau, CSVLogger, ModelCheckpoint, StepLR

# To parse input arguments
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='PyTorch CNN+LSTM network for EEGs signals Analysis')

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser.add_argument('run_number', 
                    type=int,
                    help='Run number')
parser.add_argument('--test_rate', 
                    type=float, 
                    default=0.2, 
                    help='test rate (default 0.2) to split dataset in train (0.8) - test (0.2)')
parser.add_argument('--data_type', 
                    type=str, 
                    help='data_type (Imaged or Real)')
parser.add_argument('--batch_size', 
                    type=int, 
                    default=64, 
                    help='input batch size for training (default: 128)')
parser.add_argument('--dropout', 
                    type=str2bool,
                    nargs='?',
                    help='using or not using dropout in FC layer (default: True)')
parser.add_argument('--epochs', 
                    type=int, 
                    help='number of epochs to train (more than 300)')
parser.add_argument('--lr', 
                    type=float, 
                    default=0.01, 
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr_stepsize', 
                    type=int, 
                    default=10, 
                    help='learning rate step size for decay (default: 10)')
parser.add_argument('--momentum', 
                    type=float, 
                    default=0.5, 
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no_cuda', 
                    action='store_true', 
                    default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', 
                    type=int, 
                    help='random seed (try: 42)')
parser.add_argument('--log_interval', 
                    type=int, 
                    default=10, 
                    help='how many batches to wait before logging training status')
parser.add_argument('--n_nodes', 
                    type=int, 
                    default=[3,2,2], 
                    help='list of 3 elements: Number of nodes/layers [CNN, LSTM, FC] respectively')
parser.add_argument('--cnn_filter', 
                    type=int, 
                    default=32, 
                    help='Number of filter for the first layer, other layers goes as power of two')
parser.add_argument('--cnn_kernelsize', 
                    type=int, 
                    default=3, 
                    help='dimension of kernel for CNN layer')
parser.add_argument('--lstm_hidden', 
                    type=int, 
                    default=64, 
                    help='how many batches to wait before logging training status')
parser.add_argument('--fc_dim', 
                    type=int, 
                    default=1024, 
                    help='number of neurons for fully connected layer')
parser.add_argument('--fc_dropout', 
                    type=int, 
                    default=0.5, 
                    help='drop out rate for fully connected layer')
parser.add_argument('--rcnn_output', 
                    type=int, 
                    default=5, 
                    help='output of network')
parser.add_argument('--num_classes', 
                    type=int, 
                    default=5, 
                    help='PhysioNet total classes')
parser.add_argument('--split_type', 
                    type=str, 
                    help='split type to split the dataset (user_dependent or user_independent)')


args = parser.parse_args()

print('args.dropout =', args.dropout)

'''
# Network Parameters
    class Args:
        def __init__(self):
            self.test_rate  = 0.2
            self.run_number = 0
            self.cuda       = False
            self.no_cuda    = True
            self.seed       = 42
            self.data_type  = 'Imaged'
            self.batch_size = 128
            self.val_batch_size  = 128
            self.test_batch_size = 1000
            self.dropout         = True
            self.epochs         = 1
            self.lr             = 0.001
            self.momentum       = 0.5
            self.log_interval   = 10
            self.n_nodes        = [3,2,2]# . structure of network (e.s. [3,2,2]: 3 CNN + 2 LSTM + 2 FC )
            self.cnn_filter     = 32     # . number filter for the first layer of CNN
            self.cnn_kernelsize = 3      # . dimension of kernel for CNN layer
            self.lstm_hidden    = 64     # . number of elements for hidden state of lstm
            self.fc_dim         = 1024   # . number of neurons for fully connected layer
            self.fc_dropout     = 0.5    # . drop out rate for fully connected layer
            self.rcnn_output    = 5      # . output of network
            self.num_classes    = 5      # PhysioNet total classes
            self.split_type = 'user_dependent' 

    args = Args()
'''

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


X = pickle.load( open( "./dataset/processed_data/"+args.data_type+"/X.p", "rb" ) )
y = pickle.load( open( "./dataset/processed_data/"+args.data_type+"/y.p", "rb" ) )
p = pickle.load( open( "./dataset/processed_data/"+args.data_type+"/p.p", "rb" ) )
dim = pickle.load( open( "./dataset/processed_data/"+args.data_type+"/dim.p", "rb" ) )

print('X', X.shape,'y', y.shape,'p',p.shape,'dim',len(dim))

'''

    # 3. Data Preprocessing

    The original goal of applying neural networks is to exclude hand-crafted algorithms & preprocessing as much as possible. I did not use any proprecessing techniques further than standardization to build an end-to-end classifer from the dataset

'''

def convert_mesh(X):
    
    mesh = np.zeros((X.shape[0], X.shape[2], 10, 11, 1))
    X = np.swapaxes(X, 1, 2)
    
    # 1st line
    mesh[:, :, 0, 4:7, 0] = X[:,:,21:24]; print('1st finished')
    
    # 2nd line
    mesh[:, :, 1, 3:8, 0] = X[:,:,24:29]; print('2nd finished')
    
    # 3rd line
    mesh[:, :, 2, 1:10, 0] = X[:,:,29:38]; print('3rd finished')
    
    # 4th line
    mesh[:, :, 3, 1:10, 0] = np.concatenate((X[:,:,38].reshape(-1, X.shape[1], 1),\
                                          X[:,:,0:7], X[:,:,39].reshape(-1, X.shape[1], 1)), axis=2)
    print('4th finished')
    
    # 5th line
    mesh[:, :, 4, 0:11, 0] = np.concatenate((X[:,:,(42, 40)],\
                                        X[:,:,7:14], X[:,:,(41, 43)]), axis=2)
    print('5th finished')
    
    # 6th line
    mesh[:, :, 5, 1:10, 0] = np.concatenate((X[:,:,44].reshape(-1, X.shape[1], 1),\
                                        X[:,:,14:21], X[:,:,45].reshape(-1, X.shape[1], 1)), axis=2)
    print('6th finished')
               
    # 7th line
    mesh[:, :, 6, 1:10, 0] = X[:,:,46:55]; print('7th finished')
    
    # 8th line
    mesh[:, :, 7, 3:8, 0] = X[:,:,55:60]; print('8th finished')
    
    # 9th line
    mesh[:, :, 8, 4:7, 0] = X[:,:,60:63]; print('9th finished')
    
    # 10th line
    mesh[:, :, 9, 5, 0] = X[:,:,63]; print('10th finished')
    
    return mesh

def create_folder(test_ratio, set_seed, user_independent):
    print('creating folders')
        
    if user_independent == 'user_independent':
        fold_name = 'user_independent'
    else:
        fold_name = 'user_dependent'
        
    DIRNAME = './dataset/splitted_data/'+args.data_type+'/'+fold_name
    new = False
    
    if not os.path.exists(os.path.dirname(DIRNAME)):
        os.makedirs(os.path.dirname(DIRNAME))

        
    DIRNAME = DIRNAME + '/test_rate_' + str(test_ratio) + '/seed_' + str(set_seed)+'/'
    
    if not os.path.exists(os.path.dirname(DIRNAME)):
        os.makedirs(os.path.dirname(DIRNAME))
        new = True
        
    return DIRNAME, new


def split_data(X, y, p, test_ratio, set_seed, user_independent):
    # Shuffle trials
    np.random.seed(set_seed)
    if user_independent == 'user_independent':
        trials = len(set([ele[0] for ele in p]))
    else:    
        trials = X.shape[0]
    print('trial', trials)
    shuffle_indices = np.random.permutation(trials)
    
    print('-- shaffleing X, y, p')
    if user_independent == 'user_independent':
        # Create a dict with empty list as default value.
        d = defaultdict(list)
        # print(y.shape, np.array([ele[0] for ele in y]).shape)
        for index, e in enumerate([ele[0] for ele in y]):
            # print('index', index, 'e', e)
            d[e].append(index)
        new_indexes = []
        for i in shuffle_indices:
            new_indexes += d[i]
        X = X[new_indexes]
        y = y[new_indexes]
        p = p[new_indexes]
        train_size = 0
        for i in shuffle_indices[:int(trials*(1-test_ratio))]:
            train_size += len(d[i])
                
    else:
        X = X[shuffle_indices]
        y = y[shuffle_indices]
        p = p[shuffle_indices]
        # Test set seperation
        train_size = int(trials*(1-test_ratio)) 
    
    print('-- split X, y, p in train-test',train_size)
    
    # X_train, X_test, y_train, y_test, p_train, p_test
    return  X[:train_size,:,:], X[train_size:,:,:], y[:train_size,:], y[train_size:,:], p[:train_size,:], p[train_size:,:]
               
def prepare_data(X, y, p, test_ratio, return_mesh, set_seed, user_independent):
    
    # y encoding
    # oh = OneHotEncoder(categories='auto')
    # y = oh.fit_transform(y).toarray()
    
    print('Split dataset:')
    X_train, X_test, y_train, y_test, p_train, p_test = split_data(X, y, p, test_ratio, set_seed, user_independent)
                                    
    # Z-score Normalization
    def scale_data(X):
        shape = X.shape
        for i in range(shape[0]):
            # Standardize a dataset along any axis
            # Center to the mean and component wise scale to unit variance.
            X[i,:, :] = scale(X[i,:, :])
            if i%int(shape[0]//10) == 0:
                print('{:.0%} done'.format((i+1)/shape[0]))   
        return X
    
    print('Scaling data')
    print('-- X train-test along any axis')
    X_train, X_test  = scale_data(X_train), scale_data(X_test)
    
    if return_mesh:
        print('Creating mesh')
        print('-- X train-test to mesh')
        X_train, X_test = convert_mesh(X_train), convert_mesh(X_test)
    
    return DIRNAME, X_train, X_test, y_train, y_test, p_train, p_test
    

DIRNAME, new = create_folder(args.test_rate, args.seed, args.split_type)
print('Folder: ',DIRNAME, 'dataset to split?', new)
if new:
    X_train, X_test, y_train, y_test, p_train, p_test = \
                prepare_data(X, y, p, args.test_rate, True, args.seed, args.split_type)

    print('X_train', X_train.shape, \
        'y_train', y_train.shape, \
        'p_train', len(set([p[0] for p in p_train])), '\n'
        'X_test', X_test.shape,  \
        'y_test', y_test.shape,  \
        'p_test', p_test.shape)


    X_train, X_val, y_train, y_val, p_train, p_val = \
                split_data(X_train, y_train, p_train, args.test_rate, args.seed, args.split_type)

else:

    '''

        # pickle.dump( [X_train, y_train]  , open( "./py/stack/train_tvt.p", "wb" ) , protocol=4)
        # pickle.dump( [X_val, y_val]  , open( "./py/stack/val_tvt.p", "wb" ) , protocol=4)
        # pickle.dump( [X_test, y_test]  , open( "./py/stack/test_tvt.p", "wb" ) , protocol=4)

    '''

    [X_train, y_train, p_train] = pickle.load( open( "./dataset/splitted_data/"+args.data_type+"/"+args.split_type+"/test_rate_"+str(args.test_rate)+"/seed_"+str(args.seed)+"/train.p", "rb" ) )
    [X_val, y_val, p_val]       = pickle.load( open( "./dataset/splitted_data/"+args.data_type+"/"+args.split_type+"/test_rate_"+str(args.test_rate)+"/seed_"+str(args.seed)+"/val.p"  , "rb" ) )
    [X_test, y_test, p_test]    = pickle.load( open( "./dataset/splitted_data/"+args.data_type+"/"+args.split_type+"/test_rate_"+str(args.test_rate)+"/seed_"+str(args.seed)+"/test.p" , "rb" ) )
    # num_input = X_train.shape[0]   # PhysioNet data input (mesh shape: 10*11)


'''

        As the EEG recording instrument has 3D locations over the subjects` scalp, it is essential for the model to learn from the spatial pattern as well as the temporal pattern. I transformed the data into 2D meshes that represents the locations of the electrodes so that stacked convolutional neural networks can grasp the spatial information.

        # 4. Modeling - Time-Distributed CNN + RNN
        Training Plan:

            4 GPU units (Nvidia Tesla P100) were used to train this neural network.

            Instead of training the whole model at once, I trained the first block (CNN) first. Then using the trained parameters as initial values, I trained the next blocks step-by-step. This approach can greatly reduce the time required for training and help avoiding falling into local minimums.

            The first blocks (CNN) can be applied for other EEG classification models as a pre-trained base.

            The initial learning rate is set to be
                            10^{3}
        
        with Adam optimization. I used several callbacks such as ReduceLROnPlateau which adjusts the learning rate at local minima. Also, I record the log for tensorboard to monitor the training process.

'''

print('train', 'X', X_train.shape, 'y', y_train.shape) #, 'p', p_train.shape)
print('val  ', 'X', X_val.shape  , 'y', y_val.shape) #, 'p', p_val.shape)
print('test ', 'X', X_test.shape , 'y', y_test.shape) #, 'p', p_test.shape)

print('\nsqueeze of all X, y - train, val, test\n')
# X_train = X_train.squeeze()
y_train = y_train.squeeze()
# X_val = X_val.squeeze()
y_val = y_val.squeeze()
# X_test = X_test.squeeze()
y_test = y_test.squeeze()

print('train', 'X', X_train.shape, 'y', y_train.shape) #, 'p', p_train.shape)
print('val  ', 'X', X_val.shape  , 'y', y_val.shape) #, 'p', p_val.shape)
print('test ', 'X', X_test.shape , 'y', y_test.shape) #, 'p', p_test.shape)

# X_train = X_train.reshape(*X_train.shape, 1)
# X_test = X_test.reshape(*X_test.shape, 1)

'''

        # 4.1 Pytorch Implementation

        Based on MNIST CNN + LSTM example

'''

class EEGImagesDatasetLoader(Dataset):
    """EEGs (converted in images) dataset."""

    def __init__(self, pickle_file='train.p', root_dir='./py/stack/', transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        [self.X, self.y] = pickle.load( open( root_dir + pickle_file, "rb" ) )
        
        '''
        def one_hot_embedding(labels, num_classes=args.num_classes):
            """Embedding labels to one-hot form.

            Args:
              labels: (LongTensor) class labels, sized [N,].
              num_classes: (int) number of classes.

            Returns:
              (tensor) encoded labels, sized [N, #classes].
            """
            y = torch.eye(num_classes) 
            return y[labels] 
        '''

        # self.y = one_hot_embedding(self.y, args.num_classes )
        # print(self.y)
        self.y = torch.tensor(self.y, dtype=torch.long)
        # print(self.y.size())
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        image = self.X[idx]
        label = self.y[idx]

        if self.transform:
            # If the transform variable is not empty
            # then it applies the operations in the transforms with the order that it is created.
            image = self.transform(image)

        return (image, label)


class EEGImagesDatasetHolding(Dataset):
    """EEGs (converted in images) dataset."""

    def __init__(self, X, y, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.X, self.y = X, y
        
        '''
        def one_hot_embedding(labels, num_classes=args.num_classes):
            """Embedding labels to one-hot form.

            Args:
              labels: (LongTensor) class labels, sized [N,].
              num_classes: (int) number of classes.

            Returns:
              (tensor) encoded labels, sized [N, #classes].
            """
            y = torch.eye(num_classes) 
            return y[labels] 
        '''

        # self.y = one_hot_embedding(self.y, args.num_classes )
        # print(self.y)
        self.y = torch.tensor(self.y, dtype=torch.long)
        # print(self.y.size())
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        image = self.X[idx]
        label = self.y[idx]

        if self.transform:
            # If the transform variable is not empty
            # then it applies the operations in the transforms with the order that it is created.
            image = self.transform(image)

        return (image, label)


# eeg_train = EEGImagesDatasetLoader(pickle_file='train_4.p', root_dir='./py/stack/', transform=None)
# eeg_test  = EEGImagesDatasetLoader(pickle_file='test_4.p', root_dir='./py/stack/', transform=None)

'''
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ]))
'''


eeg_train = EEGImagesDatasetHolding(X_train, y_train, transform=None)
eeg_val   = EEGImagesDatasetHolding(X_val  , y_val  , transform=None)
eeg_test  = EEGImagesDatasetHolding(X_test , y_test , transform=None)

train_loader = torch.utils.data.DataLoader(
    eeg_train,
    batch_size=args.batch_size,
    shuffle=True,
    **kwargs)

val_loader = torch.utils.data.DataLoader(
    eeg_val,
    batch_size=args.batch_size,
    shuffle=True,
    **kwargs)

test_loader = torch.utils.data.DataLoader(
    eeg_test,
    batch_size=args.batch_size,
    shuffle=True,
    **kwargs)

(image, label) = eeg_train[0]
print('train dimension:',eeg_train.__len__(), image.shape, label.shape)

(image, label) = eeg_val[0]
print('train dimension:',eeg_val.__len__(), image.shape, label.shape)

(image, label) = eeg_test[0]
print('test  dimension:',eeg_test.__len__(), image.shape, label.shape)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1  = nn.Conv2d(10, args.cnn_filter, kernel_size=args.cnn_kernelsize)
        self.batch1 = nn.BatchNorm2d(args.cnn_filter)
        # self.act1   = F.elu(args.cnn_filter, alpha=1.0, inplace=False)
        
        self.conv2 = nn.Conv2d(args.cnn_filter*1, args.cnn_filter*2, kernel_size=args.cnn_kernelsize)
        self.batch2 = nn.BatchNorm2d(args.cnn_filter*2**1)
        # self.act2   = F.elu(args.cnn_filter*2**1, alpha=1.0, inplace=False)
        
        self.conv3 = nn.Conv2d( args.cnn_filter * 2, args.cnn_filter * 2**2, kernel_size=args.cnn_kernelsize)
        self.batch3 = nn.BatchNorm2d(args.cnn_filter*2**2)
        # self.act3   = F.elu(args.cnn_filter*2**2, alpha=1.0, inplace=False)
        
        self.FC1 = nn.Linear(2560, args.fc_dim)
        if args.dropout:
            self.dropout1 = nn.Dropout(p=args.fc_dropout)
        self.batch4 = nn.BatchNorm1d(args.fc_dim)

    def forward(self, x):
        # print('cnn i', x.shape)
        x = F.elu(self.batch1(self.conv1(x)))
        # print('cnn 1', x.shape)
        x = F.elu(self.batch2(self.conv2(x)))
        # print('cnn 2', x.shape)
        x = F.elu(self.batch3(self.conv3(x)))
        # print('cnn 3', x.shape)
        x = x.view(-1, 2560) # 2560 = args.batch_size * x.shape[-1] * x.shape[-2]
        # print('cnn o/fc1 in', x.shape)
        if args.dropout:
            x = F.elu(self.batch4(self.dropout1(self.FC1(x))))
        else:
            x = F.elu(self.batch4(self.FC1(x)))
        # print('fc1 o', x.shape)
        return x

class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        self.cnn = CNN() 
        self.rnn = nn.LSTM(
                input_size  = args.fc_dim, 
                hidden_size = args.lstm_hidden, 
                num_layers  = 2, 
                batch_first = True )
        self.FC2 = nn.Linear(args.lstm_hidden, args.fc_dim)
        if args.dropout:
            self.dropout2 = nn.Dropout(p=args.fc_dropout)
        self.FC3 = nn.Linear(args.fc_dim, args.num_classes)

    def forward(self, x):
        # print('combine i', x.size())
        batch_size, C, H, W = x.size()
        c_in = x.view(batch_size , C, H, W)
        # print('combine cin', c_in.size())
        c_out = self.cnn(c_in)
        # print('combine cout/rin', c_out.size())
        r_in = c_out.view(batch_size, 1, -1)
        # print('combine rin', r_in.size())
        r_out, (h_n, h_c) = self.rnn(r_in)
        # print('combine rout', r_out.size(), '(h_n, h_c)', (h_n.size(), h_c.size()))
        if args.dropout:
            r_out2 = self.FC3(self.dropout2(self.FC2(r_out[:, -1, :])))
        else:
            r_out2 = self.FC3(self.FC2(r_out[:, -1, :]))
        # print('combine o', r_out2.size())
        # r_out2 = F.log_softmax(r_out2, dim=1)
        return r_out2 # CrossEntropyLoss adds softmax

class ConvColumn(nn.Module):
    def __init__(self, num_classes):
        super(ConvColumn, self).__init__()

        self.conv_layer1 = self._make_conv_layer(
            10, args.cnn_filter)
        self.conv_layer2 = self._make_conv_layer(
            args.cnn_filter, args.cnn_filter*2)
        self.conv_layer3 = self._make_conv_layer(
            args.cnn_filter*2, args.cnn_filter*4)

        self.fc5 = nn.Linear(args.cnn_filter*4*110, args.fc_dim)
        if args.dropout:
            self.fc5_dropout = nn.Dropout(p=args.fc_dropout)
        self.fc5_batch = nn.BatchNorm1d(args.fc_dim)
        self.fc5_act = nn.ELU()
        self.fc6 = nn.Linear(args.fc_dim, num_classes)

    def _make_conv_layer(self, in_c, out_c):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=args.cnn_kernelsize, stride=1, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ELU()
        )
        return conv_layer

    def forward(self, x):
        # print('concolumn i', x.size())
        x = self.conv_layer1(x)
        # print('cnn 1 o', x.size())
        x = self.conv_layer2(x)
        # print('cnn 2 o', x.size())
        x = self.conv_layer3(x)
        # print('cnn 3 o/flatten i', x.size())
        x = x.view(x.size(0), -1)
        # print('flatten o/fc i', x.size())
        x = self.fc5(x)
        # print('fc ', x.size())
        if args.dropout:
            x = self.fc5_dropout(x)
        x = self.fc5_act(self.fc5_batch(x))
        x = self.fc6(x)
        return x # CrossEntropyLoss adds softmax


current_time = datetime.now().strftime('%b%d_%H-%M-%S')
run_to_path = str(args.run_number)+str(args.n_nodes)+'_'+str(current_time)

my_log = './torchbearer/log_dir/'
my_log_dir = my_log +'CNN_LSTM_run'+run_to_path+'/'
if not os.path.exists(os.path.dirname(my_log_dir)):
    os.makedirs(os.path.dirname(my_log_dir))

save_model_dir = './torchbearer/save_model/CNN_LSTM_run'+run_to_path+'/'
if not os.path.exists(os.path.dirname(save_model_dir)):
    os.makedirs(os.path.dirname(save_model_dir))

if not os.path.exists(os.path.dirname(save_model_dir + 'last/')):
    os.makedirs(os.path.dirname(save_model_dir + 'last/'))

def write_report(model, history):
    infopath = save_model_dir + "/info.txt"
    with open(infopath, 'w') as fh:
        fh.write("Training parameters : \n")
        fh.write("Input dimensions (train) :  X " + str(X_train.shape) + ", y " + str(X_train.shape) + "\n")
        fh.write("Input dimensions (val  ) :  X " + str(X_val.shape)   + ", y " + str(X_val.shape)   + "\n")
        fh.write("Input dimensions (test ) :  X " + str(X_test.shape)  + ", y " + str(X_test.shape)  + "\n\n")
        fh.write("Epochs - LR - Dropout - Momentum : " + str(args.epochs) + " - " + str(args.lr) + " - " + str(args.fc_dropout) + " - " + str(args.momentum) + "\n")
        fh.write("Batch_size - Steps_train - Steps_valid : " + str(args.batch_size) + " - " + str(args.batch_size) + " - " + str(args.batch_size) +"\n")
        # fh.write("Final loss - val_loss : " + str(min([ history[i]['loss'] for i in range(len(history))])) + " - " + str(min([ history[i]['val_acc'] for i in range(len(history))])) + "\n\n")
        fh.write("Network architecture : \n")
        # string = model.state_dict() 
        # Pass the file handle in as a lambda function to make it callable
        # model.summary(print_fn=lambda x: fh.write(x + '\n'))
        print( model, file=fh)
        

callbacks_list = [
                # Best(save_model_dir + 'model_EP{epoch:02d}_VA{val_acc:.4f}.pt', save_model_params_only=True), # monitor='val_acc', mode='max'),
                # ExponentialLR(gamma=0.1),
                # TensorBoardText(comment=current_time),
                ModelCheckpoint(save_model_dir + 'model_[epo]{epoch:02d}_[val]{val_acc:.4f}.pt', save_best_only=True, monitor='val_acc', mode='max'),
                # ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5),
                StepLR(step_size=args.lr_stepsize, gamma=0.1),
                CSVLogger(my_log_dir + "log.csv", separator=',', append=True),
                # TensorBoard(my_log, write_graph=True, write_batch_metrics=False, write_epoch_metrics=False, comment='run'+run_to_path),
                # TensorBoard(my_log, write_graph=False, write_batch_metrics=True, batch_step_size=10, write_epoch_metrics=False, comment='run'+run_to_path),
                # TensorBoard(my_log, write_graph=False, write_batch_metrics=False, write_epoch_metrics=True, comment='run'+run_to_path)
                ]


# model = CNN_LSTM()
model = ConvColumn(args.num_classes)
if args.cuda:
    model.cuda()


# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
loss = nn.CrossEntropyLoss()

# Depending on Loss function it needed or not the One Hot Encoding
# the CrossEntropyLoss

print('\n# ==================================================== #')
print('#                        Model                         #')
print('# ==================================================== #\n\n',model)


print()
trial = Trial(model, optimizer, loss, metrics=['acc', 'loss'], callbacks=callbacks_list)
if args.cuda:
    trial.to('cuda')
history = trial.with_generators(train_generator=train_loader, val_generator=val_loader, test_generator=test_loader).run(args.epochs)
print(history)
