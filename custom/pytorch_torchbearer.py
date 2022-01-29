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
                    default=128, 
                    help='input batch size for training (default: 128)')
parser.add_argument('--val_batch_size', 
                    type=int, 
                    default=128, 
                    help='input batch size for validation (default: 128)')
parser.add_argument('--test_batch_size', 
                    type=int, 
                    default=1000, 
                    help='input batch size for testing (default: 1000)')
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
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    os.system('cat /proc/sys/kernel/hostname')
    sys.stdout = old_stdout
    d = '######################           '+ ( mystdout.getvalue()+'\n' if os.system('uname') == 'Linux' else 'MAC\n' )
    d += '######################           Network Parameters'
    for k, v in vars(args).items():
        d += '\n'+str(k)+' : '+str(v)+', '
    d += '\n\n'
    print(d)
    os.system("curl -X POST -H 'Content-type: application/json' --data '{\"text\":\" "+d+"\"}' https://hooks.slack.com/services/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")
'''


# Network Parameters
# class Args:
#     def __init__(self):
#         self.test_rate  = 0.2
#         self.run_number = 0
#         self.cuda       = False
#         self.no_cuda    = True
#         self.seed       = 42
#         self.data_type  = 'Imaged'
#         self.batch_size = 128
#         self.val_batch_size  = 128
#         self.test_batch_size = 1000
#         self.dropout         = True
#         self.epochs         = 1
#         self.lr             = 0.001
#         self.momentum       = 0.5
#         self.log_interval   = 10
#         self.n_nodes        = [3,2,2]# . structure of network (e.s. [3,2,2]: 3 CNN + 2 LSTM + 2 FC )
#         self.cnn_filter     = 32     # . number filter for the first layer of CNN
#         self.cnn_kernelsize = 3      # . dimension of kernel for CNN layer
#         self.lstm_hidden    = 64     # . number of elements for hidden state of lstm
#         self.fc_dim         = 1024   # . number of neurons for fully connected layer
#         self.fc_dropout     = 0.5    # . drop out rate for fully connected layer
#         self.rcnn_output    = 5      # . output of network
#         self.num_classes    = 5      # PhysioNet total classes
#         self.split_type = 'user_dependent' 
# 
# args = Args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

'''
# Get Data

        # just to take the dataset from physionet web site
        CONTEXT = 'pn4/'
        MATERIAL = 'eegmmidb/'
        URL = 'https://www.physionet.org/' + CONTEXT + MATERIAL

        # Change this directory according to your setting
        USERDIR = './py/data/'

        page = requests.get(URL).text
        FOLDERS = sorted(list(set(re.findall(r'S[0-9]+', page))))

        URLS = [URL+x+'/' for x in FOLDERS]

        # Warning: Executing this block will create folders
        for folder in FOLDERS:
            pathlib.Path(USERDIR +'/'+ folder).mkdir(parents=True, exist_ok=True)

        # Warning: Executing this block will start downloading data
        for i, folder in enumerate(FOLDERS):
            page = requests.get(URLS[i]).text
            subs = list(set(re.findall(r'S[0-9]+R[0-9]+', page)))
            
            print('Working on {}, {:.1%} completed'.format(folder, (i+1)/len(FOLDERS)))
            for sub in subs:
                urllib.request.urlretrieve(URLS[i]+sub+'.edf', os.path.join(USERDIR, folder, sub+'.edf'))

'''

'''
# Data Description

        Subjects performed different motor/imagery tasks while 64-channel EEG were recorded using the BCI2000 system (http://www.bci2000.org). Each subject performed 14 experimental runs:

        two one-minute baseline runs (one with eyes open, one with eyes closed)
        three two-minute runs of each of the four following tasks:
        1:
        A target appears on either the left or the right side of the screen.
        The subject opens and closes the corresponding fist until the target disappears.
        Then the subject relaxes.
        2:
        A target appears on either the left or the right side of the screen.
        The subject imagines opening and closing the corresponding fist until the target disappears.
        Then the subject relaxes.
        3:
        A target appears on either the top or the bottom of the screen.
        The subject opens and closes either both fists (if the target is on top) or both feet (if the target is on the bottom) until the target disappears.
        Then the subject relaxes.
        4:
        A target appears on either the top or the bottom of the screen.
        The subject imagines opening and closing either both fists (if the target is on top) or both feet (if the target is on the bottom) until the target disappears.
        Then the subject relaxes.
        The data are provided here in EDF+ format (containing 64 EEG signals, each sampled at 160 samples per second, and an annotation channel). For use with PhysioToolkit software, rdedfann generated a separate PhysioBank-compatible annotation file (with the suffix .event) for each recording. The .event files and the annotation channels in the corresponding .edf files contain identical data.

        # Summary tasks

        Remembering that:

        - Task 1 (open and close left or right fist)
        - Task 2 (imagine opening and closing left or right fist)
        - Task 3 (open and close both fists or both feet)
        - Task 4 (imagine opening and closing both fists or both feet)
        we will referred to 'Task *' with the meneaning above.

        In summary, the experimental runs were:

        Baseline, eyes open
        Baseline, eyes closed
        Task 1
        Task -2
        Task --3
        Task ---4
        Task 1
        Task -2
        Task --3
        Task ---4
        Task 1
        Task -2
        Task --3
        Task ---4

        # Annotation

        Each annotation includes one of three codes (T0, T1, or T2):

        T0 corresponds to rest
        T1 corresponds to onset of motion (real or imagined) of
            the left fist (in runs 3, 4, 7, 8, 11, and 12)
            both fists (in runs 5, 6, 9, 10, 13, and 14)
        T2 corresponds to onset of motion (real or imagined) of
            the right fist (in runs 3, 4, 7, 8, 11, and 12)
            both feet (in runs 5, 6, 9, 10, 13, and 14)
        In the BCI2000-format versions of these files, which may be available from the contributors of this data set, these annotations are encoded as values of 0, 1, or 2 in the TargetCode state variable.

        {'T0':0, 'T1':1, 'T2':2}

        In our experiments we will see only :

        run_type_0:
            append_X
        run_type_1
            append_X_y
        run_type_2
            append_X_y
        and the coding is:

        T0 corresponds to rest
            (2)
        T1 (real or imagined)
            (4, 8, 12) the left fist
            (6, 10, 14) both fists
        T2 (real or imagined)
            (4, 8, 12) the right fist
            (6, 10, 14) both feet

        # 2. Raw Data Import

        I will use a EEG data handling package named MNE (https://martinos.org/mne/stable/index.html) to import raw data and annotation for events from edf files. This package also provides essential signal analysis features, e.g. band-pass filtering. The raw data were filtered using 1Hz of high-pass filter.

        In this research, there are 5 classes for the data, imagined motion of:

        - right fist, 
        - left fist, 
        - both fists, 
        - both feet,
        - rest with eyes closed.
        A data (S089) from one of the 109 subjects was excluded as the record was severely corrupted.

'''

'''
# Get file paths
    PATH = './py/data/'
    SUBS = glob(PATH + 'S[0-9]*')
    FNAMES = sorted([x[-4:] for x in SUBS])

    REMOVE = ['S088', 'S089', 'S092', 'S100']

    # Remove subject 'S089' with damaged data and 'S088', 'S092', 'S100' with 128Hz sampling rate (we want 160Hz)
    FNAMES = [ x for x in FNAMES if x not in REMOVE] 

    emb = {'T0': 1, 'T1': 2, 'T2': 3}

    def my_get_data(subj_num=FNAMES, epoch_sec=0.0625):
        """ Import from edf files data and targets in the shape of 3D tensor
        
            Output shape: (Trial*Channel*TimeFrames)
            
            Some edf+ files recorded at low sampling rate, 128Hz, are excluded. 
            Majority was sampled at 160Hz.
            
            epoch_sec: time interval for one segment of mashes (0.0625 is 1/16 as a fraction)
        """
        
        # Event codes mean different actions for two groups of runs    
        run_type_0 = '02'.split(',')
        run_type_1 = '04,08,12'.split(',')
        run_type_2 = '06,10,14'.split(',')
        
        # Initiate X, y
        X = []
        y = []
        
        # To compute the completion rate
        count = len(subj_num)
        
        # fixed numbers
        nChan = 64 
        sfreq = 160
        sliding = epoch_sec/2 
        timeFromQue = 0.5
        timeExercise = 4.1 #secomds

        # Sub-function to assign X and X, y
        def append_X(n_segments, data, event=[]):
            # Data should be changed
            """This function generate a tensor for X and append it to the existing X"""
        
            def window(n):
                # (80) + (160 * 1/16 * n) 
                windowStart = int(timeFromQue*sfreq) + int(sfreq*sliding*n) 
                # (80) + (160 * 1/16 * (n+2))
                windowEnd = int(timeFromQue*sfreq) + int(sfreq*sliding*(n+2)) 
                
                while (windowEnd - windowStart) != sfreq*epoch_sec:
                    windowEnd += int(sfreq*epoch_sec) - (windowEnd - windowStart)
                    
                return [windowStart, windowEnd]
            
            new_x = []
            for n in range(n_segments):
                # print('data[:, ',window(n)[0],':',window(n)[1],'].shape = ', data[:, window(n)[0]:window(n)[1]].shape, '(',nChan,',',int(sfreq*epoch_sec),')')
                
                if data[:, window(n)[0]:window(n)[1]].shape==(nChan, int(sfreq*epoch_sec)):
                    new_x.append(data[:, window(n)[0]: window(n)[1]])
                    
            return new_x
        
        def append_X_Y(run_type, event, old_x, old_y, data):
            """This function seperate the type of events 
            (refer to the data descriptitons for the list of the types)
            Then assign X and Y according to the event types"""
            # Number of sliding windows

            # print('data', data.shape[1])
            n_segments = floor(data.shape[1]/(epoch_sec*sfreq*timeFromQue) - 1/epoch_sec - 1)
            # print('run_'+str(run_type),' n_segments', n_segments)
            
            # Rest excluded
            if event[2] == emb['T0']:
                return old_x, old_y
            
            # y assignment
            if run_type == 1:
                temp_y = [1] if event[2] == emb['T1'] else [2]
            
            elif run_type == 2:
                temp_y = [3] if event[2] == emb['T1'] else [4]
                
            print('event[2]', event[2], 'run_type', run_type, 'temp_y', temp_y)            
            
            # print('timeExercise * sfreq', timeExercise*sfreq, ' ?= 656')
            new_x = append_X(n_segments, data, event)
            new_y = old_y + temp_y*len(new_x)
            
            return old_x + new_x, new_y
        
        # Iterate over subj_num: S001, S002, S003, ...
        for i, subj in enumerate(subj_num):

            # Return completion rate
            if i%((len(subj_num)//10)+1) == 0:
                print('\n')
                print('working on {}, {:.0%} completed'.format(subj, i/count))
                print('\n')
            
            old_size = np.array(y).shape[0]
            # print('subj:', subj, '| y.shape', np.array(y).shape ,'| X.shape', np.array(X).shape)

            # Get file names
            fnames = glob(os.path.join(PATH, subj, subj+'R*.edf'))
            # Hold only the files that have an even number
            fnames = sorted([name for name in fnames if name[-6:-4] in run_type_0+run_type_1+run_type_2])

            # for each of ['02', '04', '06', '08', '12', '14']
            for i, fname in enumerate(fnames):
                
                # Import data into MNE raw object
                raw = read_raw_edf(fname, preload=True, verbose=False)
                
                picks = pick_types(raw.info, eeg=True)
                # print('n_times', raw.n_times)
                
                if raw.info['sfreq'] != 160:
                    print('{} is sampled at 128Hz so will be excluded.'.format(subj))
                    break
                
                # High-pass filtering
                raw.filter(l_freq=1, h_freq=None, picks=picks)

                # Get annotation
                try:
                    events = events_from_annotations(raw, verbose=False)
                except:
                    continue

                # Get data
                data = raw.get_data(picks=picks)

                # print('event.shape', np.array(events[0]).shape, '| data.shape', data.shape)

                # Number of this run
                which_run = fname[-6:-4]

                """ Assignment Starts """ 
                # run 1 - baseline (eye closed)
                if which_run in run_type_0:

                    # Number of sliding windows
                    n_segments = floor(data.shape[1]/(epoch_sec*sfreq*timeFromQue) - 1/epoch_sec - 1)
                    # print('run_0 n_segments', n_segments)

                    # Append 0`s based on number of windows
                    new_X = append_X(n_segments, data)
                    X += new_X
                    y.extend([0] * len(new_X))
                    # print(events[0])   

                # run 4,8,12 - imagine opening and closing left or right fist    
                elif which_run in run_type_1:

                    for i, event in enumerate(events[0]):

                        X, y = append_X_Y(run_type=1, event=event, old_x=X, old_y=y, data=data[:, int(event[0]) : int(event[0] + timeExercise*sfreq)])
                        # print(event)   

                # run 6,10,14 - imagine opening and closing both fists or both feet
                elif which_run in run_type_2:

                    for i, event in enumerate(events[0]):      

                        X, y = append_X_Y(run_type=2, event=event, old_x=X, old_y=y, data=data[:, int(event[0]) : int(event[0] + timeExercise*sfreq)])
                        # print(event)    

            print('subj:', subj, '|', np.array(y).shape[0] - old_size, '| y.shape', np.array(y).shape ,'| X.shape', np.array(X).shape)
            
        print(np.array(X).shape)

        X = np.stack(X)
        y = np.array(y).reshape((-1,1))
        return X, y


    def get_data(subj_num=FNAMES, epoch_sec=0.0625):
        """ Import from edf files data and targets in the shape of 3D tensor
        
            Output shape: (Trial*Channel*TimeFrames)
            
            Some edf+ files recorded at low sampling rate, 128Hz, are excluded. 
            Majority was sampled at 160Hz.
            
            epoch_sec: time interval for one segment of mashes
            """
        
        # Event codes mean different actions for two groups of runs
        run_type_0 = '02'.split(',')
        run_type_1 = '04,08,12'.split(',')
        run_type_2 = '06,10,14'.split(',')
        
        # Initiate X, y
        X = []
        y = []
        
        # To compute the completion rate
        count = len(subj_num)
        
        # fixed numbers
        nChan = 64 
        sfreq = 160
        sliding = epoch_sec/2 
        timeFromQue = 0.5

        # Sub-function to assign X and X, y
        def append_X(n_segments, data, event=[]):
            # Data should be changed
            """This function generate a tensor for X and append it to the existing X"""
            
            if len(event):
                event_start = ceil(event[0] * sfreq)
            else:
                event_start = 0
        
            def window(n):
                windowStart = int(timeFromQue*sfreq) + int(sfreq*sliding*n) + event_start
                windowEnd = int(timeFromQue*sfreq) + int(sfreq*sliding*(n+2)) + event_start
                
                while (windowEnd - windowStart) != 10:
                    windowEnd += int(sfreq*epoch_sec) - (windowEnd - windowStart)
                    
                return [windowStart, windowEnd]
            
            new_x = [data[:, window(n)[0]: window(n)[1]] for n in range(n_segments)\
                    if data[:, window(n)[0]:window(n)[1]].shape==(nChan, int(sfreq*epoch_sec))]
            return new_x
        
        def append_X_Y(run_type, event, old_x, old_y, data):
            """This function seperate the type of events 
            (refer to the data descriptitons for the list of the types)
            Then assign X and Y according to the event types"""
            # Number of sliding windows
            n_segments = int(event[1]/epoch_sec)
            print('run_'+str(run_type),' n_segments', n_segments) 

            # Rest excluded
            if event[2] == emb['T0']:
                return old_x, old_y
            
            # y assignment
            if run_type == 1:
                temp_y = [1] if event[2] == emb['T1'] else [2]
            
            elif run_type == 2:
                temp_y = [3] if event[2] == emb['T1'] else [4]
                    
            print('event[2]', event[2], 'run_type', run_type, 'temp_y', temp_y)            

            new_x = append_X(n_segments, data, event)
            new_y = old_y + temp_y*len(new_x)
            
            return old_x + new_x, new_y
        
        # Iterate over subj_num: S001, S002, S003...
        for i, subj in enumerate(subj_num):

            # Return completion rate
            if i%((len(subj_num)//10)+1) == 0:
                print('\n')
                print('working on {}, {:.0%} completed'.format(subj, i/count))
                print('\n')

            old_size = np.array(y).shape[0]
            # Get file names
            fnames = glob(os.path.join(PATH, subj, subj+'R*.edf'))
            fnames = sorted([name for name in fnames if name[-6:-4] in run_type_0+run_type_1+run_type_2])
            
            for i, fname in enumerate(fnames):
                
                # Import data into MNE raw object
                raw = read_raw_edf(fname, preload=True, verbose=False)

                picks = pick_types(raw.info, eeg=True)
                
                if raw.info['sfreq'] != 160:
                    print('{} is sampled at 128Hz so will be excluded.'.format(subj))
                    break
                
                # High-pass filtering
                raw.filter(l_freq=1, h_freq=None, picks=picks)
                
                # Get annotation
                try:
                    events = raw.find_edf_events()
                except:
                    continue
                    
                # Get data
                data = raw.get_data(picks=picks)
                
                # Number of this run
                which_run = fname[-6:-4]
                
                """ Assignment Starts """ 
                # run 1 - baseline (eye closed)
                if which_run in run_type_0:
                    
                    # Number of sliding windows
                    n_segments = int((raw.n_times/(epoch_sec*sfreq)))

                    # Append 0`s based on number of windows
                    new_X = append_X(n_segments, data)
                    X += new_X
                    y.extend([0] * len(new_X))
                        
                # run 4,8,12 - imagine opening and closing left or right fist    
                elif which_run in run_type_1:
                    
                    for i, event in enumerate(events):
                        X, y = append_X_Y(run_type=1, event=event, old_x=X, old_y=y, data=data)
                            
                # run 6,10,14 - imagine opening and closing both fists or both feet
                elif which_run in run_type_2:
                    
                    for i, event in enumerate(events):         
                        X, y = append_X_Y(run_type=2, event=event, old_x=X, old_y=y, data=data)
            
            print('subj:', subj, '|', np.array(y).shape[0] - old_size, '| y.shape', np.array(y).shape ,'| X.shape', np.array(X).shape)

        X = np.stack(X)
        y = np.array(y).reshape((-1,1))
        return X, y

    X,y = get_data(FNAMES, epoch_sec=0.0625) # to get 20 set 0.125

    print(X.shape)
    print(y.shape)

    pickle.dump( X , open( "./py/stack/X.p", "wb" ) , protocol=4)

    pickle.dump( y , open( "./py/stack/y.p", "wb" ) , protocol=4)

    X = pickle.load( open( "./py/stack/X.p", "rb" ) )
    y = pickle.load( open( "./py/stack/y.p", "rb" ) )

'''

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
X_train = X_train.squeeze()
y_train = y_train.squeeze()
X_val = X_val.squeeze()
y_val = y_val.squeeze()
X_test = X_test.squeeze()
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
    batch_size=args.val_batch_size,
    shuffle=True,
    **kwargs)

test_loader = torch.utils.data.DataLoader(
    eeg_test,
    batch_size=args.test_batch_size,
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
        return r_out2 # F.log_softmax(r_out2, dim=1)


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
        fh.write("Batch_size - Steps_train - Steps_valid : " + str(args.batch_size) + " - " + str(args.val_batch_size) + " - " + str(args.test_batch_size) +"\n")
        fh.write("Final loss - val_loss : " + str(min([ history[i]['loss'] for i in range(len(history))])) + " - " + str(min([ history[i]['val_loss'] for i in range(len(history))])) + "\n\n")
        fh.write("Network architecture : \n")
        # string = model.state_dict() 
        # Pass the file handle in as a lambda function to make it callable
        # model.summary(print_fn=lambda x: fh.write(x + '\n'))
        print( model, file=fh)
        

callbacks_list = [
                # Best(save_model_dir + 'model_EP{epoch:02d}_VA{val_acc:.4f}.pt', save_model_params_only=True), # monitor='val_acc', mode='max'),
                # ExponentialLR(gamma=0.1),
                # TensorBoardText(comment=current_time),
                ModelCheckpoint(save_model_dir + 'model_[epo]{epoch:02d}_[val]{val_acc:.4f}.pt', save_best_only=True, monitor='val_loss'),
                # ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5),
                StepLR(step_size=100, gamma=0.1),
                CSVLogger(my_log_dir + "log.csv", separator=',', append=True),
                # TensorBoard(my_log, write_graph=True, write_batch_metrics=False, write_epoch_metrics=False, comment='run'+run_to_path),
                # TensorBoard(my_log, write_graph=False, write_batch_metrics=True, batch_step_size=10, write_epoch_metrics=False, comment='run'+run_to_path),
                TensorBoard(my_log, write_graph=False, write_batch_metrics=False, write_epoch_metrics=True, comment='run'+run_to_path)]

model = CNN_LSTM()

'''
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    os.system('cat /proc/sys/kernel/hostname')
    sys.stdout = old_stdout

    d = '######################           '+ ( mystdout.getvalue()+'\n' if os.system('uname') == 'Linux' else 'MAC\n' )

    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    print( model )
    sys.stdout = old_stdout
    d += '######################           Network Architectures'+mystdout.getvalue()
    os.system("curl -X POST -H 'Content-type: application/json' --data '{\"text\":\""+str(d)+"\n\"}' https://hooks.slack.com/services/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")
'''

if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
loss = nn.CrossEntropyLoss()
print('\n# ==================================================== #')
print('#                        Model                         #')
print('# ==================================================== #\n\n',model)


print()
trial = Trial(model, optimizer, loss, metrics=['acc', 'loss'], callbacks=callbacks_list).to('cuda')
history = trial.with_generators(train_generator=train_loader, val_generator=val_loader, test_generator=test_loader).run(args.epochs)
print(history)

'''
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    os.system('cat /proc/sys/kernel/hostname')
    sys.stdout = old_stdout

    d = '######################           '+ ( mystdout.getvalue()+'\n' if os.system('uname') == 'Linux' else 'MAC\n' )
    d = '######################           Network History'
    for i, epoch in enumerate(history):
        d += '\n ####    EPOCH '+str(i)
        for k, v in epoch.items():
            d += '\n'+str(k)+' : '+str(v)
        d += '\n\n'
    print(d)
    os.system("curl -X POST -H 'Content-type: application/json' --data '{\"text\":\" "+d+"\"}' https://hooks.slack.com/services/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")
'''
write_report(model, history)
