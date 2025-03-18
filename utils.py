import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import os
import glob
import sys

def plot_data(data, obj):
    if obj == 'MET':
        k = 1
    elif obj == 'Ele':
        k = 2
    elif obj == 'Mu':
        k = 3
    elif obj == 'Jet':
        k = 4

    plt.figure(figsize=(7,5))

    for i in range(len(data)):
        plt.hist(data[i][:,:,0][data[i][:,:,3]==k], log=True, label=files[i], alpha=1, histtype='step',
                 density=True, bins=100)
        plt.xlabel('Pt [GeV]')
        plt.title(obj)
        plt.legend()
    plt.show()
    
    plt.figure(figsize=(7,5))
    for i in range(len(data)):
        plt.hist(data[i][:,:,1][data[i][:,:,3]==k], log=True, label=files[i], alpha=1, histtype='step',
                 density=True, bins=100)
        plt.xlabel('Eta')
        plt.title(obj)
        #plt.legend()
    plt.show()
    
    plt.figure(figsize=(7,5))
    for i in range(len(data)):
        plt.hist(data[i][:,:,2][data[i][:,:,3]==k], log=True, label=files[i], alpha=1, histtype='step',
                 density=True, bins=100)
        plt.xlabel('Phi')
        plt.title(obj)
        #plt.legend()
    plt.show()


