import csv

from numpy.f2py.auxfuncs import throw_error
from pandas import read_csv
import numpy as np


def load_data(filename):
    """
    Load data from a csv file

    Parameters
    ----------
    filename : string
        Filename to be loaded.

    Returns
    -------
    X : ndarray
        the data matrix.

    y : ndarray
        the labels of each sample.
    """

    print(f"Loading data...{filename}")

    try:
        df = read_csv(filename)
        z = df.values

        y = z[:, 0]
        X = z[:, 1:]
        print('Y shape:', y.shape)
        print('X shape:', X.shape)
        return X, y

    except Exception as e:

        print(f"Error loading file: {e}")
        raise

try:
    load_data("data/mnist_data.csv")
except Exception as e:
    print(f'File not found{e}')







def split_data(x, y, tr_fraction=0.5):
    """
    Split the data x, y into two random subsets

    
    Split the data X,y into two random subsets.
    
    input:
        x: set of images
        y: labels
        fract_tr: float, percentage of samples to put in the training set.
            If necessary, number of samples in the training set is rounded to
            the lowest integer number.
    
    output:
        Xtr: set of images (numpy array, training set)
        Xts: set of images (numpy array, test set)
        ytr: labels (numpy array, training set)
        yts: labels (numpy array, test set)
    
    """
    trainSize = int(y.size * tr_fraction)
    idx = np.array(range(0, y.size))

    np.random.shuffle(idx)

    trainIdx = idx[0:trainSize]
    testIdx = idx[trainSize:]
    
    xTrain = x[trainIdx, :]
    yTrain = y[trainIdx]
    xTest = x[testIdx, :]
    yTest = y[testIdx]



