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
    data = read_csv(filename)
    z = np.array(data)
    y = z[:, 0]
    X = z[:, 1:]
    return X, y


def split_data(x, y, tr_fraction=0.5):
    """
    Split the data x, y into two random subsets

    """
    """Split the data X,y into two random subsets.
    
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
    yTrain = y[testIdx]
    xTest = x[testIdx, :]
    yTest = y[testIdx]

    return xTrain, yTrain, xTest, yTest
