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

    """
    pass





