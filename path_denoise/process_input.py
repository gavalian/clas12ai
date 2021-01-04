"""
This file contains a series of utility functions for manipulating SVM data
for the purpose of this project.
"""

import math
import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file

def read_svm_file(filename, num_features):
    """
    Reads the SVM file with the passed name and returns
    the rows in dense format and the labels.

    Args:
        filename (string): The path of the SVM file

    Returns:
        dense rows, labels numpy arrays
    """

    data = load_svmlight_file(filename, n_features = num_features)
    data0_dense = data[0].todense()

    return np.array(data0_dense), data[1]

def read_svm_to_X_Y_datasets(filename, num_features):
    """
    Reads the SVM file with the passed name and returns
    the X (noisy) and Y (clean) data arrays

    Args:
        filename (string): SVM path
        num_features (int): Number of features
    """

    features, labels = read_svm_file(filename, num_features)
    valid = []
    invalid = []

    for v, array in zip(labels, features):
        if v == 1:
            valid.append(array)
        else:
            invalid.append(array)

    valid = np.array(np.vstack(valid)).reshape(-1, 36, 112, 1)
    invalid = np.array(np.vstack(invalid)).reshape(-1, 36, 112, 1)

    return invalid, valid