"""
This file contains a series of utility functions for manipulating SVM data
for the purpose of this project.
"""

import math
import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file, dump_svmlight_file


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
    tracks_per_event = []

    for v, array in zip(labels, features):
        if v != 0:
            valid.append(array)
            tracks_per_event.append(v)
        else:
            invalid.append(array)

    valid = np.array(np.vstack(valid)).reshape(-1, 36, 112, 1)
    invalid = np.array(np.vstack(invalid)).reshape(-1, 36, 112, 1)

    return invalid, valid, tracks_per_event


def write_raw_clean_denoised_to_svm(filename, raw, clean, denoised, num_features):
    """
    Writes the raw, clean and denoised data arrays to svml file

    Args:
        filename (string): SVM path
        raw (np.array): The raw data to be denoised
        clean (np.array): The clean data
        denoised (np.array): The denoised data
        num_features (int): Number of features
    """
    invalid = np.array(np.vstack(raw)).reshape(-1, num_features)
    valid = np.array(np.vstack(clean)).reshape(-1, num_features)
    denoised = np.array(np.vstack(denoised)).reshape(-1, num_features)
    X = np.empty((invalid.shape[0] + valid.shape[0] + denoised.shape[0], num_features), dtype=invalid.dtype)
    y = np.empty((invalid.shape[0] + valid.shape[0] + denoised.shape[0], ), dtype=np.int32)
    y[0::3] = np.zeros(invalid.shape[0])
    y[1::3] = np.ones(valid.shape[0])
    y[2::3] = np.ones(denoised.shape[0]) * 2
    X[0::3] = invalid
    X[1::3] = valid
    X[2::3] = denoised

    # y = y.reshape(-1,1)
    dump_svmlight_file(X, y, filename)
