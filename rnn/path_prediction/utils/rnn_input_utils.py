"""
This file contains a series of utility functions for creating a dataset
for the RNN models.
"""

import numpy as np


def remove_zeros(dataset):
    for v in dataset:
        for i in range(0, v.size):
            if v[i] == 0:
                if i > 0 and i < 35:
                    v[i] = (v[i-1] + v[i+1]) / 2
                elif i == 0:
                    v[i] = v[i+1]
                elif i == 35:
                    v[i] = v[i-1]
    return dataset


def create_rnn_dataset_segment_major(rows):
    rows_reshaped = rows.reshape(-1, 6, 6)
    rows_new = np.delete(rows_reshaped, np.s_[4:6], axis=1)

    labels_new = np.take(rows_reshaped, np.r_[4:6], axis=1)
    labels_new = labels_new.reshape(-1, 12)

    return rows_new, labels_new


def create_rnn_dataset_segment_major_2(rows):
    rows_reshaped = rows.reshape(-1, 6, 6)
    rows_new = np.delete(rows_reshaped, np.s_[4:6], axis=1)

    labels_new = np.take(rows_reshaped, np.r_[2:6], axis=1)
    labels_new = labels_new.reshape(-1, 24)

    return rows_new, labels_new


def create_rnn_dataset_layer_major(rows):
    rows_reshaped = rows.reshape(-1, 36, 1)
    rows_new = np.delete(rows_reshaped, np.s_[24:36], axis=1)

    labels_new = np.take(rows_reshaped, np.r_[24:36], axis=1)
    labels_new = labels_new.reshape(-1, 12)

    return rows_new, labels_new


def create_rnn_dataset_layer_major_2(rows):
    rows_reshaped = rows.reshape(-1, 36, 1)
    rows_new = np.delete(rows_reshaped, np.s_[24:36], axis=1)

    labels_new = np.take(rows_reshaped, np.r_[12:36], axis=1)
    labels_new = labels_new.reshape(-1, 24)

    return rows_new, labels_new


def create_rnn_dataset_double_layer_major(rows):
    rows_reshaped = rows.reshape(-1, 18, 2)
    rows_new = np.delete(rows_reshaped, np.s_[12:18], axis=1)

    labels_new = np.take(rows_reshaped, np.r_[12:18], axis=1)
    labels_new = labels_new.reshape(-1, 12)

    return rows_new, labels_new


def create_rnn_dataset_triple_layer_major(rows):
    rows_reshaped = rows.reshape(-1, 12, 3)
    rows_new = np.delete(rows_reshaped, np.s_[8:12], axis=1)

    labels_new = np.take(rows_reshaped, np.r_[8:12], axis=1)
    labels_new = labels_new.reshape(-1, 12)

    return rows_new, labels_new
