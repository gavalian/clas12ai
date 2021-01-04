import random
import numpy as np
import time
from detector_configuration import *
from sklearn.model_selection import train_test_split

# 21 planes
Z_OFFSET_RANGES_21 = [(-20, -1.5), (-1.5, 1.5), (1.5, 21)]

# 32 planes
Z_OFFSET_RANGES_32 = [(-20, -10), (-10, -1.5), (-1.5, 1.5), (1.5, 10), (10, 21)]

# 43 planes
Z_OFFSET_RANGES_43 = [(-20, -15), (-15, -10), (-10, -1.5), (-1.5, 1.5), (1.5, 10), (10, 15), (15, 21)]


def add_salt_pepper_noise(images, prob):
    """
    Generate a dataset by adding noise to the passed dataset.

    Args:
        images: Dataset
        prob: Noise probability

    Returns:
        Noisy dataset.
    """
    output = np.zeros(images.shape, np.float64)
    thres = 1 - prob
    for k in range(images.shape[0]):
        output[k] = images[k]
        for i in range(images[k].shape[0]):
            max_noise_pts = random.randint(0, 6)
            rdn = random.random()
            if rdn > thres:
                j = random.randint(0, 121)
                for _ in range(0, max_noise_pts):
                    output[k][random.randint(max(0, i - 3), min(images[k].shape[0] - 1, i + 3))][
                        random.randint(max(0, j - 3), min(j + 3, 121))] = 1
    return output


def preprocess_data_training(data, total_planes, padding=6, noise_percentage=0.45):
    """
    Preprocess data for training. Generates training and testing (validation) data.

    Args:
        data: Full dataset
        total_planes: Total number of planes in the dataset
        padding: Padding to add to each plane
        noise_percentage: Percentage of noise to add for generating noisy data

    Returns:
        Training features, training labels, testing features, testing labels
    """

    valid_data = data.reshape(-1, total_planes * RINGS_PER_PLANE, PADS_PER_RING)
    noisy_data = add_salt_pepper_noise(valid_data, noise_percentage)

    # Add padding to data
    valid_data = np.pad(valid_data, [(0, 0), (0, 0), (0, padding)], mode="constant")
    noisy_data = np.pad(noisy_data, [(0, 0), (0, 0), (0, padding)], mode="constant")

    x_train, x_test, y_train, y_test = train_test_split(noisy_data, valid_data, test_size=0.1, random_state=42)

    return x_train, x_test, y_train, y_test


def preprocess_data_predicting(data, total_planes, padding=6):
    """
     Preprocess data for predicting. Padds data with the requested columns.

    Args:
        data (numpy array): Full dataset
        total_planes (int): Total number of planes in the dataset
        padding (int, optional): Padding to add to each plane. Defaults to 6.

    Returns:
        numpy array, Dataset padded accordingly
    """

    valid_data = data.reshape(-1, total_planes * RINGS_PER_PLANE, PADS_PER_RING)
    padded_data = np.pad(valid_data, [(0, 0), (0, 0), (0, padding)], mode='constant')

    return padded_data



def preprocess_data_testing(data, total_planes, padding=6, noise_percentage=0.45):
    """"
    Preprocess data for testing. Generates testing (validation) data.

    Args:
        data: Full dataset
        total_planes: Total number of planes in the dataset
        padding: Padding to add to each plane
        noise_percentage: Percentage of noise to add for generating noisy data

    Returns:
        Testing features, testing labels
    """

    valid_data = data.reshape(-1, total_planes * RINGS_PER_PLANE, PADS_PER_RING)
    noisy_data = add_salt_pepper_noise(valid_data, noise_percentage)

    # Add padding to data
    valid_data = np.pad(valid_data, [(0, 0), (0, 0), (0, padding)], mode='constant')
    noisy_data = np.pad(noisy_data, [(0, 0), (0, 0), (0, padding)], mode='constant')

    return noisy_data, valid_data
