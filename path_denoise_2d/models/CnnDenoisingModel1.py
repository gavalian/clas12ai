import numpy as np
from termcolor import colored
import tensorflow as tf
from tensorflow.keras.models import Sequential
from models.AbstractKerasRegressor import AbstractKerasRegressor
from models.CnnDenoisingModelBase import CnnDenoisingModelBase


class CnnDenoisingModel1(CnnDenoisingModelBase):
    """
    A denoising convolutional neural network Keras regressor.
    """

    def build_new_model(self):
        k_model = Sequential()
        k_model.add(tf.keras.layers.Conv2D(64, kernel_size=(6, 6), strides=(6, 1), activation='relu', padding="same",
                                         input_shape=(self.layers, self.sensors_per_layer, 1)))
        k_model.add(tf.keras.layers.Conv2D(64, kernel_size=(2, 2), strides=(2, 1), activation='relu', padding="same"))
        k_model.add(tf.keras.layers.Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 1), activation='relu', padding="same"))
        k_model.add(tf.keras.layers.Conv2DTranspose(1, kernel_size=(6, 6), strides=(6, 1), activation='sigmoid', padding="same"))
        k_model.summary()

        k_model.compile(optimizer='nadam', loss='binary_crossentropy')

        self.model = k_model
