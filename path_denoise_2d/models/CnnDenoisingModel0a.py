import numpy as np
from termcolor import colored
import tensorflow as tf
from tensorflow.keras.models import Sequential
from models.AbstractKerasRegressor import AbstractKerasRegressor
from models.CnnDenoisingModelBase import CnnDenoisingModelBase


class CnnDenoisingModel0a(CnnDenoisingModelBase):
    """
    A denoising convolutional neural network Keras regressor.
    """

    def build_new_model(self):
        k_model = Sequential()
        k_model.add(tf.keras.layers.Conv2D(48, kernel_size=(5, 4), activation='relu', padding="same",
                                         input_shape=(self.layers, self.sensors_per_layer, 1)))
        k_model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        k_model.add(tf.keras.layers.Conv2D(48, kernel_size=(4, 3), activation='relu', padding="same"))
        k_model.add(tf.keras.layers.Conv2D(48, kernel_size=(4, 3), activation='relu', padding="same"))
        k_model.add(tf.keras.layers.MaxPooling2D((3, 2)))
        k_model.add(tf.keras.layers.Conv2D(48, kernel_size=(4, 3), activation='relu', padding="same"))
        k_model.add(tf.keras.layers.Conv2D(48, kernel_size=(4, 3), activation='relu', padding="same"))
        k_model.add(tf.keras.layers.UpSampling2D((3, 2)))
        k_model.add(tf.keras.layers.Conv2D(48, kernel_size=(5, 4), activation='relu', padding="same"))
        k_model.add(tf.keras.layers.Conv2D(48, kernel_size=(5, 4), activation='relu', padding="same"))
        k_model.add(tf.keras.layers.UpSampling2D((2, 2)))
        k_model.add(tf.keras.layers.Conv2D(1, kernel_size=(5, 4), activation='sigmoid', padding="same"))
        k_model.summary()

        k_model.compile(optimizer='nadam', loss='binary_crossentropy')

        self.model = k_model
