import numpy as np
from termcolor import colored
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from models.CnnDenoisingModelBase import CnnDenoisingModelBase


class CnnDenoisingModel0(CnnDenoisingModelBase):
    """
    A denoising convolutional neural network Keras regressor.
    """

    def build_new_model(self):
        k_model = Sequential()
        k_model.add(tf.keras.layers.Conv2D(48, kernel_size=(4, 6), activation='relu', padding="same",
                                         input_shape=(self.layers, self.sensors_per_layer, 1)))
        k_model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        k_model.add(tf.keras.layers.Conv2D(48, kernel_size=(4, 6), activation='relu', padding="same"))
        k_model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        k_model.add(tf.keras.layers.Conv2D(48, kernel_size=(4, 6), activation='relu', padding="same"))
        k_model.add(tf.keras.layers.UpSampling2D((2, 2)))
        k_model.add(tf.keras.layers.Conv2D(48, kernel_size=(4, 6), activation='relu', padding="same"))
        k_model.add(tf.keras.layers.UpSampling2D((2, 2)))
        k_model.add(tf.keras.layers.Conv2D(1, kernel_size=(4, 6), activation='sigmoid', padding="same"))
        k_model.summary()

        optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
        k_model.compile(optimizer=optimizer, loss='binary_crossentropy')

        self.model = k_model
