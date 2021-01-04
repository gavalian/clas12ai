import numpy as np
from termcolor import colored
import tensorflow as tf

from tensorflow.keras.models import Sequential

from models.AbstractKerasRegressor import AbstractKerasRegressor


class CnnDenoisingModel(AbstractKerasRegressor):
    """
    A denoising convolutional neural network Keras regressor.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'input_dict' in kwargs:
            input_dict = kwargs['input_dict']
            self.total_planes = input_dict["configuration"]["total_planes"]
            self.rings_per_plane = input_dict["configuration"]["rings_per_plane"]
            self.pads_per_ring = input_dict["configuration"]["pads_per_ring"]
            self.preprocess_input(input_dict)
        self.model = None
        
    # POLYKARPOS
    # This model did not work for me. I had to comment out lines 34 -> 37 to make it work in order to test things. 
    def build_new_model(self):
        k_model = Sequential()
        k_model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 4), activation='relu', padding="same",
                                         input_shape=(self.total_planes * self.rings_per_plane, self.pads_per_ring, 1)))
        k_model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        k_model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 4), activation='relu', padding="same"))
        k_model.add(tf.keras.layers.MaxPooling2D((3, 2)))
#        k_model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 4), activation='relu', padding="same"))
#        k_model.add(tf.keras.layers.MaxPooling2D((2, 2)))
#        k_model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 4), activation='relu', padding="same"))
#        k_model.add(tf.keras.layers.UpSampling2D((2, 2)))
        k_model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 4), activation='relu', padding="same"))
        k_model.add(tf.keras.layers.UpSampling2D((3, 2)))
        k_model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 4), activation='relu', padding="same"))
        k_model.add(tf.keras.layers.UpSampling2D((2, 2)))
        k_model.add(tf.keras.layers.Conv2D(1, kernel_size=(3, 4), activation='sigmoid', padding="same"))
        k_model.summary()

        k_model.compile(optimizer='nadam', loss='binary_crossentropy')

        self.model = k_model

    def preprocess_input(self, input_dict):
        if 'prediction' in input_dict:
            reshaped_data = input_dict["prediction"]["data"].reshape(-1, self.total_planes * self.rings_per_plane, self.pads_per_ring, 1)
            input_dict["prediction"]["data"] = reshaped_data
            
            return
        elif 'training' in input_dict:
            reshaped_x_train = input_dict["training"]["data"].reshape(-1, self.total_planes * self.rings_per_plane, self.pads_per_ring, 1)
            reshaped_y_train = input_dict["training"]["labels"].reshape(-1, self.total_planes * self.rings_per_plane, self.pads_per_ring, 1)

            input_dict["training"]["data"] = reshaped_x_train
            input_dict["training"]["labels"] = reshaped_y_train

        reshaped_x_test = input_dict["testing"]["data"].reshape(-1, self.total_planes * self.rings_per_plane, self.pads_per_ring, 1)
        reshaped_y_test = input_dict["testing"]["labels"].reshape(-1, self.total_planes * self.rings_per_plane, self.pads_per_ring, 1)

        input_dict["testing"]["data"] = reshaped_x_test
        input_dict["testing"]["labels"] = reshaped_y_test

    def train(self, input_dict) -> dict:
        print(colored("Training CNN denoising model...", "green"))
        return super().train(input_dict)

    def test(self, input_dict) -> dict:
        print(colored("Testing CNN denoising model...", "green"))
        return super().test(input_dict)

    def predict(self, input_dict) -> dict:
        print(colored("Setting up CNN model for prediction...", "green"))

        # TODO implement

        return super().predict(input_dict)
