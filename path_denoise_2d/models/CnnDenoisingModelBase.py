import numpy as np
from termcolor import colored
import tensorflow as tf
from tensorflow.keras.models import Sequential
from models.AbstractKerasRegressor import AbstractKerasRegressor
from abc import abstractmethod

class CnnDenoisingModelBase(AbstractKerasRegressor):
    """
    A denoising convolutional neural network Keras regressor base.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'input_dict' in kwargs:
            input_dict = kwargs['input_dict']
            self.layers = input_dict["configuration"]["layers"]
            self.sensors_per_layer = input_dict["configuration"]["sensors_per_layer"]
            self.preprocess_input(input_dict)
        self.model = None

    @abstractmethod
    def build_new_model(self):
        raise NotImplementedError

    def preprocess_input(self, input_dict):
        if 'prediction' in input_dict:
            reshaped_data = input_dict["prediction"]["data"].reshape(-1, self.layers, self.sensors_per_layer, 1)
            input_dict["prediction"]["data"] = reshaped_data

            return
        elif 'training' in input_dict:
            reshaped_x_train = input_dict["training"]["data"].reshape(-1, self.layers, self.sensors_per_layer, 1)
            reshaped_y_train = input_dict["training"]["labels"].reshape(-1, self.layers, self.sensors_per_layer, 1)

            input_dict["training"]["data"] = reshaped_x_train
            input_dict["training"]["labels"] = reshaped_y_train

        reshaped_x_test = input_dict["testing"]["data"].reshape(-1, self.layers, self.sensors_per_layer, 1)
        reshaped_y_test = input_dict["testing"]["labels"].reshape(-1, self.layers, self.sensors_per_layer, 1)

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
        return super().predict(input_dict)
