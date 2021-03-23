import numpy as np
import copy
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
            #self.padding_x = input_dict["configuration"]["padding_x"]
            #self.padding_y = input_dict["configuration"]["padding_y"]
            #self.use_padding = self.padding_x > 0 or self.padding_y > 0
        self.model = None

    def build_new_model(self):
        raise NotImplementedError

    def preprocess_input(self, input_dict):
        # Padding
        expected_input_height = self.model.layers[0].input_shape[1]
        expected_input_width = self.model.layers[0].input_shape[2]

        if expected_input_height < self.layers:
            print("Error: Expected model input height less than number of layers in configuration")
        elif expected_input_width < self.sensors_per_layer:
            print("Error: Expected model input width less than number of sensors per layer in configuration")

        if (expected_input_height - self.layers ) % 2 != 0:
            print("Error: Height padding cannot be applied as the difference between the number of layers and the model input height is not divible by 2")
        elif (expected_input_width - self.sensors_per_layer) % 2 != 0:
            print("Error: Width padding cannot be applied as the difference between the number of sensors per layer and the model input width is not divible by 2")

        py = int((expected_input_height - self.layers) / 2)
        px = int((expected_input_width - self.sensors_per_layer) / 2)
        self.padding_y = py
        self.padding_x = px
        self.use_padding = self.padding_x > 0 or self.padding_y > 0

        out_dict = copy.deepcopy(input_dict)

        if 'prediction' in input_dict:
            reshaped_data = input_dict["prediction"]["data"].reshape(-1, self.layers, self.sensors_per_layer, 1)

            if self.use_padding:
                reshaped_data = np.pad(reshaped_data, pad_width=[(0, 0), (py, py), (px, px), (0, 0)], mode='constant')

            out_dict["prediction"]["data"] = reshaped_data

            return
        elif 'training' in input_dict:
            reshaped_x_train = input_dict["training"]["data"].reshape(-1, self.layers, self.sensors_per_layer, 1)
            reshaped_y_train = input_dict["training"]["labels"].reshape(-1, self.layers, self.sensors_per_layer, 1)

            if self.use_padding:
                reshaped_x_train = np.pad(reshaped_x_train, pad_width=[(0, 0), (py, py), (px, px), (0, 0)], mode='constant')
                reshaped_y_train = np.pad(reshaped_y_train, pad_width=[(0, 0), (py, py), (px, px), (0, 0)], mode='constant')

            out_dict["training"]["data"] = reshaped_x_train
            out_dict["training"]["labels"] = reshaped_y_train

        # Testing data
        reshaped_x_test = input_dict["testing"]["data"].reshape(-1, self.layers, self.sensors_per_layer, 1)
        reshaped_y_test = input_dict["testing"]["labels"].reshape(-1, self.layers, self.sensors_per_layer, 1)

        if self.use_padding:
            reshaped_x_test = np.pad(reshaped_x_test, pad_width=[(0, 0), (py, py), (px, px), (0, 0)], mode='constant')
            reshaped_y_test = np.pad(reshaped_y_test, pad_width=[(0, 0), (py, py), (px, px), (0, 0)], mode='constant')

        out_dict["testing"]["data"] = reshaped_x_test
        out_dict["testing"]["labels"] = reshaped_y_test

        return out_dict

    def train(self, input_dict) -> dict:
        print(colored("Training CNN denoising model...", "green"))
        return super().train(input_dict)

    def test(self, input_dict) -> dict:
        print(colored("Testing CNN denoising model...", "green"))
        return super().test(input_dict)

    def predict(self, input_dict) -> dict:
        print(colored("Setting up CNN model for prediction...", "green"))
        return super().predict(input_dict)

    def remove_padding(self, input):
        px = self.padding_x
        py = self.padding_y

        slices = []

        slices.append(slice(0,None))

        if py > 0:
            slices.append(slice(py,-py))
        else:
            slices.append(slice(0,None))

        if px > 0:
            slices.append(slice(px,-px))
        else:
            slices.append(slice(0,None))

        slices.append(slice(0,None))

        return input[tuple(slices)]
