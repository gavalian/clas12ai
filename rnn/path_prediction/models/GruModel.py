import numpy as np
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import RMSprop

from termcolor import colored

from models.AbstractKerasRegressor import AbstractKerasRegressor

from utils.rnn_input_utils import *


class GruModel(AbstractKerasRegressor):
    """
    An RNN-GRU Keras regressor.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "input_dict" in kwargs:
            self.preprocess_input(kwargs["input_dict"])
        self.model = None

    def build_new_model(self, input_dict):


        model = Sequential()

        model.add(GRU(40, input_shape=(input_dict["features"], 1), return_sequences=True))
        model.add(GRU(240))
        model.add(Dense(input_dict["features"]))
        model.compile(loss='mean_absolute_error', optimizer=RMSprop())
        model.summary()

        self.model = model

    def preprocess_input(self, input_dict):
        if "training" in input_dict:
            X_train = input_dict["training"]["data"]

            X_train_new = None
            y_train_new = None
            
            if X_train.shape[1] == 36:
                X_train_new, y_train_new = create_rnn_dataset_layer_major_2(remove_zeros(X_train))

                # Normalize dataset
                X_train_new = X_train_new / 112
                y_train_new = y_train_new / 112
                input_dict["features"] = 24

            elif X_train.shape[1] == 6:
                X_train_new, y_train_new = create_rnn_dataset_layer_major_6(X_train)
                input_dict["features"] = 4

            input_dict["training"]["data"] = X_train_new
            input_dict["training"]["labels"] = y_train_new


        if "testing" in input_dict:
            X_test = input_dict["testing"]["data"]

            X_test_new = None
            y_test_new = None
            
            if X_test.shape[1] == 36:
                X_test_new, y_test_new = create_rnn_dataset_layer_major_2(remove_zeros(X_test))
                input_dict["features"] = 24

                # Normalize dataset
                X_test_new = X_test_new / 112
                y_test_new = y_test_new / 112

            elif X_test.shape[1] == 6:
                X_test_new, y_test_new = create_rnn_dataset_layer_major_6(X_test)
                input_dict["features"] = 4

            input_dict["testing"]["data"] = X_test_new
            input_dict["testing"]["labels"] = y_test_new

        if "prediction" in input_dict:
            X_pred = remove_zeros(input_dict["prediction"]["data"])
            X_pred_new = X_pred / 112

            input_dict["prediction"]["data"] = (X_pred_new)

    def train(self, input_dict) -> dict:
        print(colored("Training GRU model...", "green"))
        return super().train(input_dict)

    def test(self, input_dict) -> dict:
        print(colored("Testing GRU model...", "green"))
        return super().test(input_dict)

    def predict(self, input_dict) -> dict:
        print(colored("Setting up GRU model for prediction...", "green"))
        return super().predict(input_dict)
