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

    def __init__(self, input_dict):
        super().__init__(input_dict)
        self.preprocess_input(input_dict)
        self.model = None
        self.num_features = input_dict["features"]

    def build_new_model(self):


        model = Sequential()

        model.add(GRU(40, input_shape=(self.num_features, 1), return_sequences=True))
        model.add(GRU(240))
        model.add(Dense(self.num_features))
        model.compile(loss='mean_absolute_error', optimizer=RMSprop())
        model.summary()

        self.model = model

    def preprocess_input(self, input_dict):
        if "training" in input_dict:
            X_train = np.array( input_dict["training"]["data"] )

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
            input_dict["training"]["truth"] = y_train_new
            input_dict["training"]["labels"] = (input_dict["training"]["labels"] != 0).astype(int)


        if "testing" in input_dict:
            X_test = np.array(input_dict["testing"]["data"])

            X_test_new = None
            y_test_new = None
            
            print("X_test", X_test)
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
            input_dict["testing"]["truth"] = y_test_new
            input_dict["testing"]["labels"] = (input_dict["testing"]["labels"] != 0).astype(int)

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
