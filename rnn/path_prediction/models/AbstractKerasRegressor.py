from timeit import default_timer as timer
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import softmax
from termcolor import colored
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import load_model
import numpy as np
from common.models.AbstractModel import AbstractModel


class AbstractKerasRegressor(AbstractModel):
    """
    Represents an abstract Keras regressor model.
    All Keras regressors in the code must inherit from this model.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None

    def build_new_model(self, input_dict):
        raise NotImplementedError

    def load_model(self, path):
        """
        Loads a Keras model from an HDF5 file.

        Args:
            path: The path to the file containing the model to load
        """

        print(colored(f'\nLoading Keras model from {path}\n', "green"))
        self.model = load_model(path)

    def save_model(self, path):
        """
        Saves a Keras model to an HDF5 file.

        Args:
            path: The path in which to write the file
        """

        print(colored(f'\nSaving Keras model in {path}.h5', "green"))
        self.model.save(f'{path}.h5')

    def preprocess_input(self, input_dict):
        raise NotImplementedError

    def train(self, input_dict) -> dict:
        """
        Trains the abstract Keras regressor.

        Args:
            input_dict: Input dictionary

        Returns:
            Dictionary containing the results from training
        """

        X_train = input_dict["training"]["data"]
        y_train = input_dict["training"]["labels"]
        epochs = input_dict["training"]["epochs"]
        batch_size = input_dict["training"]["batch_size"]

        start = timer()
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        end = timer()
        training_time = end - start

        training_loss = self.model.evaluate(X_train, y_train, batch_size=batch_size)
        
        if input_dict["features"] == 24:
            training_loss = training_loss * 112

        return {
            "training_loss": training_loss,
            "training_time": training_time,
        }

    def test(self, input_dict) -> dict:
        """
        Tests the abstract Keras regressor.

        Args:
            input_dict: Input dictionary

        Returns:
            Dictionary containing the results from testing
        """

        X_test = input_dict["testing"]["data"]
        y_test = input_dict["testing"]["labels"]
        batch_size = input_dict["testing"]["batch_size"]

        start = timer()
        testing_predictions = self.model.predict(X_test)
        end = timer()

        testing_prediction_time = end - start
        given_data = None
        testing_loss = None
        if input_dict["features"] == 24:
            testing_loss = mean_absolute_error(testing_predictions.reshape(-1,24)*112,y_test.reshape(-1,24)*112)
            given_data = np.hstack((X_test.reshape(-1,24),y_test[:,-12:].reshape(-1,12)))
        elif input_dict["features"] == 4:
            testing_loss = mean_absolute_error(testing_predictions.reshape(-1,4),y_test.reshape(-1,4))
            given_data = np.hstack((X_test.reshape(-1,4),y_test[:,-2:].reshape(-1,2)))

        output_dict = {
            "given_data" : given_data,
            "testing_predictions" : testing_predictions,
            "testing_loss": testing_loss,
            "testing_prediction_time": testing_prediction_time
        }

        return output_dict

    def predict(self, input_dict) -> dict:
        """
        Uses the abstract Keras regressor for prediction.

        Args:
            input_dict: Input dictionary

        Returns:
            Dictionary containing the results from testing
        """

        x = input_dict["prediction"]["data"]
        output_dict = {
            "predictions": self.model.predict(x)
        }

        return output_dict