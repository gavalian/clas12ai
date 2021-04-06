from timeit import default_timer as timer
from termcolor import colored

from tensorflow.keras.models import load_model
from models.AbstractModel import AbstractModel

class AbstractKerasRegressor(AbstractModel):
    """
    Represents an abstract Keras regressor model.
    All Keras regressors in the code must inherit from this model.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None

    def build_new_model(self):
        raise NotImplementedError

    def load_model(self, path):
        """
        Loads a Keras model from an HDF5 file.

        Args:
            path: The path to the file containing the model to load
        """

        print(colored(f'\nLoading keras model from {path}\n', "green"))
        self.model = load_model(path)

    def save_model(self, path):
        """
        Saves a Keras model to an HDF5 file.

        Args:
            path: The path in which to write the file
        """

        print(colored(f'\nSaving Keras model in {path}.h5', "green"))
        self.model.save(f'{path}_full.h5', save_format="h5")
        with open(f'{path}_config.json', 'w+') as f:
            f.writelines(self.model.to_json())
        self.model.save_weights(f'{path}_weights.h5', save_format="h5")

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

        x_train = input_dict["training"]["data"]
        y_train = input_dict["training"]["labels"]
        x_test = input_dict["testing"]["data"]
        y_test = input_dict["testing"]["labels"]
        epochs = input_dict["training"]["epochs"]
        batch_size = input_dict["training"]["batch_size"]

        start = timer()
        history = self.model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=True, validation_data=(x_test, y_test))
        end = timer()
        training_time = end - start

        training_metrics = {
            "training_loss": history.history['loss'][-1],
            "training_time": training_time,
            "training_loss_history": history.history['loss'],
            "validation_loss_history": history.history['val_loss']
        }

        return training_metrics

    def test(self, input_dict) -> dict:
        """
        Tests the abstract Keras classifier.

        Args:
            input_dict: Input dictionary

        Returns:
            Dictionary containing the results from testing
        """

        x_test = input_dict["testing"]["data"]
        y_test = input_dict["testing"]["labels"]
        batch_size = input_dict["testing"]["batch_size"]
        threshold = input_dict["testing"]["threshold"]

        testing_loss = self.model.evaluate(x_test, y_test, batch_size=batch_size)
        start = timer()
        y_pred = self.model.predict(x_test)
        y_pred = (y_pred[:] >= threshold).astype(int)

        end = timer()

        testing_metrics = {
            "testing_loss": testing_loss,
            "testing_prediction_time": end - start,
            "predictions" : y_pred,
            "truth": y_test
        }

        return testing_metrics

    def predict(self, input_dict) -> dict:
        """
        Uses the abstract Keras regressor for prediction.

        Args:
            input_dict: Input dictionary

        Returns:
            Dictionary containing the predictions
        """\

        threshold = input_dict["prediction"]["threshold"]
        x = input_dict["prediction"]["data"]
        y_pred = self.model.predict(x)
        y_pred = (y_pred[:] >= threshold).astype(int)
        predictions_dict = {
            "predictions": y_pred
        }

        return predictions_dict
