from timeit import default_timer as timer
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import softmax
from termcolor import colored
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import load_model
import numpy as np
from models.AbstractModel import AbstractModel
from utils.accuracy_utils import *

def one_hot_encode(y, n_classes):
	y_one_hot = np.zeros([y.shape[0], n_classes])
	for i in range(y.shape[0]):
		y_one_hot[i, y[i]] = 1

	return y_one_hot

class AbstractKerasRegressor(AbstractModel):
    """
    Represents an abstract Keras regressor model.
    All Keras regressors in the code must inherit from this model.
    """

    def __init__(self, input_dict):
        super().__init__(input_dict)
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

        print(colored(f'\nSaving Keras model in {path}.p', "green"))
        self.model.save(f'{path}.p')

    def preprocess_input(self, input_dict):
        raise NotImplementedError
    
    def compute_accuracy_metrics(self, input_dict) -> dict:
        conf_matrix = input_dict["confusion_matrix"]
        total_test_samples = input_dict["total_test_samples"]
        y_pred = input_dict["y_pred"]
        y_pred_proba = input_dict["y_pred_proba"]
        y_test_segmented = input_dict["y_test_segmented"]
        X_test_segmented = input_dict["X_test_segmented"]

        accuracy_A1 = get_accuracy_A1(conf_matrix, total_test_samples)
        accuracy_Ac = get_accuracy_Ac(conf_matrix, y_pred, y_test_segmented)
        accuracy_Ah = get_accuracy_Ah(y_pred_proba, y_test_segmented)
        # accuracy_new_Ah = get_accuracy_new_Ah(y_pred_proba, y_test_segmented, X_test_segmented)
        accuracy_Af = get_accuracy_Af(conf_matrix, total_test_samples)

        return {
            "accuracy_A1": accuracy_A1,
            "accuracy_Ac": accuracy_Ac,
            "accuracy_Ah": accuracy_Ah,
            "accuracy_new_Ah": accuracy_new_Ah,
            "accuracy_Af": accuracy_Af
        }
    def train(self, input_dict) -> dict:
        """
        Trains the abstract Keras regressor.

        Args:
            input_dict: Input dictionary

        Returns:
            Dictionary containing the results from training
        """

        X_train = input_dict["training"]["data"]
        labels = input_dict["training"]["labels"]
        y_train = input_dict["training"]["truth"]
        epochs = input_dict["training"]["epochs"]
        batch_size = input_dict["training"]["batch_size"]

        start = timer()
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        end = timer()
        training_time = end - start

        training_loss = self.model.evaluate(X_train, y_train, batch_size=batch_size)
        y_pred = self.model.predict(X_train) * 112
        diff = np.absolute(y_train[:,-12:] * 112 - y_pred[:,-12:])
        y_pred = (np.mean(diff, axis=1)<=4).astype(int)
        eq = np.equal(y_pred, labels.astype(int))
        accuracy_training = np.sum(eq)/eq.shape[0]
        if input_dict["features"] == 24:
            training_loss = training_loss * 112

        return {
            "training_loss": training_loss,
            "training_time": training_time,
            "accuracy_training": accuracy_training
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
        labels = input_dict["testing"]["labels"]
        print(labels)
        y_test = input_dict["testing"]["truth"]
        batch_size = input_dict["testing"]["batch_size"]

        start = timer()
        testing_predictions = self.model.predict(X_test)
        y_pred = testing_predictions
        end = timer()
        diff = np.absolute(y_test[:,-12:] * 112 - y_pred[:,-12:] * 112)
        y_pred = (np.mean(diff, axis=1)<=4).astype(int)
        y_pred_one_hot = one_hot_encode(y_pred, 2)


        eq = np.equal(y_pred, labels.astype(int))
        testing_prediction_time = end - start
        accuracy_testing = np.sum(eq)/eq.shape[0]
        given_data = None
        testing_loss = None
        if input_dict["features"] == 24:
            testing_loss = mean_absolute_error(testing_predictions.reshape(-1,24)*112,y_test.reshape(-1,24)*112)

        elif input_dict["features"] == 4:
            testing_loss = mean_absolute_error(testing_predictions.reshape(-1,4),y_test.reshape(-1,4))


        y_pred_proba = y_pred_one_hot* (100 - np.mean(diff, axis=1).reshape(-1,1))
        
        # Compute confusion matrix
        conf_matrix = confusion_matrix(labels, y_pred)

        # Compute accuracy metrics
        input_dict_accuracy_metrics = {
            "confusion_matrix": conf_matrix, 
            "y_pred": y_pred, 
            "y_pred_proba": y_pred_proba,
            "total_test_samples": input_dict["total_test_samples"], 
            "y_test_segmented": input_dict["testing_segmented"]["labels"],
            "X_test_segmented": input_dict["testing_segmented"]["data"]
        }

        accuracy_metrics_dict = self.compute_accuracy_metrics(input_dict_accuracy_metrics)

        output_dict = {
            "accuracy_testing": accuracy_testing, 
            "testing_prediction_time": testing_prediction_time, 
            "confusion_matrix": conf_matrix,
            "prediction_matrix": y_pred_proba
        }
        output_dict.update(accuracy_metrics_dict)

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

    def compute_accuracy_metrics(self, input_dict) -> dict:
        conf_matrix = input_dict["confusion_matrix"]
        total_test_samples = input_dict["total_test_samples"]
        y_pred = input_dict["y_pred"]
        y_pred_proba = input_dict["y_pred_proba"]
        y_test_segmented = input_dict["y_test_segmented"]

        accuracy_A1 = get_accuracy_A1(conf_matrix, total_test_samples)
        accuracy_Ac = get_accuracy_Ac(conf_matrix, y_pred, y_test_segmented)
        accuracy_Ah = get_accuracy_Ah(y_pred_proba, y_test_segmented)
        accuracy_Af = get_accuracy_Af(conf_matrix, total_test_samples)

        return {
            "accuracy_A1": accuracy_A1,
            "accuracy_Ac": accuracy_Ac,
            "accuracy_Ah": accuracy_Ah,
            "accuracy_Af": accuracy_Af
        }