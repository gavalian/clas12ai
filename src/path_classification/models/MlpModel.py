from sklearn.neural_network import MLPClassifier
from termcolor import colored

from models.AbstractScikitLearnClassifier import AbstractScikitLearnClassifier


class MlpModel(AbstractScikitLearnClassifier):
    """
    A multi-layer perceptron scikit-learn classifier.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None

    def build_new_model(self):
        self.model = MLPClassifier((64, 64, 64), solver="adam", batch_size=32,
                                   learning_rate="adaptive", max_iter=160, verbose=True)

    def train(self, input_dict) -> dict:
        print(colored("Training MLP model...", "green"))
        return super().train(input_dict)

    def test(self, input_dict) -> dict:
        print(colored("Testing MLP model...", "green"))
        return super().test(input_dict)

    def predict(self, input_dict) -> dict:
        print(colored("Setting up MLP model for prediction...", "green"))
        return super().predict(input_dict)
