from sklearn.ensemble import ExtraTreesClassifier
from termcolor import colored

from models.AbstractScikitLearnClassifier import AbstractScikitLearnClassifier


class ExtraTreesModel(AbstractScikitLearnClassifier):
    """
    An ExtraTrees scikit-learn classifier model.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None

    def build_new_model(self):
        self.model = ExtraTreesClassifier(n_estimators=300, criterion='entropy', max_features=None,
                                          n_jobs=-1, verbose=1, random_state=3333)

    def train(self, input_dict) -> dict:
        print(colored("Training ExtraTrees model...", "green"))
        return super().train(input_dict)

    def test(self, input_dict) -> dict:
        print(colored("Testing ExtraTrees model...", "green"))
        return super().test(input_dict)

    def predict(self, input_dict) -> dict:
        print(colored("Setting up ET model for prediction...", "green"))
        return super().predict(input_dict)
