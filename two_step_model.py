import datetime

from model import CustomModel
import pandas as pd

class TwoStepModel(CustomModel):
    def __init__(self, file_path: str, univariate: str, copula: str, split_point: float|datetime =0.8):
        super().__init__(file_path, split_point)
        self.univariate_type = univariate
        self.copula_type = copula

    def split(self):
        pass

    def train(self):
        self._train_univariate(self.univariate_type)
        self._train_copula(self.copula_type)
        pass

    def test(self):
        pass

    def predict(self):
        pass

    def evaluate(self, true_values, predicted_values):
        pass

    def _train_univariate(self, method: str):
        pass

    def _train_copula(self, method: str = "Gaussian"):
        from copula_fitting import CopulaEstimator
        copula_estimator = CopulaEstimator(self.data, self.split_point, file_path='data_for_kit.csv')
        self.copula_type, self.marginals = copula_estimator.run()

