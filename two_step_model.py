import datetime

from model import CustomModel
import pandas as pd

class TwoStepModel(CustomModel):
    def __init__(self, data: pd.DataFrame, univariate: str, copula: str, split_point: float|datetime =0.8):
        super().__init__(data, split_point)
        self.univariate = univariate
        self.copula = copula
        self.data = data

    def split(self):
        pass

    def train(self):
        self._train_univariate(self.univariate)
        self._train_copula(self.copula)
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
        self.copula, self.marginals = copula_estimator.run()

