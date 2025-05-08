from forecasting import Forecast
import pandas as pd

class TwoStepModel(Forecast):
    def __init__(self, data: pd.DataFrame, univariate: str, copula: str):
        super().__init__(data)
        self.univariate = univariate
        self.copula = copula
        self.data = data

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
        copula_estimator = CopulaEstimator(self.data)
        self.copula, self.marginals = copula_estimator.run()

