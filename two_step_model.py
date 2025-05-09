import datetime

from model import CustomModel
import pandas as pd

class TwoStepModel(CustomModel):
    def __init__(self, file_path: str = "data_for_kit.csv", split_point: float|datetime =0.8,
                 univariate_type: str = "ARMAGARCH", copula_type: str ="Gaussian"):
        super().__init__(file_path,  split_point)
        self.univariate_type = univariate_type
        self.copula_type = copula_type

    def train(self):
        self._split()
        self._train_univariate()
        self._train_copula()
        pass

    def test(self):
        pass

    def predict(self):
        pass

    def evaluate(self, true_values, predicted_values):
        pass

    def _train_univariate(self):
        from univariate_models import UnivariateModel
        univariate_model = UnivariateModel(self.data, split_point=0.8, method=self.univariate_type,)

        pass

    def _train_copula(self):
        from copula_fitting import CopulaEstimator
        copula_estimator = CopulaEstimator(self.data, split_point=0.8, method=self.copula_type,
                                           features=['open_crsp', 'close_crsp', 'log_ret_lag_close_to_open'])
        self.fitted_copula, self.copula_marginals = copula_estimator.run()

