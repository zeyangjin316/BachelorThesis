import pandas as pd
import logging
from datetime import datetime
from typing import Union
from model import CustomModel
from copulas.multivariate import GaussianMultivariate
from copula_method.copula_marginals import fit_marginal_distributions, transform_to_uniform

logger = logging.getLogger(__name__)

class CopulaEstimator(CustomModel):
    def __init__(self, data_input: str|pd.DataFrame, split_point: Union[float, datetime] = 0.8,
                 method: str = "Gaussian", features: list[str] = None):
        logger.info(f"Initializing copula fitting model with {method} copula")
        super().__init__(data_input, split_point)

        self.method = method
        self.target = 'ret_crsp'
        self.features = ['open_crsp', 'close_crsp', 'log_ret_lag_close_to_open'] if not features else features
        self.fitted_copula = None
        self.fitted_marginals = None
        logger.info("Copula fitting model initialized")

    def _build(self):
        self._split()
        self.fit()
        return self.fitted_copula, self.fitted_marginals

    def fit(self):
        logger.info("Starting training step for copula fitting model")
        uniform_data = self._transform_train_data(self.train_set)
        self.fitted_copula = self._fit_copula(uniform_data)
        logger.info("Finished training step for copula fitting model")

    def test(self):
        pass

    def predict(self):
        pass

    def evaluate(self, true_values, predicted_values):
        pass

    def _transform_train_data(self, data: pd.DataFrame) -> dict[str, object]:
        logger.info("Starting to transform training data")
        self.fitted_marginals = fit_marginal_distributions(data)
        uniform_data = transform_to_uniform(data, self.fitted_marginals)
        logger.info("Training data transformed successfully")

        return uniform_data

    def _fit_copula(self, marginals: dict[str, object]):
        logger.info("Starting to fit copula")
        uniform_data = pd.DataFrame(marginals)
        logger.debug("Uniform data: %s", uniform_data.info())

        # Initialize and fit the Gaussian copula
        if self.method == "Gaussian":
            logger.info("Fitting Gaussian copula")
            gaussian_copula = GaussianMultivariate()
            gaussian_copula.fit(uniform_data)
            logger.info("Gaussian copula fitted successfully")
            return gaussian_copula
        else:
            raise ValueError("Unsupported copula type")