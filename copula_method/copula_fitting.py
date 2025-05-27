import pandas as pd
import logging
from copulas.multivariate import GaussianMultivariate

logger = logging.getLogger(__name__)


class CopulaEstimator():
    def __init__(self, method: str = "Gaussian"):
        self.method = method
        self.fitted_copula = None

    def fit(self, transformed_data: pd.DataFrame):
        if self.method == "Gaussian":
            copula = GaussianMultivariate()
            copula.fit(transformed_data)
            self.fitted_copula = copula
        else:
            raise ValueError(f"Copula type '{self.method}' is not supported yet.")