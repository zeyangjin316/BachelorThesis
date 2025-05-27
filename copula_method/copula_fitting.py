import pandas as pd
import logging
from copulas.multivariate import GaussianMultivariate

logger = logging.getLogger(__name__)


class CopulaEstimator():
    def __init__(self, method: str = "Gaussian", features: list[str] = None):
        logger.info(f"Initializing copula fitting model with {method} copula")
        self.method = method
        self.features = ['open_crsp', 'close_crsp', 'log_ret_lag_close_to_open'] if not features else features
        self.fitted_copula = None
        logger.info("Copula fitting model initialized")

    def fit(self, transformed_data: pd.DataFrame):
        """
        Fit a copula directly from preprocessed Gaussianized PIT values (Z_d,h).

        Args:
            transformed_data: DataFrame (rows = days, columns = symbols), values = Z-scores
        """
        logger.info("Fitting copula from transformed (Gaussianized) data")

        if self.method == "Gaussian":
            copula = GaussianMultivariate()
            copula.fit(transformed_data)
            self.fitted_copula = copula
            logger.info("Gaussian copula fitted successfully")
        else:
            raise ValueError(f"Copula type '{self.method}' is not supported yet.")