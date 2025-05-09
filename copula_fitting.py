import datetime
import pandas as pd
from model import CustomModel
from copulas.multivariate import GaussianMultivariate

class CopulaEstimator(CustomModel):
    def __init__(self, data_input: str|pd.DataFrame, split_point: float|datetime =0.8,
                 method: str = "Gaussian", features: list[str] = None):
        super().__init__(data_input, split_point)

        self.method = method
        self.target = 'ret_crsp'
        self.features = ['open_crsp', 'close_crsp', 'log_ret_lag_close_to_open'] if not features else features

        self.fitted_copula = None
        self.fitted_marginals = None

    def run(self):
        self._split()
        self.train()
        return self.fitted_copula, self.fitted_marginals

    def train(self):
        marginals = self._transform_train_data(self.train_set)
        self.fitted_copula = self._fit_copula(marginals)

    def test(self):
        pass

    def predict(self):
        pass

    def evaluate(self, true_values, predicted_values):
        pass

    def _transform_train_data(self, data: pd.DataFrame) -> dict[str, object]:
        """
        Fits marginal distributions to the provided data and transforms the data using
        Probability Integral Transform (PIT) to uniform distribution.

        :param data: The input data as a pandas DataFrame to fit and transform.
        :type data: pd.DataFrame

        :return: Dictionary containing transformed data mapped to uniform distribution.
        :rtype: dict[str, object]
        """
        from copula_marginals import fit_marginal_distributions, transform_to_uniform

        # First fit the distributions
        self.fitted_marginals = fit_marginal_distributions(data)

        # Then transform to uniform using PIT
        uniform_data = transform_to_uniform(data, self.fitted_marginals)

        return uniform_data

    def _fit_copula(self, marginals: dict[str, object]):
        # Convert the marginals dictionary to a DataFrame
        # Each column will be the PIT values for a symbol
        uniform_data = pd.DataFrame(marginals)

        # Initialize and fit the Gaussian copula
        if self.method == "Gaussian":
            gaussian_copula = GaussianMultivariate()
            gaussian_copula.fit(uniform_data)
            return gaussian_copula
        else:
            raise ValueError("Unsupported copula type")