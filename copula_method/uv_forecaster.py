import logging
from copula_method.univariate_models import UnivariateModel

logger = logging.getLogger(__name__)

class UnivariateForecaster:
    def __init__(self, data, method):
        self.model = UnivariateModel(data, method)
        self.model.fit()

    def generate_samples(self, symbols, n_samples):
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.model.sample(symbol, n_samples)
            except Exception as e:
                logger.warning(f"Failed sampling {symbol}: {e}")
        return results