import pandas as pd
from copula_method.copula_fitting import CopulaEstimator

def test_fit_copula_from_transformed_data():
    df = pd.DataFrame({
        "AAPL": [0.1, -0.3, 0.2],
        "TSLA": [0.5, -0.1, 0.0]
    })
    estimator = CopulaEstimator(method="Gaussian")
    estimator.fit(df)
    assert estimator.fitted_copula is not None