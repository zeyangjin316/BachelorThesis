import pytest
from copula_method.two_step_experiment import TwoStepExperiment

def test_model_initialization():
    model = TwoStepExperiment(split_point=0.8)
    assert model.train_data is not None
    assert model.test_data is not None

def test_fit_univariate_models():
    model = TwoStepExperiment()
    model._fit_univariate_models()
    assert hasattr(model, "univariate_models")

def test_compute_gaussian_inputs():
    model = TwoStepExperiment()
    model._fit_univariate_models()
    model._predict_univariate()
    days = sorted(model.test_data['date'].unique())[:5]
    z = model._compute_gaussian_copula_inputs(days)
    assert isinstance(z, pd.DataFrame)
    assert not z.empty