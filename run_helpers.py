from cgm_method.cgm_experiment import CGMExperiment
from copula_method.two_step_experiment import TwoStepExperiment

def run_experiment(model_type, params, fit_model=True, sample_model=True, evaluate=True):
    if model_type == "cgm_method":
        model = CGMExperiment(**params)

    elif model_type == "2step":
        model = TwoStepExperiment(**params)

    else:
        raise ValueError(f"Unknown model_type '{model_type}'")

    if fit_model:
        model.fit()

    samples = None
    if sample_model:
        samples = model.sample()
        print(f"Generated {model_type} samples shape: {samples.shape}")

    results = None
    if evaluate and samples is not None:
        try:
            results = model.evaluate(samples)
        except Exception as e:
            print(f"{model_type} Evaluation skipped:", e)

    return samples, results