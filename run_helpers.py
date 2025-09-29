from cgm_method.cgm_experiment import CGMExperiment
from copula_method.two_step_experiment import TwoStepExperiment

def run_experiment(model_type, config_params, fit_model=True, sample_model=True, evaluate=True):
    """
    Run a single forecasting experiment (CGM or Two-Step).

    Parameters
    ----------
    model_type : str
        Either "cgm_method" or "2step" to select the model.
    config_params : dict
        Dictionary of configuration dataclasses or parameter sets
        required by the selected model.
    fit_model : bool, default=True
        If True, fit the model before sampling.
    sample_model : bool, default=True
        If True, generate forecast samples after fitting.
    evaluate : bool, default=True
        If True and samples exist, compute evaluation metrics.

    Returns
    -------
    samples : np.ndarray | None
        Forecast samples, shape (T, S, N), or None if not generated.
    results : dict | None
        Evaluation results (metrics dict), or None if evaluation skipped.
    """
    # Select model type
    if model_type == "cgm_method":
        model = CGMExperiment(**config_params)
    elif model_type == "2step":
        model = TwoStepExperiment(**config_params)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'")

    # Fit model if requested
    if fit_model:
        model.fit()

    samples = None
    # Generate forecast samples if requested
    if sample_model:
        samples = model.sample()
        print(f"Generated {model_type} samples shape: {samples.shape}")

    results = None
    # Evaluate forecasts if requested and samples exist
    if evaluate and samples is not None:
        try:
            results = model.evaluate(samples)
        except Exception as e:
            print(f"{model_type} Evaluation skipped:", e)

    return samples, results
