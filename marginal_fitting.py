import pandas as pd
import numpy as np
from scipy import stats


def fit_marginal_distributions(df) -> dict[str, stats.t]:
    """
    Fit t-distributions to each stock's returns and return the distributions.

    Parameters:
    df (pandas.DataFrame): DataFrame containing 'sym_root' and 'ret_crsp' columns

    Returns:
    dict[str, stats.t]: Dictionary mapping stock symbols to their fitted scipy.stats.t objects
    """
    # Get unique symbols in the same order they appear
    symbols = df['sym_root'].unique()

    # Dictionary to store fitted distributions
    fitted_dists = {}

    for symbol in symbols:
        # Get returns for this symbol
        returns = df[df['sym_root'] == symbol]['ret_crsp'].values

        # Fit t distribution using MLE
        df_t, loc_t, scale_t = stats.t.fit(returns)

        # Create and store the fitted distribution object
        t_dist = stats.t(df=df_t, loc=loc_t, scale=scale_t)
        fitted_dists[symbol] = t_dist

    return fitted_dists


def transform_to_uniform(df, fitted_distributions: dict[str, stats.t]) -> dict[str, np.ndarray]:
    """
    Transform the data to uniform using the fitted distributions (PIT transform).

    Parameters:
    df (pandas.DataFrame): DataFrame containing 'sym_root' and 'ret_crsp' columns
    fitted_distributions (dict[str, stats.t]): Dictionary of fitted t-distributions per symbol

    Returns:
    dict[str, np.ndarray]: Dictionary mapping stock symbols to their PIT-transformed values
    """
    pit_values = {}

    for symbol, t_dist in fitted_distributions.items():
        # Get returns for this symbol
        returns = df[df['sym_root'] == symbol]['ret_crsp'].values
        
        # Apply the CDF to get uniform values (PIT)
        pit_values[symbol] = t_dist.cdf(returns)

    return pit_values