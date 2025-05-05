import pandas as pd
import numpy as np
from scipy import stats


def fit_marginal_distributions(df, pit=True) -> dict[str, object]:
    """
    Fit t-distributions to each stock's returns and return the distributions in a dictionary.

    Parameters:
    df (pandas.DataFrame): DataFrame containing 'sym_root' and 'ret_crsp' columns

    Returns:
    dict: Dictionary mapping stock symbols to their fitted scipy.stats.t objects
    """
    # Get unique symbols in the same order they appear
    symbols = df['sym_root'].unique()

    # Dictionary to store fitted distributions
    fitted_dists = {}
    pit_values = {}

    for symbol in symbols:
        # Get returns for this symbol
        returns = df[df['sym_root'] == symbol]['ret_crsp'].values

        # Fit t distribution using MLE
        df_t, loc_t, scale_t = stats.t.fit(returns)

        # Create and store the fitted distribution object
        t_dist = stats.t(df=df_t, loc=loc_t, scale=scale_t)
        fitted_dists[symbol] = t_dist

        # If PIT is True, transform returns to uniform values
        if pit:
            # Apply the CDF to get uniform values (PIT)
            pit_values[symbol] = t_dist.cdf(returns)

    # Return either the fitted distributions or the PIT values
    if pit:
        return pit_values
    else:
        return fitted_dists