import numpy as np
import pandas as pd
import logging
from scipy import stats

logger = logging.getLogger(__name__)

def fit_marginal_distributions(df) -> dict[str, stats.rv_continuous]:
    logger.info("Fitting marginal distributions for each symbol in training data")
    # Get unique symbols
    symbols = df['sym_root'].unique()

    # Dictionary to store fitted distributions
    fitted_dists = {}

    for symbol in symbols:
        # Get returns for this symbol
        returns = df[df['sym_root'] == symbol]['ret_crsp'].dropna().values

        # Only fit if we have enough data points
        if len(returns) > 0:
            # Fit t distribution using MLE
            logger.debug(f"Fitting t distribution for symbol {symbol}")
            df_t, loc_t, scale_t = stats.t.fit(returns)
            logger.debug(f"Fitted t distribution parameters: df={df_t}, loc={loc_t}, scale={scale_t}")

            # Create and store the fitted distribution object
            t_dist = stats.t(df=df_t, loc=loc_t, scale=scale_t)
            fitted_dists[symbol] = t_dist

    logger.info("Marginal distributions fitted successfully")
    return fitted_dists


def transform_to_uniform(df, fitted_distributions: dict[str, stats.rv_continuous]) -> dict[str, np.ndarray]:
    logger.info("Transforming training data to uniform distribution")

    # Create a list to store transformed data for each symbol
    transformed_data = []

    # Get the common dates for all symbols
    dates = df['date'].unique()

    for symbol, t_dist in fitted_distributions.items():
        symbol_data = df[df['sym_root'] == symbol].set_index('date')

        # Create a series with the transformed values
        transformed_values = pd.Series(
            t_dist.cdf(symbol_data['ret_crsp']),
            index=symbol_data.index,
            name=symbol
        )
        transformed_data.append(transformed_values)

    # Combine all transformed series into a single DataFrame
    result = pd.concat(transformed_data, axis=1)

    # Ensure all symbols have values for all dates by forward filling
    result = result.reindex(dates).ffill()

    logger.info("Training data transformed successfully")
    return result
