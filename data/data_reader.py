import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class SmartScaler:
    """
    Utility class that selects a per-column scaler (MinMax or Standard)
    for numeric features and enforces a strict round-trip check.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Parameters
        ----------
        data : pd.DataFrame
            Data used to decide scaling strategy and fit scalers.
        """
        self.data = data
        self.scalers = {}
        self._choose_scaler()

    def _choose_scaler(self):
        """
        Decide and fit a scaler per numeric column.

        Rules
        -----
        - Constant columns → no scaler (None).
        - Range <= 1 → MinMaxScaler.
        - Otherwise → StandardScaler.
        """
        numeric_cols = self.data.select_dtypes(include='number').columns

        for col in numeric_cols:
            series = self.data[col]
            if series.nunique() <= 1 or series.std() == 0:
                self.scalers[col] = None
                continue

            rng = series.max() - series.min()
            scaler = MinMaxScaler() if rng <= 1 else StandardScaler()
            self.scalers[col] = scaler.fit(series.values.reshape(-1, 1))

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted scalers to a DataFrame and verify round-trip consistency.

        Parameters
        ----------
        data : pd.DataFrame
            Input to transform.

        Returns
        -------
        pd.DataFrame
            Transformed copy of `data`.

        Raises
        ------
        ValueError
            If inverse_transform(transform(x)) deviates beyond tolerance.
        """
        df_transformed = data.copy()
        for col, scaler in self.scalers.items():
            if scaler is not None and col in df_transformed:
                scaled = scaler.transform(df_transformed[[col]].values).flatten()
                df_transformed[col] = scaled

                # strict round-trip check
                inv = scaler.inverse_transform(scaled.reshape(-1, 1)).flatten()
                orig = data[col].values
                if not np.allclose(inv, orig, atol=1e-6, rtol=1e-6):
                    raise ValueError(
                        f"[SmartScaler] Roundtrip check failed for column '{col}'. "
                        "Original vs inverse(transform) differ!"
                    )
        return df_transformed

    def inverse_transform(self, variable: str, data):
        """
        Undo scaling for a single variable.

        Parameters
        ----------
        variable : str
            Column name whose scaler should be used.
        data : array-like or pd.DataFrame
            Values to inverse-transform.

        Returns
        -------
        np.ndarray | pd.DataFrame
            Data restored to original scale and shape.
        """
        scaler = self.scalers.get(variable)
        if scaler is None:
            return data  # no scaling was applied

        arr = np.asarray(data).reshape(-1, 1)
        inv_flat = scaler.inverse_transform(arr).flatten()

        if isinstance(data, pd.DataFrame):
            return pd.DataFrame(inv_flat.reshape(np.shape(data)),
                                index=getattr(data, 'index', None))
        else:
            return inv_flat.reshape(np.shape(data))
