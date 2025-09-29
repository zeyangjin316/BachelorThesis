import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class SmartScaler:
    """
    Utility class that automatically selects and applies feature scaling
    (MinMax or Standard) per numeric column, with strict round-trip checks.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Parameters
        ----------
        data : pd.DataFrame
            Input data used to decide scaling strategy and fit scalers.
        """
        self.data = data
        self.scalers = {}
        self._choose_scaler()

    def _choose_scaler(self):
        """
        Decide a scaling method per numeric column:
        - MinMaxScaler if range <= 1
        - StandardScaler otherwise
        - None if constant column
        """
        numeric_cols = self.data.select_dtypes(include="number").columns

        for col in numeric_cols:
            series = self.data[col]

            # Skip constant columns
            if series.nunique() <= 1 or series.std() == 0:
                self.scalers[col] = None
                continue

            rng = series.max() - series.min()

            # Choose scaling method
            if rng <= 1:
                scaler = MinMaxScaler()
            else:
                scaler = StandardScaler()

            # Fit scaler on column values
            self.scalers[col] = scaler.fit(series.values.reshape(-1, 1))

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted scalers to a DataFrame.

        Performs a strict round-trip check:
        inverse_transform(transform(x)) ≈ x (within tolerance).

        Parameters
        ----------
        data : pd.DataFrame
            Data to transform.

        Returns
        -------
        pd.DataFrame
            Scaled copy of the input.
        """
        df_transformed = data.copy()

        for col, scaler in self.scalers.items():
            if scaler is not None and col in df_transformed:
                # Apply scaling
                scaled = scaler.transform(df_transformed[[col]].values).flatten()
                df_transformed[col] = scaled

                # Consistency check
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
            Name of the column/variable to invert.
        data : array-like or pd.DataFrame
            Scaled data to be inverse-transformed.

        Returns
        -------
        np.ndarray | pd.DataFrame
            Data restored to original scale.
        """
        scaler = self.scalers.get(variable)
        if scaler is None:
            return data  # no transform applied

        arr = np.asarray(data).reshape(-1, 1)
        inv_flat = scaler.inverse_transform(arr).flatten()

        # Return in original shape/type
        if isinstance(data, pd.DataFrame):
            return pd.DataFrame(
                inv_flat.reshape(np.shape(data)),
                index=getattr(data, "index", None)
            )
        else:
            return inv_flat.reshape(np.shape(data))
