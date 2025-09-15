import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class SmartScaler:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.scalers = {}
        self._choose_scaler()

    def _choose_scaler(self):
        """
        Choose a scaling method for each numeric column based on distribution shape.
        """
        numeric_cols = self.data.select_dtypes(include='number').columns

        for col in numeric_cols:
            series = self.data[col]
            if series.nunique() <= 1 or series.std() == 0:
                self.scalers[col] = None  # constant column, no scaling
                continue

            rng = series.max() - series.min()

            # Use MinMaxScaler for compact-ranged features, else StandardScaler
            if rng <= 1:
                scaler = MinMaxScaler()
            else:
                scaler = StandardScaler()

            self.scalers[col] = scaler.fit(series.values.reshape(-1, 1))

    def transform(self, data) -> pd.DataFrame:
        """
        Transform the data using the chosen scalers.
        Strictly verifies that inverse_transform(transform(x)) == x (within tolerance).
        """
        df_transformed = data.copy()
        for col, scaler in self.scalers.items():
            if scaler is not None and col in df_transformed:
                scaled = scaler.transform(df_transformed[[col]].values).flatten()
                df_transformed[col] = scaled

                # --- Strict consistency check ---
                inv = scaler.inverse_transform(scaled.reshape(-1, 1)).flatten()
                orig = data[col].values
                if not np.allclose(inv, orig, atol=1e-6, rtol=1e-6):
                    raise ValueError(
                        f"[SmartScaler] Roundtrip check failed for column '{col}'.\n"
                        f"Original vs inverse(transform) differ!"
                    )
        return df_transformed

    def inverse_transform(self, variable: str, data):
        """
        Apply inverse transformation for a single variable.
        """
        scaler = self.scalers.get(variable)
        if scaler is None:
            return data  # No transform was originally applied

        arr = np.asarray(data).reshape(-1, 1)  # Flatten to column
        inv_flat = scaler.inverse_transform(arr).flatten()

        # Return in original shape/type
        if isinstance(data, pd.DataFrame):
            return pd.DataFrame(inv_flat.reshape(np.shape(data)), index=getattr(data, 'index', None))
        else:
            return inv_flat.reshape(np.shape(data))
