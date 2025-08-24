import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

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

    def transform(self) -> pd.DataFrame:
        """
        Transform the full_df using the chosen scalers.
        """
        df_transformed = self.data.copy()
        for col, scaler in self.scalers.items():
            if scaler is not None and col in df_transformed:
                df_transformed[col] = scaler.transform(df_transformed[[col]].values).flatten()
        return df_transformed

    def inverse_transform(self, variable: str, data):

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
