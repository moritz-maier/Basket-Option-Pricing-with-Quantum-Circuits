from sklearn.preprocessing import MinMaxScaler
import jax.numpy as jnp


class DataScaler:
    """
    Utility class for feature and target scaling.

    - Input features (X) are scaled to [0, π]

    - Targets (y) are scaled to [-1, 1]
     matches quantum model output ranges.

    The class supports:
        - Joint or separate scaling of X and y
        - Optional fitting during transformation
        - Inverse transformation back to original scale
    """

    def __init__(self):
        # Separate scalers for inputs and targets
        self.x_scaler = None
        self.y_scaler = None

    def transform(self, x_raw=None, y_raw=None, fit=False):
        """
        Scale input features and/or targets.

        Parameters
        ----------
        x_raw : array-like, optional
            Raw input features.
        y_raw : array-like, optional
            Raw target values.
        fit : bool
            If True, fit the scaler before transforming.

        Returns
        -------
        x_scaled : jnp.ndarray or None
        y_scaled : jnp.ndarray or None
        """
        x_scaled = None
        y_scaled = None

        # ----- Scale features -----
        if x_raw is not None:
            if fit or self.x_scaler is None:
                # Map features to [0, π]
                self.x_scaler = MinMaxScaler(feature_range=(0, jnp.pi))
                x_scaled = self.x_scaler.fit_transform(x_raw)
            else:
                x_scaled = self.x_scaler.transform(x_raw)

            # Convert to JAX array for compatibility with JAX models
            x_scaled = jnp.array(x_scaled)

        # ----- Scale targets -----
        if y_raw is not None:
            if fit or self.y_scaler is None:
                # Map targets to [-1, 1]
                self.y_scaler = MinMaxScaler(feature_range=(-1, 1))
                y_scaled = (
                    self.y_scaler
                    .fit_transform(y_raw.reshape(-1, 1))
                    .flatten()
                )
            else:
                y_scaled = (
                    self.y_scaler
                    .transform(y_raw.reshape(-1, 1))
                    .flatten()
                )

            # Convert to JAX array
            y_scaled = jnp.array(y_scaled)

        return x_scaled, y_scaled

    def inverse_transform(self, x_scaled=None, y_scaled=None):
        """
        Inverse transformation from scaled space back to original scale.

        Parameters
        ----------
        x_scaled : array-like, optional
            Scaled input features.
        y_scaled : array-like, optional
            Scaled targets.

        Returns
        -------
        x_inv : jnp.ndarray or None
        y_inv : jnp.ndarray or None
        """
        x_inv = None
        y_inv = None

        # ----- Inverse transform features -----
        if x_scaled is not None and self.x_scaler is not None:
            x_inv = jnp.array(
                self.x_scaler.inverse_transform(x_scaled)
            )

        # ----- Inverse transform targets -----
        if y_scaled is not None and self.y_scaler is not None:
            y_inv = jnp.array(
                self.y_scaler
                .inverse_transform(y_scaled.reshape(-1, 1))
                .flatten()
            )

        return x_inv, y_inv