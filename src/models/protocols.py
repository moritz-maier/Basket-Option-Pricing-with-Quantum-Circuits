# src/models/protocols.py
from __future__ import annotations

from typing import Protocol, Any, Optional
import numpy as np
from numpy.typing import NDArray


class Model(Protocol):
    """
    Common contract for all models (Classic + Quantum).

    The purpose of this protocol is to allow the training pipeline
    (e.g. build_pipeline, run_params) to treat all models uniformly,
    independent of the underlying framework (TensorFlow or JAX).
    """

    # Training history (structure depends on framework implementation)
    #   Classic model -> dict with numpy arrays
    #   Quantum model -> dict with jax arrays
    cost_history: Optional[Any]

    def train(
        self,
        x_train: NDArray[np.float32],
        y_train: NDArray[np.float32],  # shape (n,) or (n,1)
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        total_steps: Optional[int] = None,
        steps_per_epoch: Optional[int] = None,
    ) -> Any:
        """
        Train the model.

        Parameters
        ----------
        x_train : np.ndarray (float32)
            Training features of shape (n_samples, n_features).

        y_train : np.ndarray (float32)
            Training targets of shape (n_samples,) or (n_samples, 1).

        epochs : int | None
            Number of training epochs (for epoch-based training).

        batch_size : int | None
            batch size.

        total_steps : int | None
            Alternative training budget measured in optimizer steps

        steps_per_epoch : int | None

        Returns
        -------
        Any
            Typically returns self.
        """
        ...

    def predict(
        self,
        x: NDArray[np.float32],
        weights: Any = None,
    ) -> Any:
        """
        Predict model outputs for input features x.

        Parameters
        ----------
        x : np.ndarray (float32)
            Feature matrix of shape (n_samples, n_features).

        weights : Any, optional
            Optional explicit weights.

        Returns
        -------
        Any
            Predicted values (usually numpy array or jax array).
        """
        ...

    def get_weights(self) -> Any:
        """
        Return model parameters / weights.

        Returns
        -------
        Any
            Framework-specific weight representation
            (e.g. list of numpy arrays or jax PyTree).
        """
        ...

    def count_params(self) -> int:
        """
        Return total number of trainable parameters.

        Returns
        -------
        int
            Number of parameters.
        """
        ...