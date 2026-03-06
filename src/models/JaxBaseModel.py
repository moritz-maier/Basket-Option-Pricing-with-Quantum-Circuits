from abc import ABC
import math

import jax
import jax.numpy as jnp
import optax

from src.models.DataScaler import DataScaler
from src.models.utils import _auto_batch_size, compute_log_every


class JaxBaseModel(ABC):
    """
    Base class for JAX-based models with a common training and inference interface.

    This class provides:
      - A standard training loop using Optax optimizers
      - A mean squared error (MSE) loss
      - JIT-compiled cost and gradient update functions
      - Optional input/target scaling via a DataScaler
      - A vectorized batch prediction wrapper via jax.vmap

    Subclasses must provide:
      - `predict_fn(weights, x)` which predicts a scalar output for a single input row x
        (e.g., a compiled quantum circuit evaluation).
    """

    def __init__(
        self,
        key,
        weights=None,
        scaler=None,
        output_dir: str = "results",
        learning_rate: float = 0.001,
    ):
        # Output directory is kept for compatibility with result saving logic
        self.output_dir = output_dir

        # JAX PRNG key for reproducible shuffling / sampling
        self.key = key

        # Trainable parameters
        self.weights = weights

        # Will be filled after training
        self.cost_history = None

        # Build JIT-compiled functions for loss + updates
        self.cost_fn = self.make_cost_fn()
        self.update_step_fn = self.make_update_step_fn()

        # Optimizer (Adam by default)
        self.optimizer = optax.adam(learning_rate=learning_rate)

        # Feature/target scaling (kept generic via DataScaler)
        self.scaler = DataScaler() if scaler is None else scaler

    def fit_scaler(self, x_train, y_train):
        """
        Fit the scaler on training data.

        DataScaler is expected to support a transform(..., fit=True) interface.
        """
        self.scaler.transform(x_train, y_train, fit=True)

    @staticmethod
    def square_loss(targets, predictions):
        """Mean squared error (MSE) loss."""
        return jnp.mean((targets - predictions) ** 2)

    def make_cost_fn(self):
        """
        Create a JIT-compiled cost function.

        Returns a function: cost_fn(weights, x_batch, y_batch) -> scalar loss
        """

        @jax.jit
        def cost_fn(weights, x, y):
            preds = self.predict_batch(weights, x)
            return self.square_loss(y, preds)

        return cost_fn

    def make_update_step_fn(self):
        """
        Create a JIT-compiled update step function.

        Returns a function: update_step(weights, x_batch, y_batch) -> (loss, grads)
        Note: The optimizer update is applied outside this function.
        """

        @jax.jit
        def update_step(weights, x_batch, y_batch):
            def loss_fn(w):
                preds = self.predict_batch(w, x_batch)
                return self.square_loss(y_batch, preds)

            loss, grads = jax.value_and_grad(loss_fn)(weights)
            return loss, grads

        return update_step

    def predict_batch(self, weights, x):
        """
        Vectorized prediction for a batch of inputs.

        Assumes `predict_fn(weights, x_single)` exists in subclasses and produces
        a scalar output for one sample.
        """
        return jax.vmap(self.predict_fn, in_axes=(None, 0))(weights, x)

    def fit(
        self,
        x_train,
        y_train,
        epochs: int = None,
        batch_size: int = None,
        total_steps: int = None,
        steps_per_epoch: int = None,
        shuffle: bool = True,
        num_logs: int = 1000,
    ):
        """
        Train the model using either:
          - epochs (epoch-based training), or
          - total_steps (fixed optimizer-step budget)

        During training, the code:
          - scales inputs/targets using the scaler
          - optionally shuffles each epoch using a JAX PRNG key
          - runs batch SGD updates with Optax
          - logs loss values at a fixed step interval (log_every)
        """
        # Scale inputs and targets
        x_train, y_train = self.scaler.transform(x_train, y_train, fit=True)

        if self.weights is None:
            raise ValueError("Weights must be initialized before training.")

        # Determine batch size
        num_samples = len(x_train)
        if batch_size is None:
            batch_size = _auto_batch_size(num_samples)
        batch_size = min(batch_size, num_samples)

        # Determine steps per epoch
        if steps_per_epoch is None:
            steps_per_epoch = int(math.ceil(num_samples / batch_size))

        # Enforce exactly one training regime
        if total_steps is not None and epochs is not None:
            raise ValueError("Set either total_steps or epochs, not both.")

        # Convert the chosen regime into a number of epochs to run
        if total_steps is not None:
            epochs_run = int(math.ceil(total_steps / steps_per_epoch))
        else:
            if epochs is None:
                raise ValueError("Set either epochs or total_steps.")
            epochs_run = int(epochs)

        # Logging frequency for step-based training curves
        # (If total_steps is None, compute_log_every will raise by design.)
        log_every = compute_log_every(total_steps, num_logs=num_logs)

        # Initialize optimizer state
        opt_state = self.optimizer.init(self.weights)

        # Ensure targets are 1D for the training loop
        target_y_flat = y_train.flatten()

        history_steps = []
        history_costs = []

        step_count = 0
        last_loss = None

        for epoch in range(epochs_run):
            # Optional shuffling per epoch
            if shuffle:
                self.key, subkey = jax.random.split(self.key)
                permutation = jax.random.permutation(subkey, num_samples)
                x_epoch = x_train[permutation]
                y_epoch = target_y_flat[permutation]
            else:
                x_epoch = x_train
                y_epoch = target_y_flat

            for b in range(steps_per_epoch):
                # batch slicing
                start = b * batch_size
                end = min(start + batch_size, num_samples)

                x_batch = x_epoch[start:end]
                y_batch = y_epoch[start:end]

                # Compute loss and gradients
                loss, grads = self.update_step_fn(self.weights, x_batch, y_batch)
                last_loss = loss

                # Apply optimizer update
                updates, opt_state = self.optimizer.update(grads, opt_state)
                self.weights = optax.apply_updates(self.weights, updates)

                step_count += 1

                # Record loss at regular intervals
                if step_count % log_every == 0:
                    history_steps.append(step_count)
                    history_costs.append(loss)

                # Stop when the step budget is reached
                if step_count >= total_steps:
                    break

            if step_count >= total_steps:
                break

        # Ensure the last point is always recorded
        if len(history_steps) == 0 or history_steps[-1] != step_count:
            history_steps.append(step_count)
            history_costs.append(last_loss if last_loss is not None else jnp.nan)

        # Store training curve
        self.cost_history = {
            "steps": jnp.array(history_steps),
            "loss": jnp.array(history_costs),
            "log_every": log_every,
            "total_steps": int(total_steps),
        }
        return self

    def train(self, x_train, y_train, epochs=None, batch_size=None, total_steps=None, steps_per_epoch=None):
        """
        Convenience wrapper to match a unified model API (Classic + Quantum).
        """
        return self.fit(
            x_train=x_train,
            y_train=y_train,
            epochs=epochs,
            batch_size=batch_size,
            total_steps=total_steps,
            steps_per_epoch=steps_per_epoch,
        )

    def predict(self, x, weights=None):
        """
        Predict on raw (unscaled) inputs.

        Steps:
          1) scale x using the scaler
          2) compute batch predictions
          3) inverse-transform predictions back to original target scale
        """
        if weights is None:
            if self.weights is None:
                raise ValueError("Model has no trained weights.")
            weights = self.weights

        # Scale features (DataScaler defines the exact transformation)
        x_scaled, _ = self.scaler.transform(x_raw=x)

        # Forward pass in scaled space
        prediction_scaled = self.predict_batch(weights, x_scaled)

        # Inverse-transform predicted targets back to original scale
        _, y_inv = self.scaler.inverse_transform(y_scaled=prediction_scaled)

        return y_inv

    def get_predictions(self, weights, x):
        """Predict using an explicit set of weights (without changing internal state)."""
        return self.predict(x=x, weights=weights)

    def get_weights(self):
        """Return current model weights (JAX PyTree)."""
        return self.weights

    def predict_fn(self, weights, x):
        """
        Subclasses must implement this.

        Expected signature:
            predict_fn(weights, x_single) -> scalar prediction
        """
        raise NotImplementedError("predict_fn must be implemented by subclasses.")

    def count_params(self):
        """
        Count the total number of scalar parameters in the weight PyTree.
        """
        leaves, _ = jax.tree_util.tree_flatten(self.weights)
        return sum(w.size for w in leaves)