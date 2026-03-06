import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import numpy as np

from src.models.utils import _auto_batch_size, compute_log_every


class StepLossLogger(tf.keras.callbacks.Callback):
    """
    Keras callback that logs the training loss every `log_every` optimizer steps.
    """

    def __init__(self, log_every: int):
        super().__init__()
        self.log_every = int(log_every)
        self.steps = []
        self.losses = []
        self._step = 0

    def on_train_begin(self, logs=None):
        # Reset history at the beginning of each training run
        self.steps = []
        self.losses = []
        self._step = 0

    def on_train_batch_end(self, batch, logs=None):
        # Called once per gradient update (batch)
        self._step += 1
        if self.log_every > 0 and (self._step % self.log_every == 0):
            loss = float(logs.get("loss")) if logs and "loss" in logs else np.nan
            self.steps.append(self._step)
            self.losses.append(loss)


class ClassicMLModel:
    """
    TensorFlow/Keras baseline model with a unified API similar to the quantum model.

    Supported architectures:
      - "ferguson": 6 hidden layers with ReLU, linear output
      - "culkin"  : mixed activations + dropout, exponential output

    Public interface (to match pipeline/protocol):
      - train(...)
      - predict(...)
      - get_weights()
      - count_params()
      - cost_history (stored after training)
    """

    def __init__(
        self,
        input_shape: int,
        units: int,
        dropout: float,
        seed: int,
        learning_rate: float,
        model_type: str = "ferguson",
        scale_inputs: bool = True,
    ):
        """
        Parameters
        ----------
        input_shape : int
            Number of input features.
        units : int
            Width of hidden layers.
        dropout : float
            Dropout rate (used only in the "culkin" architecture).
        seed : int
            Random seed for TF and NumPy.
        learning_rate : float
            Adam learning rate.
        model_type : str
            "ferguson" or "culkin".
        scale_inputs : bool
            If True, scale inputs using StandardScaler.
        """
        self.input_shape = int(input_shape)
        self.units = int(units)
        self.dropout = float(dropout)
        self.seed = int(seed)
        self.learning_rate = float(learning_rate)
        self.model_type = model_type.lower()

        # Feature scaling
        self.scaler = StandardScaler()

        # Build + compile Keras model
        self.model = self._build_model()

        # Will be filled after training
        self.cost_history = None

        self.weights = self.model.get_weights()

    def get_weights(self):
        """Return current model weights (Keras format: list of numpy arrays)."""
        return self.model.get_weights()

    def _build_model(self):
        """
        Construct and compile the Keras model according to the chosen architecture.
        """
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)

        if self.model_type == "culkin":
            # Deeper network with dropout and exponential output
            model = Sequential([
                Input(shape=(self.input_shape,)),
                Dense(self.units, activation="leaky_relu"),
                Dropout(self.dropout),
                Dense(self.units, activation="elu"),
                Dropout(self.dropout),
                Dense(self.units, activation="relu"),
                Dropout(self.dropout),
                Dense(self.units, activation="elu"),
                Dropout(self.dropout),
                Dense(1, activation="exponential"),
            ])

        elif self.model_type == "ferguson":
            # 6x ReLU hidden layers + linear output layer
            model = Sequential()
            model.add(Input(shape=(self.input_shape,)))
            for _ in range(6):
                model.add(Dense(self.units, activation="relu"))
            model.add(Dense(1))

        else:
            raise ValueError(
                f"Unknown model_type='{self.model_type}'. Use 'ferguson' or 'culkin'."
            )

        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss="mse")
        return model

    def train(
        self,
        x_train,
        y_train,
        epochs: int = None,
        batch_size: int = None,
        total_steps: int = None,
        steps_per_epoch: int = None,
        early_stopping: bool = False,
        patience: int = 20,
        num_logs: int = 1000,
    ):
        """
        Train the model.

        You can specify either:
          - epochs (standard Keras training), OR
          - total_steps (fixed optimizer-step budget).

        If total_steps is used:
          - A repeating tf.data.Dataset is created
          - compute epochs_needed = ceil(total_steps / steps_per_epoch)
          - StepLossLogger logs the loss every ~total_steps/num_logs steps
        """
        # Fit scaler on training features and transform X
        self.fit_scaler(x_train)
        x_train_scaled = self.scaler.transform(x_train)

        # Set seeds for reproducibility
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)

        n_train = x_train_scaled.shape[0]
        if batch_size is None:
            batch_size = _auto_batch_size(n_train)

        callbacks = []

        # Enforce exactly one training regime
        if total_steps is not None and epochs is not None:
            raise ValueError("Set either total_steps or epochs, not both.")

        # Step-based loss logger only makes sense if total_steps is given
        if total_steps is not None:
            log_every = compute_log_every(total_steps, num_logs=num_logs)
            step_logger = StepLossLogger(log_every=log_every)
            callbacks.append(step_logger)
        else:
            step_logger = None

        # Optional early stopping
        if early_stopping:
            callbacks.append(EarlyStopping(
                monitor="loss",
                patience=patience,
                restore_best_weights=True
            ))

        # ---- Epoch-based training ----
        if total_steps is None:
            if epochs is None:
                raise ValueError("Set either epochs or total_steps.")
            history = self.model.fit(
                x_train_scaled, y_train,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                callbacks=callbacks
            )

            # Store loss per epoch as a simple history
            self.cost_history = {
                "steps": np.arange(len(history.history["loss"])),
                "loss": np.array(history.history["loss"]),
            }
            return self

        # ---- Step-based training ----
        if steps_per_epoch is None:
            steps_per_epoch = int(np.ceil(n_train / batch_size))
        epochs_needed = int(np.ceil(total_steps / steps_per_epoch))

        # Infinite dataset: shuffle -> batch -> repeat
        ds = tf.data.Dataset.from_tensor_slices((x_train_scaled, y_train))
        ds = ds.shuffle(buffer_size=n_train, seed=self.seed, reshuffle_each_iteration=True)
        ds = ds.batch(batch_size, drop_remainder=False).repeat()

        self.model.fit(
            ds,
            epochs=epochs_needed,
            steps_per_epoch=steps_per_epoch,
            verbose=0,
            callbacks=callbacks
        )

        # Store logged step-level history
        self.cost_history = {
            "steps": np.array(step_logger.steps, dtype=np.int64),
            "loss": np.array(step_logger.losses, dtype=np.float32),
            "log_every": log_every,
            "total_steps": int(total_steps),
        }
        return self

    def fit_scaler(self, x_train, y_train=None):
        """Fit the StandardScaler on the training features."""
        self.scaler.fit(x_train)

    def predict(self, x, weights=None):
        """
        Predict outputs for inputs x.
.
        Use get_predictions(weights, x) if you want to set weights explicitly.
        """
        x_scaled = self.scaler.transform(x)
        preds = self.model.predict(x_scaled, verbose=0).flatten()
        return preds

    def get_predictions(self, weights, x):
        """
        Predict outputs using an explicit weight set.
        """
        self.model.set_weights(weights)
        x_scaled = self.scaler.transform(x)
        preds = self.model.predict(x_scaled, verbose=0).flatten()
        return preds

    def predict_batch(self, weights, x):
        """Compatibility helper; delegates to predict()."""
        return self.predict(x)

    def count_params(self):
        """Return the number of trainable parameters in the Keras model."""
        return self.model.count_params()

    def summary(self):
        """Print the Keras model summary."""
        return self.model.summary()