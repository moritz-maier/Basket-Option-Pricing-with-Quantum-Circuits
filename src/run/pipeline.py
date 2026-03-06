from typing import Any

from src.models.protocols import Model

from src.data_generation.DataManager import DataManager
from src.data_generation.MLData import MLData
from src.models.ClassicMLModel import ClassicMLModel
from src.models.QuantumModel import QuantumModel


def build_pipeline(
    dm: "DataManager",
    seed: int,
    dataParams,
    modelParams,
    trainingParams=None,
    *,
    n_samples: int = -1,
) -> tuple[Model, Any, Any, Any, Any]:
    """
    Build the full experiment pipeline:
        1) load the dataset
        2) create train/test splits + feature matrix
        3) instantiate the requested model architecture

    Parameters
    ----------
    dm : DataManager
        Data access layer responsible for loading precomputed datasets.
    seed : int
        Random seed used for data splitting and model initialization.
    dataParams : DataParams
        Dataset configuration (tickers, dates, option type, split mode, etc.).
    modelParams : ModelParams
        Model configuration (classic vs quantum hyperparameters).
    trainingParams : TrainingParams | None
        Training configuration (learning rate, epochs/steps, batch size, ...).
    n_samples : int
        Number of training samples to use (-1 means full training set).

    Returns
    -------
    model : Model
        Instantiated model (classic or quantum) implementing the common Model protocol.
    x_train, x_test, y_train, y_test : Any
        Prepared train/test arrays (usually NumPy arrays).
    """

    # ------------------------------------------------------------
    # 1) Load precomputed dataset from disk
    # ------------------------------------------------------------
    data = dm.load_by_params(
        dataParams.tickers,
        dataParams.start_date,
        dataParams.end_date,
        dataParams.window,
        dataParams.maturity,
    )

    # ------------------------------------------------------------
    # 2) Build ML-ready features and targets and create train/test split
    # ------------------------------------------------------------
    mld = MLData(
        data=data,
        option_type=dataParams.option_type,
        use_percentage=dataParams.use_percentage,
        seed=seed,
        noiseScale=dataParams.noiseScale,
        split_mode=dataParams.split_mode,
    )

    # Optionally subsample training set (useful for scaling studies)
    x_train, y_train = mld.get_train(N=n_samples)
    x_test, y_test = mld.get_test()

    # The model input dimension is determined by feature construction in MLData
    n_features = x_train.shape[1]

    # ------------------------------------------------------------
    # 3) Instantiate the requested model type
    # ------------------------------------------------------------
    if modelParams.model_name == "quantum":
        # Quantum model: JAX + PennyLane circuit with data re-uploading and variational blocks
        model: Model = QuantumModel(
            seed=seed,
            L=modelParams.layers,
            n_trainable_blocks=modelParams.n_trainable_blocks,
            n_features=n_features,
            learning_rate=trainingParams.learning_rate ,
            encoding_base=modelParams.encoding_base,
        )

    elif modelParams.model_name == "classic_ferguson":
        # Classical baseline: fully-connected ReLU network (Ferguson-style)
        model = ClassicMLModel(
            input_shape=n_features,
            units=modelParams.units,
            dropout=modelParams.dropout,
            seed=seed,
            learning_rate= trainingParams.learning_rate ,
            model_type="ferguson",
        )

    elif modelParams.model_name in ("classic_culkin", "classic"):
        # Classical baseline: deeper network with dropout and exponential output (Culkin-style)
        model = ClassicMLModel(
            input_shape=n_features,
            units=modelParams.units,
            dropout=modelParams.dropout,
            seed=seed,
            learning_rate= trainingParams.learning_rate ,
            model_type="culkin",
        )

    else:
        raise ValueError(f"Unknown model_name: {modelParams.model_name}")

    return model, x_train, x_test, y_train, y_test