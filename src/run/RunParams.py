from src.data_generation.DataManager import DataManager
from src.run.pipeline import build_pipeline
from src.visualize.metrics import get_metrics
from src.run.params import Params
from src.run.Result import Result


def run_params(params: Params) -> Result:
    """
    Run a single experiment defined by `Params`.

    This function executes the full workflow:
        1) Load dataset and build train/test splits via `build_pipeline`
        2) Instantiate the requested model (classic or quantum)
        3) Train the model using the provided training configuration
        4) Compute predictions and evaluation metrics
        5) Package everything into a `Result` object for saving / analysis

    Parameters
    ----------
    params : Params
        Full experiment configuration including data, model, training, and seed.

    Returns
    -------
    Result
        Experiment result containing parameters, metrics, trained weights,
        and optional training history.
    """
    # Data access layer (loads precomputed datasets from disk)
    dm = DataManager()

    # Unpack configuration sections for readability
    dataParams = params.data
    modelParams = params.model_params
    trainingParams = params.training_params
    seed = params.seed

    # ------------------------------------------------------------
    # 1) Build pipeline (data loading + splitting + model creation)
    # ------------------------------------------------------------
    model, x_train, x_test, y_train, y_test = build_pipeline(
        dm=dm,
        seed=seed,
        dataParams=dataParams,
        modelParams=modelParams,
        trainingParams=trainingParams,
        n_samples=dataParams.n_samples,
    )

    # Report the number of trainable parameters
    print("Params:", model.count_params())

    # ------------------------------------------------------------
    # 2) Train the model
    # ------------------------------------------------------------
    model.train(
        x_train,
        y_train,
        epochs=trainingParams.epochs,
        batch_size=trainingParams.batch_size,
        total_steps=trainingParams.total_steps,
        steps_per_epoch=trainingParams.steps_per_epoch,
    )

    # ------------------------------------------------------------
    # 3) Evaluate on train and test sets
    # ------------------------------------------------------------
    predictions_train = model.predict(x_train)
    predictions_test = model.predict(x_test)

    # Compute metrics
    metrics = get_metrics(
        y_train=y_train,
        predictions_train=predictions_train,
        y_test=y_test,
        predictions_test=predictions_test,
    )

    # ------------------------------------------------------------
    # 4) Package results for storage and later analysis
    # ------------------------------------------------------------
    weights = model.get_weights()
    cost_history = model.cost_history

    return Result(
        params=params,
        metrics=metrics,
        weights=weights,
        cost_history=cost_history,
    )