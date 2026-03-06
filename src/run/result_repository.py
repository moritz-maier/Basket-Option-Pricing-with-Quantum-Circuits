import json
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.data_generation.DataManager import DataManager
from src.paths import paths
from src.run.Result import Result, load_result_json
from src.run.pipeline import build_pipeline


def find_results(
    subdir: Optional[str | Path] = None,
    model: Optional[str] = None,
    tickers: Optional[List[str]] = None,
    option_type: Optional[str] = None,
    seed: Optional[int] = None,
) -> List[Path]:
    """
    Search the results directory for experiment runs and filter them by metadata.

    Each experiment run is assumed to be stored in a folder containing:
        - result.json
        - weights.pkl (or another weights file referenced in the JSON)

    Filtering is performed by reading the metadata in result.json.

    Parameters
    ----------
    subdir : str | Path | None
        Optional subfolder under the main results directory (used to separate runs).
    model : str | None
        Filter by model name (e.g. "quantum", "classic_ferguson", "classic_culkin").
    tickers : list[str] | None
        Filter by ticker list (matched as a joined string "A-B-C").
    option_type : str | None
        Filter by option type (e.g. "best", "worst", "average").
    seed : int | None
        Filter by experiment seed.

    Returns
    -------
    list[Path]
        List of folders that match the filters.
    """
    # Base results directory
    root = paths.results

    # Optional subdirectory for grouped experiment batches
    if subdir is not None:
        root = root / subdir

    results: List[Path] = []

    # Search recursively for experiment metadata files
    for json_file in root.rglob("result.json"):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
        except Exception:
            # Skip corrupted/unreadable JSON files
            continue

        p = data.get("params", {})

        # ---- filter: model ----
        if model is not None:
            if p.get("model_params", {}).get("model_name") != model:
                continue

        # ---- filter: tickers ----
        if tickers is not None:
            if "-".join(tickers) != "-".join(p.get("data", {}).get("tickers", [])):
                continue

        # ---- filter: option type ----
        if option_type is not None:
            if p.get("data", {}).get("option_type") != option_type:
                continue

        # ---- filter: seed ----
        if seed is not None:
            if p.get("seed") != seed:
                continue

        # Store the parent folder of result.json
        results.append(json_file.parent)

    return results


def latest_result(results: List[Result]) -> Optional[Result]:
    """
    Select the most recent result based on the timestamp stored during loading.

    Parameters
    ----------
    results : list[Result]
        Loaded results.

    Returns
    -------
    Result | None
        Most recent result or None if input is empty.
    """
    if not results:
        return None
    return max(results, key=lambda r: r.timestamp)


def load_all_results(
    subdir: Optional[str | Path] = None,
    **filters
) -> List[Result]:
    """
    Load all results matching filters.

    This function combines:
        - find_results(...) to locate folders
        - load_result_json(...) to reconstruct Result objects

    Parameters
    ----------
    subdir : str | Path | None
        Optional subfolder under results directory.
    **filters
        Passed through to find_results(...).

    Returns
    -------
    list[Result]
        Loaded result objects.
    """
    result_dirs = find_results(subdir=subdir, **filters)
    loaded: List[Result] = []

    for folder in result_dirs:
        try:
            loaded.append(load_result_json(folder))
        except Exception as e:
            print(f"Warning: could not load Result in {folder}: {e}")
            raise

    return loaded


def results_to_df(
    results: List[Result],
    include_weights: bool = False,
    include_cost_history: bool = False,
    include_object: bool = False
) -> pd.DataFrame:
    """
    Convert a list of Result objects into a flat pandas DataFrame.

    Parameters
    ----------
    results : list[Result]
        Result objects loaded from disk.
    include_weights : bool
        If True, include raw model weights (can be large).
    include_cost_history : bool
        If True, include training loss history (can be large).
    include_object : bool
        If True, store the full Result object in one column (not recommended for CSV).

    Returns
    -------
    pd.DataFrame
        One row per result with flattened metadata and metrics.
    """
    rows = []

    for r in results:
        params = r.params
        model = params.model_params

        # ---- core metadata ----
        row = {
            "seed": params.seed,
            "model": model.model_name,
            "tickers": "-".join(params.data.tickers),
            "option_type": getattr(params.data, "option_type", None),
            "window": params.data.window,
            "noiseScale": params.data.noiseScale,
            "start_date": params.data.start_date,
            "end_date": params.data.end_date,
            "batch_size": params.training_params.batch_size,
            "total_steps": params.training_params.total_steps,
            "steps_per_epoch": params.training_params.steps_per_epoch,
            "n_samples": params.data.n_samples,
            "split_mode": params.data.split_mode,
        }

        # ---- model-specific hyperparameters ----
        if "classic" in model.model_name.lower():
            row["units"] = model.units
            row["dropout"] = model.dropout

        elif model.model_name == "quantum":
            row["encoding_base"] = getattr(model, "encoding_base", 3)
            row["n_trainable_blocks"] = model.n_trainable_blocks
            row["layers"] = model.layers

        # ---- flatten metrics ----
        # metrics expected format:
        for split_name, metric_dict in r.metrics.items():
            for metric_name, metric_val in metric_dict.items():
                row[f"{split_name}_{metric_name}"] = metric_val

        # ---- optional payload ----
        if include_weights:
            row["weights"] = r.weights

        if include_cost_history:
            row["cost_history"] = r.cost_history

        if include_object:
            row["result_obj"] = r

        rows.append(row)

    return pd.DataFrame(rows)


def reload_model(result: Result):
    """
    Rebuild the model and dataset pipeline for a given stored Result.


    Note:
        This function rebuilds the model architecture and dataset and returns
        the model instance along with train/test splits. It does not automatically
        set weights; the caller can apply result.weights afterwards if needed.

    Returns
    -------
    model, x_train, x_test, y_train, y_test
    """
    dm = DataManager()
    dataParams = result.params.data
    modelParams = result.params.model_params
    trainingParams = result.params.training_params
    seed = result.params.seed

    model, x_train, x_test, y_train, y_test = build_pipeline(
        dm=dm,
        seed=seed,
        dataParams=dataParams,
        modelParams=modelParams,
        trainingParams=trainingParams,
        n_samples=dataParams.n_samples,
    )

    return model, x_train, x_test, y_train, y_test