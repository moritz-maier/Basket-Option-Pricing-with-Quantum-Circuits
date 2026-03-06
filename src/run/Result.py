import os
import json
import pickle
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
from numpy.typing import NDArray
import jax.numpy as jnp

from src.paths import paths
from src.run.params import DataParams, Params, TrainingParams


@dataclass
class Result:
    """
    Container for the outcome of a single experiment run.

    Stores:
      - The full experiment configuration (Params)
      - Evaluation metrics (e.g. train/test R2, MSE, MAE, ...)
      - The trained model weights
      - Optional training curve information (cost_history)

    This object is designed to be serializable to disk for reproducibility.
    """
    params: Params
    metrics: Dict[str, Dict[str, float]]
    weights: NDArray[np.float32]
    cost_history: List[float] | None = None

    def save(self, path: Path) -> None:
        """
        Save this Result object to disk.

        Note:
            In the current implementation, saving is delegated to `save_result(...)`,
            which handles folder naming and JSON/PKL output.
        """
        save_result(self, path, True)


def to_python(obj):
    """
    Convert nested objects into JSON-serializable Python types.

    This is mainly required because:
      - NumPy arrays are not JSON-serializable by default
      - JAX arrays are not JSON-serializable by default
      - Some scalar types need conversion via .item()

    Returns plain Python objects (dict/list/float/int/str/None).
    """
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, jnp.ndarray):
        return np.array(obj).tolist()
    if isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
        return float(obj)
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return obj


def save_result(result: Result, subdir=None, save_cost_history: bool = True):
    """
    Save a Result object to disk.

    Output format (inside an experiment folder):
      - weights.pkl   : binary weights (pickle)
      - result.json   : parameters + metrics + metadata

    Parameters
    ----------
    result : Result
        The result object to store.
    subdir : str | None
        Optional subdirectory for organizing different experiment batches
    save_cost_history : bool
        If False, omit cost_history from JSON to reduce file size.

    Returns
    -------
    Path
        The folder where results were stored.
    """
    # Determine experiment folder path from params (centralized naming logic)
    folder = paths.get_result_folder(result.params, subdir=subdir)
    folder.mkdir(parents=True, exist_ok=True)

    # Store weights separately
    with open(folder / "weights.pkl", "wb") as f:
        pickle.dump(result.weights, f)

    # Store metadata + metrics in JSON
    result_dict = {
        "params": asdict(result.params),                     # dataclass -> dict
        "metrics": result.metrics,
        "weights_file": "weights.pkl",
        "cost_history": result.cost_history if save_cost_history else None,
    }

    with open(folder / "result.json", "w") as f:
        json.dump(to_python(result_dict), f, indent=4)

    print("Result saved to:", folder)
    return folder


def load_result_json(folder: str | Path) -> Result:
    """
    Load a Result from a result folder.

    Expects:
      - result.json
      - weights.pkl (or another filename referenced inside result.json)

    Reconstructs:
      - Params dataclass (including correct model param subclass)
      - metrics
      - weights
      - cost_history (optional)

    """
    folder = Path(folder)
    json_path = folder / "result.json"

    # Load JSON metadata
    with open(json_path, "r") as f:
        data = json.load(f)

    # Load weights
    with open(folder / data["weights_file"], "rb") as f:
        weights = pickle.load(f)

    # Reconstruct Params (including the correct ModelParams type)
    p = data["params"]
    model_type = p["model_params"]["model_name"]

    if "classic" in model_type.lower():
        from src.run.params import ClassicModelParams
        model_params = ClassicModelParams(**p["model_params"])

    elif model_type == "quantum":
        from src.run.params import QuantumModelParams

        layers = p["model_params"]["layers"]

        encoding_base = p["model_params"].get("encoding_base")

        if encoding_base is None:
            encoding_base = 1 if layers == 1 else 3

        model_params = QuantumModelParams(
            layers=layers,
            n_trainable_blocks=p["model_params"]["n_trainable_blocks"],
            encoding_base=encoding_base,
            model_name=p["model_params"].get("model_name", "quantum"),

        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    params = Params(
        seed=p["seed"],
        data=DataParams(**p["data"]),
        model_params=model_params,
        training_params=TrainingParams(**p["training_params"]),
    )

    # Use file modification time as a simple timestamp for "latest result" selection
    timestamp = datetime.fromtimestamp(os.path.getmtime(json_path))

    result = Result(
        params=params,
        metrics=data["metrics"],
        weights=weights,
        cost_history=data.get("cost_history"),
    )

    # Attach timestamp dynamically
    result.timestamp = timestamp
    return result