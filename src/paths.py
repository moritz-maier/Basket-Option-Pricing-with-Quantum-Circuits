from datetime import datetime
from pathlib import Path
import re

from src.run.params import Params


class PathConfig:
    """
    Centralized path and naming utilities for the project.

    This class provides a single place to define:
      - Project root discovery
      - Standard directory layout (data, processed, results, ...)
      - File naming conventions for datasets and results
      - Convenience helpers for locating saved artifacts on disk

    It is implemented as a singleton so all modules share the same configuration.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)

            # Project root is inferred relative to this file location
            root = Path(__file__).resolve().parents[1]

            cls._instance.root = root
            cls._instance.src = root / "src"
            cls._instance.data = root / "data"
            cls._instance.results = root / "results"

            # Raw downloaded data (e.g., from yfinance)
            cls._instance.yf = cls._instance.data / "yf"

            # Processed datasets stored in a serialized format (e.g., .pkl)
            cls._instance.processed = cls._instance.data / "processed"

        return cls._instance

    # ------------------------------------------------------------------
    # Raw market data paths
    # ------------------------------------------------------------------

    def get_yf_data_path(self, ticker: str) -> Path:
        """
        Path to locally cached yfinance data for one ticker.
        """
        return Path(self.yf) / f"{ticker}.csv"

    # ------------------------------------------------------------------
    # Dataset file naming and lookup
    # ------------------------------------------------------------------

    def build_dataset_filename(self, tickers, start, end, window, maturity, ext="pkl"):
        """
        Create a standardized filename for a processed dataset.

        The naming convention encodes the full dataset configuration, so the file
        can be uniquely identified and reproduced.
        """
        tick_str = "_".join(tickers)
        return f"{tick_str}_{start}_{end}_win{window}_mat{int(maturity)}.{ext}"

    def get_dataset_path(self, filename: str | None = None, **params) -> Path:
        """
        Return the full path to a processed dataset file.

        Supports two modes:
          - Provide an explicit filename
          - Provide the dataset parameters to generate the filename automatically
        """
        if filename is None:
            required = {"tickers", "start", "end", "window", "maturity"}
            if not required <= params.keys():
                raise ValueError("tickers, start, end, window, maturity must be provided.")

            filename = self.build_dataset_filename(
                params["tickers"],
                params["start"],
                params["end"],
                params["window"],
                params["maturity"],
            )

        return self.processed / filename

    def find_datasets(self, tickers=None, start=None, n_days=None, window=None, n_samples=None, fix_maturity=None):
        """
        Search the processed dataset directory for files matching certain criteria.
        """
        results = []

        for file in self.processed.glob("*.pkl"):
            name = file.stem

            # Regex parses encoded metadata from filename
            m = re.match(r"(.*)_(.*)_ndays(\d+)_win(\d+)_samples(\d+)(?:_fixM(\d))?", name)
            if not m:
                continue

            tick_str, s_date, nd, win, sd, fm = m.groups()
            fm = int(fm) if fm is not None else None

            # Apply filters
            if tickers is not None and "_".join(tickers) != tick_str:
                continue
            if start is not None and start != s_date:
                continue
            if n_days is not None and int(n_days) != int(nd):
                continue
            if window is not None and int(window) != int(win):
                continue
            if n_samples is not None and int(n_samples) != int(sd):
                continue
            if fix_maturity is not None and int(fix_maturity) != fm:
                continue

            results.append(file)

        return results

    # ------------------------------------------------------------------
    # Results directory layout
    # ------------------------------------------------------------------

    def get_result_base_folder(self, params: Params, subdir: str | None = None) -> Path:
        """
        Build the base folder path for storing results of an experiment.

        The hierarchy is designed to group runs by:
            model / tickers / start_date / end_date / option_type
        """
        model = params.model_params.model_name
        tick = "-".join(params.data.tickers)
        opt = params.data.option_type
        start = params.data.start_date
        end = str(params.data.end_date)

        base = self.results
        if subdir is not None:
            base = base / subdir

        return base / model / tick / start / end / opt

    def make_result_folder_name(self, params: Params) -> str:
        """
        Create a unique folder name for a single run.

        Includes:
            - seed
            - timestamp (to avoid collisions and allow chronological sorting)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        seed = f"seed{params.seed}"
        return f"{seed}_{timestamp}"

    def get_result_folder(self, params: Params, subdir: str | None = None) -> Path:
        """
        Return the full folder path for storing the output of one experiment run.
        """
        base = self.get_result_base_folder(params, subdir=subdir)
        name = self.make_result_folder_name(params)
        return base / name


# Global singleton instance used across the project
paths = PathConfig()