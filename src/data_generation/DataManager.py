from pathlib import Path
from src.data_generation.Data import Data
from src.data_generation.utils_data import TRADING_DAYS
from src.paths import paths


def load_by_params(tickers, start, end, window, maturity=TRADING_DAYS/4) -> Data:
    """
    Convenience function to load a Data object based on dataset parameters.

    Internally:
      1) Build a deterministic filename from the parameters
      2) Resolve the dataset path
      3) Load the pickled Data object

    This assumes that the dataset was previously generated and saved
    using the same parameter naming convention.
    """
    filename = paths.build_dataset_filename(tickers, start, end, window, maturity)
    path = paths.get_dataset_path(filename)
    return Data.load(path)


class DataManager:
    """
    Thin abstraction layer around dataset storage.

    Responsibilities:
      - Keep track of the base dataset folder
      - List available datasets
      - Load datasets by filename
      - Load datasets by parameter signature

    """

    def __init__(self, folder=None):
        """
        Parameters
        ----------
        folder : Path | str | None
            Base folder where processed datasets are stored.
            If None, defaults to paths.processed.
        """
        self.folder = Path(folder or paths.processed)

    def list(self):
        """
        Return all available dataset files in the folder.
        """
        return list(self.folder.glob("*.pkl"))

    def load(self, filename):
        """
        Load a dataset by filename (relative to the dataset folder).
        """
        path = self.folder / filename
        return Data.load(path)

    def load_by_params(self, tickers, start, end, window, maturity=TRADING_DAYS/4) -> Data:
        """
        Load dataset using the same parameter-to-filename logic
        as during generation.
        """
        return load_by_params(tickers, start, end, window, maturity)