import pickle
from pathlib import Path
from dataclasses import dataclass, field
from typing import Union, Tuple, Dict
import numpy as np
import pandas as pd

# Key for result dictionaries:
# (option_type, use_percentage) distinguishes best/worst/average and strike interpretation.
ResultKey = Tuple[str, bool]


@dataclass
class Data:
    """
    Container for one generated dataset.

    Holds:
      - underlying data (prices, returns, vol estimates, correlation)
      - option pricing inputs (relative strikes, maturity, r, q, n_paths)
      - precomputed labels/results (basket option prices + attribution statistics)

    Note:
      n_paths is fixed per dataset because it defines the Monte Carlo precision of labels.
    """

    # --- core market data (time-indexed, shape: N_dates x n_assets) ---
    prices: pd.DataFrame
    returns: pd.DataFrame
    rolling_vols: pd.DataFrame

    # --- correlation information ---
    corr_matrix: np.ndarray     # full correlation matrix (n_assets x n_assets)
    corr_values: np.ndarray     # stored correlation values / diagnostics

    # --- option definition ---
    relative_strikes: np.ndarray   # shape (N_dates, M_strikes) or (N, M): moneyness/relative strikes
    maturity: Union[int, np.ndarray]  # int = fixed maturity; array = per-date maturity
    r: float                         # risk-free rate
    q: float                         # dividend yield (scalar here)
    n_paths: int                     # Monte Carlo paths used to produce labels for this dataset

    # --- computed results/labels (stored per (option_type, use_percentage)) ---
    basket_prices: Dict[ResultKey, pd.DataFrame] = field(default_factory=dict)
    counts: Dict[ResultKey, np.ndarray] = field(default_factory=dict)
    means: Dict[ResultKey, np.ndarray] = field(default_factory=dict)

    # ---------------- Key helpers ----------------
    @staticmethod
    def _key(option_type: str, use_percentage: bool) -> ResultKey:
        """Normalize the dictionary key type (string + bool)."""
        return (str(option_type), bool(use_percentage))

    # ---------------- Expand inputs for ML / pricing ----------------
    def expand(self):
        """
        Expand the dataset from date-level inputs to (date, strike) rows.

        If you have:
          - N dates
          - M strikes

        Then the expanded arrays have length N*M, so each row corresponds to one
        (date_idx, strike_idx) pair.

        Returns:
          S0_exp:     (N*M, n_assets) repeated spot prices per strike
          vols_exp:   (N*M, n_assets) repeated vol estimates per strike
          T_exp:      (N*M,)          maturity per row (repeated per strike)
          m_exp:      (N*M,)          flattened relative strike values
          obs_idx:    (N*M,)          date index for each row (0..N-1)
          strike_idx: (N*M,)          strike index for each row (0..M-1)
        """
        S0s = self.prices.to_numpy()
        vols = self.rolling_vols.to_numpy()

        N = S0s.shape[0]
        m = np.asarray(self.relative_strikes)
        M = m.shape[1]

        # Handle fixed maturity (int) vs per-date maturity (array)
        if isinstance(self.maturity, (int, np.integer)):
            T = np.full(N, int(self.maturity), dtype=int)
        else:
            T = np.asarray(self.maturity)

        # Repeat each date-row M times (once for each strike)
        S0_exp = np.repeat(S0s, M, axis=0)
        vols_exp = np.repeat(vols, M, axis=0)
        T_exp = np.repeat(T, M, axis=0)

        # Flatten the strike grid (N x M -> N*M)
        m_exp = m.reshape(-1)

        # Helpful indices for grouped splits / reshaping back
        obs_idx = np.repeat(np.arange(N), M)   # date index per expanded row
        strike_idx = np.tile(np.arange(M), N)  # strike index per expanded row

        return S0_exp, vols_exp, T_exp, m_exp, obs_idx, strike_idx

    # ---------------- Store computed labels/results ----------------
    def store_result(
        self,
        option_type: str,
        use_percentage: bool,
        prices_df: pd.DataFrame,
        counts: np.ndarray,
        means: np.ndarray,
    ):
        """
        Store precomputed basket option prices and attribution stats.

        prices_df:
          DataFrame of shape (N_dates, M_strikes) with option prices.

        counts / means:
          Optional diagnostics from Monte Carlo:
            - counts: how often each asset was selected (best/worst) among ITM paths
            - means: average payoff attributed to each selected asset
        """
        key = self._key(option_type, use_percentage)
        self.basket_prices[key] = prices_df
        self.counts[key] = np.asarray(counts)
        self.means[key] = np.asarray(means)

    # ---------------- Access stored labels/results ----------------
    def get_prices(self, option_type: str, use_percentage: bool) -> pd.DataFrame:
        """Get the stored price DataFrame for a given option_type + strike mode."""
        key = self._key(option_type, use_percentage)
        if key not in self.basket_prices:
            raise KeyError(f"No prices stored for {key}")
        return self.basket_prices[key]

    def expand_prices(self, option_type: str, use_percentage: bool) -> np.ndarray:
        """
        Return stored basket prices as a flat (N*M,) vector to match expand().
        """
        df = self.get_prices(option_type, use_percentage)
        return df.to_numpy().reshape(-1)

    def get_basket_prices_combined(self) -> pd.DataFrame:
        """
        Combine all stored basket price tables into one DataFrame with MultiIndex columns.

        Output columns:
          (option_type, use_percentage, strike)
        """
        frames = []

        for (option_type, use_percentage), df in self.basket_prices.items():
            df_copy = df.copy()
            df_copy.columns = pd.MultiIndex.from_product(
                [[option_type], [use_percentage], df.columns],
                names=["option_type", "use_percentage", "strike"]
            )
            frames.append(df_copy)

        if not frames:
            raise ValueError("No basket_prices stored.")

        return pd.concat(frames, axis=1)

    # ---------------- Persistence (pickle) ----------------
    def save(self, path: str | Path):
        """
        Save the Data object to disk via pickle.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self, f)

        print(f"Data saved to {path}")

    @staticmethod
    def load(path: str | Path) -> "Data":
        """
        Load a Data object from disk via pickle.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, "rb") as f:
            data = pickle.load(f)

        print(f"Data loaded from {path}")
        return data