import numpy as np
from typing import Literal
from sklearn.model_selection import GroupShuffleSplit
from src.data_generation.Data import Data


class MLData:
    """
    Builds ML-ready (X, y) datasets from a Data object.

    Key features:
      - Creates expanded samples: (date, strike) -> one ML row (N_dates * M_strikes rows)
      - Supports splitting:
          * split_mode="random": GroupShuffleSplit by obs_idx (keeps all strikes of a date together)
          * split_mode="time": chronological split by date index
      - Optional multiplicative noise on y_train
    """

    def __init__(
        self,
        data: Data,
        option_type: Literal["best", "worst", "average"],
        use_percentage: bool = True,
        seed: int = 123,
        test_size: float = 0.2,
        noiseScale: float = 0.0,
        split_mode: Literal["random", "time"] = "random",
        sort_index_for_time_split: bool = True,
    ):

        self.data = data
        self.option_type = option_type
        self.use_percentage = bool(use_percentage)
        self.test_size = float(test_size)
        self.seed = int(seed)
        self.noiseScale = float(noiseScale)
        self.split_mode = split_mode
        self.sort_index_for_time_split = sort_index_for_time_split

        if split_mode not in ("random", "time"):
            raise ValueError("split_mode must be 'random' or 'time'.")

        # Build X/y + split immediately
        self._build()

    def _is_random_maturity(self) -> bool:
        """
        maturity is considered 'random' if Data.maturity is an array (per-date maturity),
        rather than a single integer.
        """
        return not isinstance(self.data.maturity, (int, np.integer))

    @staticmethod
    def _as_2d(x):
        """
        Helper: ensure a 1D array becomes a 2D column vector (n, 1).
        """
        x = np.asarray(x)
        return x[:, None] if x.ndim == 1 else x

    def _build_features_and_labels(self):
        """
        Build X (features) and y (labels) in expanded (N*M) form.

        Data.expand() returns:
          - S0_exp:     (N*M, n_assets)
          - vols_exp:   (N*M, n_assets)
          - T_exp:      (N*M,)
          - m_exp:      (N*M,)
          - obs_idx:    (N*M,)   date index per row
          - strike_idx: (N*M,)   strike index per row (not needed here)
        """
        S0_exp, vols_exp, T_exp, m_exp, obs_idx, strike_idx = self.data.expand()
        y_exp = self.data.expand_prices(self.option_type, self.use_percentage)

        # Ensure numpy arrays
        S0_exp = np.asarray(S0_exp)
        vols_exp = np.asarray(vols_exp)
        T_exp = np.asarray(T_exp)
        m_exp = np.asarray(m_exp)
        obs_idx = np.asarray(obs_idx)

        # --- sanity checks ---
        n = vols_exp.shape[0]
        if vols_exp.ndim != 2:
            raise ValueError(f"vols_exp must be 2D (N*M, n_assets). Got shape {vols_exp.shape}")
        if S0_exp.ndim != 2:
            raise ValueError(f"S0_exp must be 2D (N*M, n_assets). Got shape {S0_exp.shape}")
        if m_exp.ndim != 1:
            raise ValueError(f"m_exp must be 1D (N*M,). Got shape {m_exp.shape}")
        if T_exp.ndim != 1:
            raise ValueError(f"T_exp must be 1D (N*M,). Got shape {T_exp.shape}")
        if len(y_exp) != n:
            raise ValueError(f"y_exp length {len(y_exp)} does not match X rows {n}")

        # Build feature matrix parts in a controlled order
        parts = []

        # Include absolute S0 only if NOT using percentage features/labels
        if not self.use_percentage:
            parts.append(S0_exp.astype(np.float32))

        # Always include volatility features (per asset)
        parts.append(vols_exp.astype(np.float32))

        # Always include strike / moneyness as a scalar feature
        parts.append(self._as_2d(m_exp).astype(np.float32))

        # Include maturity feature only if maturity varies per observation
        if self._is_random_maturity():
            parts.append(self._as_2d(T_exp).astype(np.float32))

        # Final design matrix
        X = np.concatenate(parts, axis=1).astype(np.float32)
        y = np.asarray(y_exp, dtype=np.float32)

        return X, y, obs_idx

    def _time_split_masks(self, obs_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Chronological split:
          - train = first (1 - test_size) fraction of dates
          - test  = last test_size fraction of dates
        Uses obs_idx to keep all strikes of a date together.
        """
        if self.sort_index_for_time_split and not self.data.prices.index.is_monotonic_increasing:
            raise ValueError("prices.index must be monotonically increasing for time split.")

        N_dates = self.data.prices.shape[0]
        n_test_dates = int(round(self.test_size * N_dates))
        n_test_dates = max(1, min(n_test_dates, N_dates - 1))
        split_date_idx = N_dates - n_test_dates

        train_mask = obs_idx < split_date_idx
        test_mask = ~train_mask
        return train_mask, test_mask

    def _group_random_split_masks(self, X: np.ndarray, y: np.ndarray, obs_idx: np.ndarray):
        """
        random split by grouping on obs_idx (date index).
        """
        gss = GroupShuffleSplit(n_splits=1, test_size=self.test_size, random_state=self.seed)
        train_idx, test_idx = next(gss.split(X, y, groups=obs_idx))

        train_mask = np.zeros(len(X), dtype=bool)
        test_mask = np.zeros(len(X), dtype=bool)
        train_mask[train_idx] = True
        test_mask[test_idx] = True
        return train_mask, test_mask

    def _build(self):
        """
        Build features/labels and perform the chosen split.
        """
        X, y, obs_idx = self._build_features_and_labels()

        if self.split_mode == "time":
            train_mask, test_mask = self._time_split_masks(obs_idx)
        else:
            # Use grouped random split instead of train_test_split to avoid leakage
            train_mask, test_mask = self._group_random_split_masks(X, y, obs_idx)

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        # Optional multiplicative noise on training labels (robustness / stress test)
        if self.noiseScale != 0.0:
            rng = np.random.default_rng(self.seed)
            noise = rng.normal(0.0, self.noiseScale, size=y_train.shape).astype(y_train.dtype)
            y_train = y_train * (1.0 + noise)

        # Store splits
        self.X_train_full = X_train
        self.Y_train_full = y_train
        self.X_test = X_test
        self.Y_test = y_test

        # Store masks + group ids for debugging / reproducibility checks
        self.train_mask = train_mask
        self.test_mask = test_mask
        self.obs_idx_train = obs_idx[train_mask]
        self.obs_idx_test = obs_idx[test_mask]

    def get_train_subset(self, N: int, subset_seed: int):
        """
        Sample a subset of the training set without replacement.
        """
        if N > len(self.X_train_full):
            raise ValueError(f"N={N} is larger than available training set ({len(self.X_train_full)})")

        rng = np.random.default_rng(subset_seed)
        idx = rng.choice(len(self.X_train_full), size=N, replace=False)
        return self.X_train_full[idx], self.Y_train_full[idx]

    def get_train_full(self):
        """Return the full training set."""
        return self.X_train_full, self.Y_train_full

    def get_train(self, N: int = -1, subset_seed: int = None):
        """
        Convenience getter:
          - N=-1 -> full train
          - N>0  -> subset of size N
        """
        if N == -1:
            return self.get_train_full()
        if N > 0:
            if subset_seed is None:
                subset_seed = self.seed
            return self.get_train_subset(N, subset_seed)
        raise ValueError(f"N must be -1 or >0: N={N}")

    def get_test(self):
        """Return the test set."""
        return self.X_test, self.Y_test