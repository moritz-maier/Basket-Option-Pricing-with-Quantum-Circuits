from typing import Literal

from src.paths import paths
import numpy as np
import pandas as pd
import yfinance as yf
from src.data_generation.utils_data import TRADING_DAYS
from src.data_generation.Data import Data
from src.data_generation.compute_basket_price import compute_basket_price_from_data


def nearest_positive_definite(A):
    """
    Ensure the correlation matrix is positive definite by clipping eigenvalues.

    Cholesky requires positive definiteness
    """
    B = (A + A.T) / 2
    eigvals, eigvecs = np.linalg.eigh(B)
    eigvals = np.clip(eigvals, 1e-12, None)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def download_asset(ticker: str):
    """
    Download long history of daily close prices for a ticker and store to CSV.

    Note:
      - This uses yfinance and stores under a project-specific path.
      - Data is later sliced to start_date/end_date in the generator.
    """
    file_path = paths.get_yf_data_path(ticker)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading data for {ticker}...")

    data = yf.download(ticker, start="1900-01-01", end="2026-01-01")["Close"].dropna()

    if data.empty:
        print(f"No data found for {ticker} — skipping.")
        return

    data.to_csv(file_path)
    print(f"Saved to {file_path}")


class DataGenerator:
    """
    Builds a Data object from historical asset prices.

    Responsibilities:
      1) Ensure raw CSV price data exists (download if missing)
      2) Load prices for chosen tickers and compute:
         - log returns
         - rolling annualized vols
         - correlation matrix (projected to positive definite)
      3) Generate maturities and strike grid (relative_strikes)
      4) Create a Data object
      5) Optionally compute and store basket option prices inside Data
    """

    def __init__(
        self,
        tickers,
        start_date,
        end_date,
        window,
        seed: int,
        maturity_days=TRADING_DAYS / 4,
        r=0.0,
        q=0.0,
        n_strikes_per_obs=5,
        m_min=0.7,
        m_max=1.3,
        n_paths=1_000_000,
    ):
        # --- dataset definition ---
        self.tickers = tickers
        self.window = window
        self.start_date = start_date
        self.end_date = end_date

        # master seed used to derive deterministic sub-streams
        self.seed = int(seed)

        # option pricing parameters stored in Data
        self.maturity_days = maturity_days   # if None => random maturities per date
        self.r = float(r)
        self.q = float(q)

        # strike grid config
        self.n_strikes_per_obs = int(n_strikes_per_obs)
        self.m_min = float(m_min)
        self.m_max = float(m_max)

        # Monte Carlo precision for labels (fixed per dataset)
        self.n_paths = int(n_paths)

        # SeedSequence can be used for deterministic child RNG streams
        self._ss = np.random.SeedSequence(self.seed)

        # Ensure raw data exists and then build the Data object
        self.ensure_all_assets()
        self._build_data_object()

    # ---------- Seed helpers ----------
    def _rng(self, stream_name: str) -> np.random.Generator:
        """
        Create a deterministic RNG for a specific logical stream (e.g., maturities, strikes)
        """
        name_hash = abs(hash(stream_name)) % (2**32)
        child = np.random.SeedSequence([self.seed, name_hash])
        return np.random.default_rng(child)

    # ---------- Build Data ----------
    def _build_data_object(self):
        """
        Load price series and compute derived statistics, then assemble Data.
        """
        prices, returns, rolling_vols, corr_matrix, corr_values = self._load_and_compute_stats()

        N = len(prices)  # number of observation dates after rolling window trimming
        maturities = self._generate_maturities(N)
        relative_strikes = self._generate_relative_strikes(N)

        self.data = Data(
            prices=prices,
            returns=returns,
            rolling_vols=rolling_vols,
            corr_matrix=corr_matrix,
            corr_values=corr_values,
            relative_strikes=relative_strikes,
            # If maturity_days is None => store random maturities array; else store fixed maturity int
            maturity=maturities if self.maturity_days is None else int(self.maturity_days),
            r=self.r,
            q=self.q,
            n_paths=self.n_paths,
        )

    def _generate_maturities(self, N: int):
        """
        Generate maturities in trading days.
          - fixed maturity if self.maturity_days is not None
          - random per date otherwise
        """
        if self.maturity_days is not None:
            return np.full(N, int(self.maturity_days), dtype=int)

        rng = self._rng("maturities")
        return rng.integers(1, TRADING_DAYS + 1, size=N)

    def _generate_relative_strikes(self, N: int):
        """
        Generate relative strikes per observation date.

        Shape:
          (N_dates, M_strikes)
        """
        rng = self._rng("relative_strikes")
        M = self.n_strikes_per_obs
        return rng.uniform(self.m_min, self.m_max, size=(N, M))

    def ensure_all_assets(self):
        """
        Ensure local CSVs exist for all tickers. Download missing ones via yfinance.
        """
        print("Checking available data files...")
        for ticker in self.tickers:
            file_path = paths.get_yf_data_path(ticker)
            if not file_path.exists():
                print(f"No local data for {ticker} — downloading...")
                download_asset(ticker)
            else:
                print(f"{ticker}.csv exists")

    def _load_and_compute_stats(self):
        """
        Load CSV price data, align dates, compute returns/vols/correlations.

        Output is trimmed so that prices/returns/vols share the same index
        after rolling window computation.
        """
        dfs = []
        for ticker in self.tickers:
            file_path = paths.get_yf_data_path(ticker)
            if not file_path.exists():
                raise FileNotFoundError(f"{file_path} not found")

            df = pd.read_csv(file_path, index_col=0, parse_dates=True)

            df = df.rename(columns={"Close": ticker})
            dfs.append(df)

        # Align all assets on the same date index and slice to requested period
        prices = pd.concat(dfs, axis=1).dropna()
        prices = prices.loc[self.start_date:self.end_date]

        # Log returns and rolling annualized volatility
        returns = np.log(prices / prices.shift(1)).dropna()
        rolling_vols = returns.rolling(self.window).std() * np.sqrt(TRADING_DAYS)
        rolling_vols = rolling_vols.dropna()

        if rolling_vols.empty:
            raise ValueError("Rolling vols empty — window too large or not enough data.")

        # Make sure all series share the same index after rolling window dropna
        prices = prices.loc[rolling_vols.index]
        returns = returns.loc[rolling_vols.index]

        # Estimate correlation and project to positive definite for Cholesky
        corr = returns.corr().to_numpy()
        corr_matrix = nearest_positive_definite(corr)

        # Store upper-triangular correlations
        n_stocks = len(self.tickers)
        corr_values = corr_matrix[np.triu_indices(n_stocks, k=1)]

        print("Loaded data and computed statistics.")
        return prices, returns, rolling_vols, corr_matrix, corr_values

    def price(
        self,
        option_type: Literal["best", "worst", "average"],
        use_percentage: bool,
        batch_size: int = 10,
        seed: int | None = None,
        store: bool = True,
    ):
        """
        Compute basket option prices using Monte Carlo and optionally store them in self.data.

        If seed is None:
          uses a deterministic seed derived from (self.seed, option_type, use_percentage, n_paths)
          so repeated runs yield identical labels.
        """
        print(f"Pricing {option_type}...")

        if seed is None:
            # deterministic per configuration
            seed = (
                abs(hash((int(self.seed), option_type, bool(use_percentage), int(self.n_paths))))
                % (2**31 - 1)
            )

        prices_df, counts, means = compute_basket_price_from_data(
            self.data,
            option_type=option_type,
            use_percentage=use_percentage,
            seed=int(seed),
            batch_size=int(batch_size),
            store=store,
        )

        return prices_df

    def save(self, filename: str | None = None):
        """
        Save the generated dataset to disk.

        """
        if filename is None:
            filename = paths.build_dataset_filename(
                tickers=self.tickers,
                start=self.start_date,
                end=self.end_date,
                window=self.window,
                maturity=self.maturity_days,
            )

        save_path = paths.get_dataset_path(filename)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        self.data.save(save_path)