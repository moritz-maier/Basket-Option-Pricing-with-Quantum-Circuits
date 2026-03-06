import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd
from jax import vmap, jit, random
from typing import Literal, Union, Tuple
from src.data_generation.utils_data import TRADING_DAYS

NOTIONAL = 1.0


def price_basket_mc(
    S0: jnp.ndarray,
    vols: jnp.ndarray,
    T: Union[int, float],
    corr_matrix: jnp.ndarray,
    relative_strike: Union[float, jnp.ndarray],
    r: float,
    q: Union[float, jnp.ndarray],
    key: jax.Array,
    option_type: Literal["best", "worst", "average"],
    use_percentage: bool = True,
    n_paths: int = 1_000_000,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Monte Carlo price for a "basket" option where the payoff depends on:
      - the best-performing asset (max return),
      - the worst-performing asset (min return),
      - or the average performance across assets.

    Returns:
      discounted_price: scalar option price
      counts:           how often each asset was selected (best/worst) among *in-the-money* paths
      mean_payoffs:     average payoff contribution per selected asset (conditional on being selected & ITM)
    """

    n_assets = S0.shape[0]

    # Cholesky factor used to correlate standard normals
    L = jnp.linalg.cholesky(corr_matrix)

    # Convert maturity from trading days to years
    T_years = T / TRADING_DAYS

    # Strike interpretation:
    # - use_percentage=True: K is already in "return" space
    # - use_percentage=False: K is an absolute strike; we map relative_strike * mean(S0)
    if use_percentage:
        K = relative_strike
    else:
        S_ref = jnp.mean(S0)
        K = relative_strike * S_ref

    # Dividend yield q: allow scalar or per-asset vector
    q_arr = jnp.asarray(q)
    if q_arr.ndim == 0:
        q_arr = jnp.full_like(S0, q_arr)

    # Draw correlated standard normals: Z_corr ~ N(0, corr_matrix)
    Z = random.normal(key, shape=(n_paths, n_assets))
    Z_corr = Z @ L.T

    # Risk-neutral GBM simulation
    drift = (r - q_arr - 0.5 * vols**2) * T_years
    diffusion = vols * jnp.sqrt(T_years) * Z_corr
    ST = S0 * jnp.exp(drift + diffusion)

    # Compute payoff depending on definition:
    # - Percentage mode uses returns ST/S0 and strike K in return space
    # - Absolute mode uses ST and strike K in price space
    if use_percentage:
        gross = ST / S0

        if option_type == "worst":
            idx = jnp.argmin(gross, axis=1)                 # chosen asset index per path
            chosen = gross[jnp.arange(n_paths), idx]        # chosen return
            payoffs = jnp.maximum(chosen - K, 0.0) * NOTIONAL

        elif option_type == "best":
            idx = jnp.argmax(gross, axis=1)
            chosen = gross[jnp.arange(n_paths), idx]
            payoffs = jnp.maximum(chosen - K, 0.0) * NOTIONAL

        elif option_type == "average":
            idx = jnp.full((n_paths,), -1)                  # sentinel: no single chosen asset
            chosen = jnp.mean(gross, axis=1)
            payoffs = jnp.maximum(chosen - K, 0.0) * NOTIONAL

        else:
            raise ValueError("Unknown option_type")

    else:
        if option_type == "worst":
            idx = jnp.argmin(ST, axis=1)
            chosen = ST[jnp.arange(n_paths), idx]
            payoffs = jnp.maximum(chosen - K, 0.0)

        elif option_type == "best":
            idx = jnp.argmax(ST, axis=1)
            chosen = ST[jnp.arange(n_paths), idx]
            payoffs = jnp.maximum(chosen - K, 0.0)

        elif option_type == "average":
            idx = jnp.full((n_paths,), -1)
            chosen = jnp.mean(ST, axis=1)
            payoffs = jnp.maximum(chosen - K, 0.0)

        else:
            raise ValueError("Unknown option_type")

    # --- Attribution stats (counts + mean payoffs per selected asset) ---
    # Only meaningful for best/worst (idx >= 0). For average, idx = -1 => excluded.
    # also restrict to in-the-money paths (payoff > 0) so "selection" is relevant.
    valid = ((idx >= 0) & (payoffs > 0.0)).astype(jnp.int32)

    safe_idx = jnp.maximum(idx, 0)
    one_hot = jax.nn.one_hot(safe_idx, n_assets, dtype=jnp.int32)

    # sel[p, a] = 1 if path p selected asset a AND payoff>0, else 0
    sel = one_hot * valid[:, None]

    # Counts: how often each asset was selected among ITM paths
    counts = jnp.sum(sel, axis=0)

    # Sum payoffs attributed to each selected asset, then average conditional on selection
    sum_payoffs = jnp.sum(sel * payoffs[:, None], axis=0)
    mean_payoffs = jnp.where(counts > 0, sum_payoffs / counts, 0.0)

    # Discount expected payoff under risk-neutral measure
    discounted_price = jnp.exp(-r * T_years) * jnp.mean(payoffs)

    return discounted_price, counts, mean_payoffs


# JIT-compile the pricing kernel.
price_basket_mc_jit = jit(
    price_basket_mc,
    static_argnames=["option_type", "use_percentage", "n_paths"],
)


def compute_basket_price(
    S0s: Union[np.ndarray, jnp.ndarray],
    vols: Union[np.ndarray, jnp.ndarray],
    maturities: Union[np.ndarray, jnp.ndarray],
    corr_matrix: Union[np.ndarray, jnp.ndarray],
    relative_strike: Union[np.ndarray, jnp.ndarray],
    seed: int,
    q: float,
    r: float,
    option_type: Literal["best", "worst", "average"],
    use_percentage: bool = True,
    n_paths: int = 1_000_000,
    batch_size: int = 1,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Vectorized wrapper around price_basket_mc_jit.

    Inputs are already "expanded" to match (N*M, ...) rows:
      - S0s:        (N*M, n_assets)
      - vols:       (N*M, n_assets)
      - maturities: (N*M,)
      - relative_strike: (N*M,)

    vmap over rows and optionally process in batches to control memory usage.
    """

    S0s = jnp.asarray(S0s)
    vols = jnp.asarray(vols)
    maturities = jnp.asarray(maturities)
    corr_matrix = jnp.asarray(corr_matrix)
    relative_strike = jnp.asarray(relative_strike).reshape((-1,))

    # One PRNG key per row so each sample uses independent randomness
    key = random.PRNGKey(seed)
    subkeys = random.split(key, S0s.shape[0])

    # Vectorize over the first dimension (rows); corr_matrix is shared for all rows
    vmapped = vmap(
        lambda S0, v, T, m, k: price_basket_mc_jit(
            S0, v, T, corr_matrix, m, r, q, k, option_type, use_percentage, n_paths
        ),
        in_axes=(0, 0, 0, 0, 0),
    )

    # Batch loop: helps avoid large peak memory with huge (N*M) or many paths
    prices_list, counts_list, means_list = [], [], []
    for i in range(0, S0s.shape[0], batch_size):
        p, c, mn = vmapped(
            S0s[i:i + batch_size],
            vols[i:i + batch_size],
            maturities[i:i + batch_size],
            relative_strike[i:i + batch_size],
            subkeys[i:i + batch_size],
        )
        prices_list.append(p)
        counts_list.append(c)
        means_list.append(mn)

    # Concatenate batch outputs back into full arrays
    prices = jnp.concatenate(prices_list, axis=0)
    counts = jnp.concatenate(counts_list, axis=0)
    means = jnp.concatenate(means_list, axis=0)

    return prices, counts, means


def compute_basket_price_from_data(
    data,  # Data
    option_type: Literal["best", "worst", "average"],
    use_percentage: bool,
    seed: int = 0,
    batch_size: int = 10,
    store: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Convenience function operating on your Data object:
      1) expand (dates x strikes) into flat arrays
      2) price each row using compute_basket_price
      3) reshape back to (N_dates, M_strikes) and return as DataFrame
      4) optionally store results in Data via data.store_result(...)
    """

    # Expand inputs to (N*M, ...) rows so each row corresponds to (date, strike)
    S0_exp, vols_exp, T_exp, m_exp, obs_idx, strike_idx = data.expand()
    N = data.prices.shape[0]
    M = data.relative_strikes.shape[1]

    n_paths = int(data.n_paths)

    prices, counts, means = compute_basket_price(
        S0s=S0_exp,
        vols=vols_exp,
        maturities=T_exp,
        corr_matrix=data.corr_matrix,
        relative_strike=m_exp,
        seed=int(seed),
        q=float(data.q),
        r=float(data.r),
        option_type=option_type,
        use_percentage=bool(use_percentage),
        n_paths=n_paths,
        batch_size=int(batch_size),
    )

    # Reshape flat prices back to (N_dates, M_strikes)
    prices_mat = np.asarray(prices).reshape(N, M)
    prices_df = pd.DataFrame(
        prices_mat,
        index=data.prices.index,
        columns=[f"strike_{j}" for j in range(M)],
    )

    # Store back into Data so later ML can reuse the computed labels/prices
    if store:
        data.store_result(
            option_type=option_type,
            use_percentage=use_percentage,
            prices_df=prices_df,
            counts=np.asarray(counts),
            means=np.asarray(means),
        )

    return prices_df, np.asarray(counts), np.asarray(means)