import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def robust_ylim_from_groups(
    df,
    y_col,
    group_cols=None,          # e.g. ["option_type", "hue"] or ["noiseScale", "hue"]
    whisker_k=1.25,           # whisker length in IQR units
    rel_pad=0.05,             # relative padding added on top of robust bounds
    abs_pad=0.00,             # absolute padding (useful when values are nearly constant)
):
    """
    Compute robust y-axis limits for plots.

    Idea:
    - Use "boxplot whisker" logic (based on IQR) to avoid extreme outliers
      dominating the y-axis range.
    - Optionally compute these bounds per group (e.g. per model/option_type)
      and then combine them to get a global y-range suitable for multi-panel plots.

    Returns:
        (ymin, ymax) tuple, or None if no valid values exist.
    """

    def whisker_bounds(vals):
        """
        Compute lower/upper bounds using an IQR-based whisker rule.

        Steps:
        - Remove NaNs
        - Compute Q1, Q3 and IQR
        - Define "fences": [Q1 - k*IQR, Q3 + k*IQR]
        - Clip to the nearest actual observations inside fences (robust min/max)

        Returns:
            (low, high) or None if vals are empty.
        """
        vals = np.asarray(vals, dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            return None

        q1, q3 = np.quantile(vals, [0.25, 0.75])
        iqr = q3 - q1

        # If everything is basically constant (IQR=0), fall back to min/max
        if iqr == 0:
            return vals.min(), vals.max()

        low_f = q1 - whisker_k * iqr
        high_f = q3 + whisker_k * iqr

        # Choose robust extremes *within* the fences if possible,
        # otherwise fall back to true min/max.
        low = vals[vals >= low_f].min() if np.any(vals >= low_f) else vals.min()
        high = vals[vals <= high_f].max() if np.any(vals <= high_f) else vals.max()
        return low, high

    # ---- No grouping: compute bounds from the full column ----
    if group_cols is None:
        b = whisker_bounds(df[y_col].values)
        if b is None:
            return None
        low, high = b

    # ---- Grouped: compute bounds per group and merge ----
    else:
        bounds = []
        for _, g in df.groupby(list(group_cols), dropna=True, observed=True):
            b = whisker_bounds(g[y_col].values)
            if b is not None:
                bounds.append(b)

        if not bounds:
            return None

        # Use the most extreme robust bounds across all groups
        low = min(b[0] for b in bounds)
        high = max(b[1] for b in bounds)

    # ---- Add padding so the plot doesn't touch edges ----
    rng = high - low
    pad = max(rel_pad * rng, abs_pad) if rng > 0 else abs_pad
    return (low - pad, high + pad)


def apply_ylim(ax, ylim):
    """
    Apply y-limits to a matplotlib axis, if provided.
    """
    if ylim is not None:
        ax.set_ylim(*ylim)


def fmt_int_series(s):
    """
    Convert a Series to a clean integer string representation:

    - coercion to numeric
    - replace infinities
    - round to nearest integer
    - convert to pandas nullable Int64 and then to string
    - fill NaNs with empty string

    """
    s = pd.to_numeric(s, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    return s.round().astype("Int64").astype("string").fillna("")


def add_hue(df):
    """
    Add a human-readable 'hue' label column for plotting.

    This column is meant for seaborn's hue=... parameter so that each
    model configuration gets its own legend entry.

    Expected input columns (depending on model type):
      - 'model'
      - classic: 'units' (and optionally dropout)
      - quantum: 'n_trainable_blocks', 'layers'
    """
    df = df.copy()

    # Ensure datetime typing (useful for sorting/filtering later)
    df["start_date"] = pd.to_datetime(df["start_date"])

    quantum = df["model"].str.contains("quantum", case=False, na=False)

    classic = df["model"].str.contains("classic", case=False, na=False)
    culkin = df["model"].str.contains("culkin", case=False, na=False)

    # Basis-Label für alle Classic-Modelle
    df.loc[classic, "hue"] = (
            df.loc[classic, "model"].astype("string")
            + ", u=" + fmt_int_series(df.loc[classic, "units"])
    )

    # Nur für Culkin zusätzlich Dropout anhängen
    df.loc[culkin, "hue"] = (
            df.loc[culkin, "hue"]
            + ", drop=" + df.loc[culkin, "dropout"].astype("string")
    )

    enc = df.loc[quantum, "encoding_base"]

    enc_label = np.where(
        enc == 1,
        "unary",
        np.where(
            enc == 3,
            "ternary",
            enc.astype("Int64").astype(str)
        )
    )

    # Quantum label: "quantum, B=20, L=2"
    df.loc[quantum, "hue"] = (
        df.loc[quantum, "model"].astype("string")
        + " (" + enc_label + ")"
        + ", L=" + fmt_int_series(df.loc[quantum, "layers"])
        + ", B=" + fmt_int_series(df.loc[quantum, "n_trainable_blocks"])
    )

    return df


def make_hue_order_and_palette(df):
    """
    Build:
      - hue_order: a stable ordering for legend entries
      - palette: a dict mapping hue labels -> colors

    Strategy:
    - Split classic and quantum models
    - Sort each group by key hyperparameters (units/dropout or layers/blocks)
    - Combine back into one ordering
    - Assign a separate colormap family:
        * quantum -> Blues
        * classic -> Oranges

    Returns:
        (hue_order, palette)
    """
    df = df.copy()

    classic = df["model"].str.contains("classic", case=False, na=False)
    quantum = df["model"].str.contains("quantum", case=False, na=False)

    # ---- Sort classic configs by (model name, units, dropout) ----
    df_classic = (
        df[classic]
        .assign(
            units_num=pd.to_numeric(df.loc[classic, "units"], errors="coerce"),
            drop_num=pd.to_numeric(df.loc[classic, "dropout"], errors="coerce"),
        )
        .sort_values(["model", "units_num", "drop_num"])
    )

    # ---- Sort quantum configs by (model name, layers, blocks) ----
    df_quantum = (
        df[quantum]
        .assign(
            encoding_base_num = pd.to_numeric(df.loc[quantum, "encoding_base"], errors="coerce"),
            layers_num=pd.to_numeric(df.loc[quantum, "layers"], errors="coerce"),
            blocks_num=pd.to_numeric(df.loc[quantum, "n_trainable_blocks"], errors="coerce"),
        )
        .sort_values(["model","encoding_base_num" ,"layers_num", "blocks_num"])
    )

    # Combine to get a consistent final order
    df_sorted = pd.concat([df_classic, df_quantum])

    # Keep first occurrence order, remove duplicates
    hue_order = df_sorted["hue"].dropna().drop_duplicates().tolist()

    # ---- Build palette dictionary (hue -> color) ----
    q = [h for h in hue_order if "quantum" in str(h).lower()]
    c = [h for h in hue_order if "classic" in str(h).lower()]

    palette = {}

    # Quantum colors
    for i, h in enumerate(q):
        h_lower = str(h).lower()

        if "ternary" in h_lower:
            palette[h] = sns.color_palette("Greens", 6)[i % 6]
        elif "unary" in h_lower:
            palette[h] = sns.color_palette("Blues", 6)[i % 6]
        else:
            palette[h] = sns.color_palette("Purples", 6)[i % 6]

    # Classic colors
    palette.update(
        dict(zip(c, sns.color_palette("Oranges", len(c) or 1)))
    )

    return hue_order, palette


def hue_order_for(df, hue_order, col="hue"):
    """
    Filter a global hue_order down to the labels that are present in df.
    """
    present = set(df[col].dropna().astype(str).unique())
    return [h for h in hue_order if str(h) in present]


def finish(ax, title, xlabel, ylabel, legend=True):
    """
    Final formatting helper for matplotlib/seaborn plots:
    - set axis labels and title
    - optionally hide legend
    """
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # ax.figure.suptitle(suptitle, fontsize=15)
    ax.set_title(title, fontsize=11)

    if legend:
        ax.legend(title=None)
    else:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

def savefig_if(path):
    """Save current figure as PDF if a path/name is provided (without extension)."""
    if path:
        plt.savefig(f"{path}.pdf", bbox_inches="tight")

def boxplot_metric(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    hue_order: list[str],
    palette,
    title: str,
    xlabel: str,
    ylabel: str,
    group_cols: list[str],
    save_as: str | None = None,
    figsize: tuple[int, int] = (12, 6),
    whisker_k: float = 1.25,
):
    """
    Draw a seaborn boxplot with robust y-limits computed group-wise.

    Parameters
    ----------
    df : pd.DataFrame
        Input table containing columns x, y, and hue.
    x, y : str
        Column names for x-axis and y-axis.
    hue : str
        Column name used for grouping and legend colors.
    hue_order : list[str]
        Fixed order for hue categories to keep legend stable across plots.
    palette : dict or seaborn palette
        Mapping from hue values to colors.
    title, xlabel, ylabel : str
        Plot annotations.
    group_cols : list[str]
        Columns used to compute robust y-limits per group (reduces outlier impact).
    save_as : str | None
        If provided, saves figure as f"{save_as}.pdf".
    figsize : tuple[int, int]
        Figure size in inches.
    whisker_k : float
        Whisker multiplier for robust y-limit selection (similar to boxplot whiskers).
    """
    fig, ax = plt.subplots(figsize=figsize)

    sns.boxplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        hue_order=hue_order,
        palette=palette,
        ax=ax,
    )

    # Robust y-limits based on IQR/whiskers to avoid extreme outliers dominating scale
    ylim = robust_ylim_from_groups(df, y_col=y, group_cols=group_cols, whisker_k=whisker_k)
    apply_ylim(ax, ylim)

    finish(
        ax,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        legend=True,
    )

    savefig_if(save_as)
    plt.show()
    return ax