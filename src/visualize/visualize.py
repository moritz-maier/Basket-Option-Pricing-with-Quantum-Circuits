import matplotlib.pyplot as plt
from pennylane import numpy as pnp 
import pandas as pd
from src.visualize.metrics import get_metrics
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_scatter(y_train, predictions_train, y_test, predictions_test, title="", save=None):

    metrics = get_metrics(y_train, predictions_train, y_test, predictions_test)

    plt.figure(figsize=(8, 6))
    plt.scatter(predictions_test, y_test, facecolor="white", edgecolor="black")
    plt.plot([pnp.min(y_test), pnp.max(y_test)], [pnp.min(y_test), pnp.max(y_test)], c="red")
    # plt.plot([0, 100], [0,100], c="red")
    plt.xlabel("Vorhergesagter Preis")
    plt.ylabel("Wahrer Preis")
    plt.title(title)
    plt.suptitle(f"R²_train = {metrics["train"]["r2"]:.4f}, RMSE_train = {metrics["train"]["rmse"]:.4f} | "
                 f"R²_test = {metrics["test"]["r2"]:.4f}, RMSE_test = {metrics["test"]["rmse"]:.4f}")
    
    plt.tight_layout()
    if save is not None:
        plt.savefig(save, bbox_inches='tight', dpi=300)
        print(f"Plot gespeichert unter: {save}")

    # plt.title(f"{title} R²_train = {r2_train:.4f}, RMSE_train = {rmse_train:.4f}, R²_test = {r2_test:.4f}, RMSE_test = {rmse_test:.4f}") 
    plt.show()

def get_result_from_group(
    df: pd.DataFrame,
    filter_col: str,
    filter_value,
    group_cols: list,
    result_col: str,
    index_to_use: int = 0):
    representative_results = (
        df[df[filter_col] == filter_value]
        .groupby(group_cols, as_index=False)
        .apply(lambda g: g.sample(1))
        .reset_index(drop=True)[group_cols + [result_col]]
    )
    res_row = representative_results.loc[index_to_use]
    res = res_row[result_col]

    return res


def plot_result_scatter(
    df: pd.DataFrame,
    reload_model_fn,
    plot_scatter_fn,
    filter_col: str = "trainable_blocks",
    filter_value: int = 3,
    group_cols: list = ["option_type", "corr"],
    result_col: str = "result",
    index_to_use: int = 3,
    verbose: bool = True,
    plot_title: str = "",
    save_path: str = None
):


    representative_results = (
        df[df[filter_col] == filter_value]
        .groupby(group_cols, as_index=False)
        .apply(lambda g: g.sample(1))
        .reset_index(drop=True)[group_cols + [result_col]]
    )


    res_row = representative_results.loc[index_to_use]
    res = res_row[result_col]


    model, x_train, x_test, y_train, y_test = reload_model_fn(res)


    res.weights = res.weights
    predictions_test = model.get_predictions(weights=res.weights, x_test=x_test)
    predictions_train = model.get_predictions(weights=res.weights, x_test=x_train)

    if verbose and hasattr(res, "train_R2"):
        print(f"Train R²: {res.train_R2:.4f}")

    group_info = " | ".join([f"{col}={res_row[col]}" for col in group_cols])
    full_title = f"{group_info}"
    if plot_title:
        full_title += f" — {plot_title}"

    plot_scatter_fn(
        y_train,
        predictions_train,
        y_test,
        predictions_test,
        title=full_title,
        save=save_path
    )

    return {
        "model": model,
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "pred_train": predictions_train,
        "pred_test": predictions_test,
        "res": res,
        "representative_results": representative_results,
        "group_info": group_info
    }


def plot_grouped_bar(
    df: pd.DataFrame,
    group_cols: list,
    value_col: str,
    hue_cols: list,
    x_col: str,
    title: str = "Gruppierte Balkengrafik",
    figsize: tuple = (8, 5),
    fmt: str = "%.3f"
):

    group_means = df.groupby(group_cols)[value_col].mean().reset_index()

    group_means["hue_combined"] = group_means[hue_cols[0]].astype(str)
    for col in hue_cols[1:]:
        group_means["hue_combined"] += " | " + col + "=" + group_means[col].astype(str)

    plt.figure(figsize=figsize)
    ax = sns.barplot(
        data=group_means,
        x=x_col,
        y=value_col,
        hue="hue_combined",
    )


    for container in ax.containers:
        ax.bar_label(container, fmt=fmt, label_type="edge", fontsize=6, padding=2)

    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_grouped_boxplot(
    df: pd.DataFrame,
    group_cols: list,
    value_col: str,
    hue_cols: list,
    x_col: list,
    title: str = "Gruppierte Boxplot-Grafik",
    figsize: tuple = (8, 5),
    bottom: float = None,
    top: float = None
):
    df = df.copy()

    # ------------------------------
    # 1) X-Kombination mit Reihenfolge
    # ------------------------------
    df["x_combined"] = df[x_col].astype(str).agg(" | ".join, axis=1)

    # Reihenfolge der X-Werte = in der Reihenfolge wie sie auftreten
    x_order = df["x_combined"].drop_duplicates().tolist()
    df["x_combined"] = pd.Categorical(df["x_combined"], categories=x_order, ordered=True)

    # ------------------------------
    # 2) Hue-Kombination mit SORTIERUNG
    # ------------------------------

    # Hue-Spalten sortieren (numerisch wenn möglich, sonst alphabetisch)
    def sort_mixed(values):
        """Sort numerically if possible, otherwise alphabetically."""
        try:
            return sorted(values, key=lambda x: float(x))
        except ValueError:
            return sorted(values)

    # Sortierte Kategorien je Hue-Spalte
    hue_orders = {
        col: sort_mixed(df[col].astype(str).unique())
        for col in hue_cols
    }

    # Spalten als kategorisch mit obiger Sortierung setzen
    for col in hue_cols:
        df[col] = pd.Categorical(df[col].astype(str), categories=hue_orders[col], ordered=True)

    # Kombinierter Hue-Wert
    df["hue_combined"] = df[hue_cols].astype(str).agg(" | ".join, axis=1)

    # Reihenfolge der vollständigen Kombinationen (sortiert aufgrund der Kategorien!)
    hue_combined_order = (
        df[hue_cols]
        .drop_duplicates()
        .sort_values(by=hue_cols)
        .astype(str)
        .agg(" | ".join, axis=1)
        .tolist()
    )

    df["hue_combined"] = pd.Categorical(df["hue_combined"],
                                        categories=hue_combined_order,
                                        ordered=True)

    # ------------------------------
    # 3) Plot
    # ------------------------------
    plt.figure(figsize=figsize)

    ax = sns.boxplot(
        data=df,
        x="x_combined",
        y=value_col,
        hue="hue_combined",
        order=x_order,
        hue_order=hue_combined_order,
        showcaps=True,
        whis=1.5,
        fliersize=3,
        linewidth=1
    )

    ax.set_ylim(bottom=bottom, top=top)

    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

import plotly.graph_objects as go
def plot_3d(x1, x2, y_true, y_pred):
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=x1,
        y=x2,
        z=y_true,
        mode='markers',
        marker=dict(size=3),
        name='y_test'
    ))

    fig.add_trace(go.Scatter3d(
        x=x1,
        y=x2,
        z=y_pred,
        mode='markers',
        marker=dict(size=3),
        name='predictions_test'
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='Feature 1',
            yaxis_title='Feature 2',
            zaxis_title='Target / Prediction'
        ),
        width=900,
        height=700
    )

    fig.show()



