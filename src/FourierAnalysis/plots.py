import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from src.visualize.metrics import get_metrics


def plot_dominant_spectrum(spectrum):
    """
    Plot a 2D Fourier spectrum as a heatmap using frequency axes.

    This function assumes `spectrum` is already a 2D array where both
    dimensions represent frequency bins. The frequency axis is shown
    as a symmetric grid from -N/2 to N/2.

    Parameters
    ----------
    spectrum : np.ndarray
        2D spectrum array of shape (N, N), typically magnitude values.
    """
    N = spectrum.shape[0]

    # Frequency coordinates (symmetric around 0)
    freq_x = np.arange(-N // 2, N // 2)
    freq_y = np.arange(-N // 2, N // 2)
    FreqX, FreqY = np.meshgrid(freq_x, freq_y)

    fig, ax = plt.subplots(figsize=(7, 6))

    mesh = ax.pcolormesh(FreqX, FreqY, spectrum, shading="auto")

    # Title and axis labels
    fig.suptitle("Normalized Spectrum", fontsize=15)
    ax.set_xlabel(r"$\omega_1$")
    ax.set_ylabel(r"$\omega_2$")

    # Colorbar shows amplitude scale
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Amplitude")

    plt.tight_layout()
    plt.show()


def plot_spectrum_projection(spectrum, dim1=0, dim2=1, normalize=True, save: str = None):
    """
    Plot a 2D projection of a multi-dimensional Fourier spectrum.

    The function sums over all dimensions except `dim1` and `dim2`
    to obtain a 2D representation of the spectrum. Optionally,
    the projection can be normalized and saved as a PDF file.

    Parameters
    ----------
    spectrum : np.ndarray
        d-dimensional Fourier spectrum (magnitude values).
    dim1 : int
        First dimension to keep in the projection.
    dim2 : int
        Second dimension to keep in the projection.
    normalize : bool
        If True, scale the projection to the interval [0, 1].
    save : str or None
        If provided, the figure is saved as "<save>.pdf".
    """

    # ------------------------------------------------------------
    # Reduce spectrum to 2D by summing over all other dimensions
    # ------------------------------------------------------------
    axes = tuple(i for i in range(spectrum.ndim) if i not in (dim1, dim2))
    spec_2d = np.sum(spectrum, axis=axes)

    # ------------------------------------------------------------
    # Optional normalization for comparability across experiments
    # ------------------------------------------------------------
    if normalize:
        spec_2d = spec_2d / (np.max(spec_2d) + 1e-12)

    # ------------------------------------------------------------
    # Construct symmetric frequency grid [-N/2, ..., N/2)
    # ------------------------------------------------------------
    N = spec_2d.shape[0]
    freq = np.arange(-N // 2, N // 2)
    FreqX, FreqY = np.meshgrid(freq, freq)

    # ------------------------------------------------------------
    # Create heatmap-style visualization
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    mesh = ax.pcolormesh(FreqX, FreqY, spec_2d, shading='auto')

    # Axis labels reflect selected frequency dimensions
    ax.set_xlabel(fr"$\omega_{{{dim1+1}}}$")
    ax.set_ylabel(fr"$\omega_{{{dim2+1}}}$")

    # ------------------------------------------------------------
    # Colorbar indicates spectral amplitude
    # ------------------------------------------------------------
    cbar = fig.colorbar(mesh, ax=ax)
    if normalize:
        cbar.set_label("Normalized amplitude")
    else:
        cbar.set_label("Amplitude")

    plt.tight_layout()

    # ------------------------------------------------------------
    # Optional: save figure to file
    # ------------------------------------------------------------
    if save is not None:
        plt.savefig(
            f"{save}.pdf",
            bbox_inches="tight",
        )

    plt.show()