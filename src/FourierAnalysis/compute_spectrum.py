import os
import sys
project_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
from sklearn.preprocessing import StandardScaler
from pynufft import NUFFT


def compute_spectrum(x_data, y_data, N=31):
    """
    Compute the multi-dimensional Fourier spectrum of a function
    sampled at irregular points using a Non-Uniform FFT (NUFFT).

    Parameters
    ----------
    x_data : np.ndarray
        Input features of shape (n_samples, d).
    y_data : np.ndarray
        Target values of shape (n_samples,).
    N : int
        Grid resolution per dimension for the FFT.

    Returns
    -------
    spectrum : np.ndarray
        Magnitude of the reconstructed Fourier spectrum
        on an N^d grid.
    """

    n_samples, d = x_data.shape

    # Standardize inputs to zero mean and unit variance
    x_data = StandardScaler().fit_transform(x_data)

    # Normalize coordinates to the interval [-π, π]
    x_min = x_data.min(axis=0)
    x_max = x_data.max(axis=0)
    om = 2 * np.pi * (x_data - x_min) / (x_max - x_min + 1e-12) - np.pi

    # Initialize and plan NUFFT object
    # Nd: output grid size
    # Kd: oversampled grid size
    # Jd: interpolation neighborhood size
    nufft_obj = NUFFT()
    nufft_obj.plan(
        om=om,
        Nd=(N,) * d,
        Kd=(2 * N,) * d,
        Jd=(6,) * d
    )

    # Convert target to complex type
    y_data_complex = y_data.astype(np.complex64)

    # Compute adjoint NUFFT (approximate inverse transform)
    # Take magnitude of complex spectrum
    spectrum = np.abs(nufft_obj.adjoint(y_data_complex))

    return spectrum


def extract_dominant_frequencies(spectrum, n_freqs=20):
    """
    Extract the dominant frequency vectors from a Fourier spectrum.

    Parameters
    ----------
    spectrum : np.ndarray
        Multi-dimensional magnitude spectrum.
    n_freqs : int
        Number of dominant frequency components to extract.

    Returns
    -------
    dominant_omega : list of tuples
        List of frequency vectors corresponding to the largest
        spectral magnitudes.
    """

    d = spectrum.ndim
    N = spectrum.shape[0]

    # Construct symmetric frequency grids for each dimension
    freq_grids = [np.arange(-N // 2, N // 2) for _ in range(d)]

    # Identify indices of largest magnitude components
    flat_indices = np.argsort(spectrum.flatten())[-n_freqs:]
    multi_indices = np.unravel_index(flat_indices, spectrum.shape)

    dominant_omega = []

    # Map array indices back to frequency coordinates
    for idx in zip(*multi_indices):
        omega_vec = tuple(grid[i] for grid, i in zip(freq_grids, idx))
        dominant_omega.append(omega_vec)

    return dominant_omega