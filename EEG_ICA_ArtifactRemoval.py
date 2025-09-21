#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG Blink Artifact Removal using ICA
-----------------------------------

This script simulates EEG signals with eye blink artifacts and demonstrates
Independent Component Analysis (ICA) for artifact removal.

Author: Sahar Jahani
Date: 2025-09-19
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from sklearn.metrics import mean_squared_error
from scipy.stats import kurtosis


def simulate_eeg(fs: int = 250, duration: int = 3) -> tuple:
    """
    Simulate clean EEG (theta, alpha, beta) + Gaussian noise.

    Parameters
    ----------
    fs : int
        Sampling frequency in Hz.
    duration : int
        Duration in seconds.

    Returns
    -------
    t : ndarray
        Time vector.
    clean_eeg : ndarray
        Simulated clean EEG.
    """
    t = np.linspace(0, duration, fs * duration)
    theta = 0.5 * np.sin(2 * np.pi * 6 * t)      # 6 Hz
    alpha = 0.3 * np.sin(2 * np.pi * 10 * t)     # 10 Hz
    beta  = 0.2 * np.sin(2 * np.pi * 20 * t)     # 20 Hz
    brain_noise = 0.05 * np.random.randn(len(t))
    clean_eeg = theta + alpha + beta + brain_noise
    return t, clean_eeg


def simulate_blinks(t: np.ndarray, fs: int, duration: int, n_blinks: int = 8) -> np.ndarray:
    """
    Simulate eye blink artifacts as Gaussian bumps at random times.

    Parameters
    ----------
    t : ndarray
        Time vector.
    fs : int
        Sampling frequency in Hz.
    duration : int
        Signal duration in seconds.
    n_blinks : int
        Number of blinks to insert.

    Returns
    -------
    blink : ndarray
        Blink artifact signal.
    """
    blink = np.zeros(len(t))
    blink_times = np.random.uniform(0.5, duration - 0.5, size=n_blinks)

    for bt in blink_times:
        sigma = np.random.uniform(0.01, 0.05)   # blink width
        amp   = np.random.uniform(1.5, 3)       # blink strength
        start = int((bt - 0.15) * fs)
        end   = int((bt + 0.15) * fs)
        blink[start:end] += amp * np.exp(-((t[start:end] - bt) ** 2) / (2 * sigma ** 2))

    return blink


def run_ica(X: np.ndarray, clean_eeg: np.ndarray) -> tuple:
    """
    Run ICA to remove blink artifacts from EEG.

    Parameters
    ----------
    X : ndarray
        Multichannel EEG (channels x samples).
    clean_eeg : ndarray
        Ground truth clean EEG (for evaluation).

    Returns
    -------
    den_eeg : ndarray
        Denoised EEG (first channel).
    mse_before : float
        Mean squared error before ICA.
    mse_after : float
        Mean squared error after ICA.
    """
    ica = FastICA(n_components=X.shape[0], random_state=0)
    S = ica.fit_transform(X.T)       # sources (samples x comps)
    A = ica.mixing_                  # mixing matrix (channels x comps)

    # Identify blink component via kurtosis
    artifact_idx = np.argmax(np.abs(kurtosis(S, axis=0)))
    S[:, artifact_idx] = 0  # remove artifact

    # Reconstruct EEG
    X_denoised = (S @ A.T + ica.mean_).T
    den_eeg = X_denoised[0]

    mse_before = mean_squared_error(clean_eeg, X[0])
    mse_after  = mean_squared_error(clean_eeg, den_eeg)

    return den_eeg, mse_before, mse_after


def plot_results(t: np.ndarray, ch1: np.ndarray, clean_eeg: np.ndarray, den_eeg: np.ndarray):
    """
    Plot EEG before and after ICA artifact removal.
    """
    # Match scales for fair comparison
    den_plot = den_eeg * (np.std(clean_eeg) / np.std(den_eeg))

    plt.figure(figsize=(12, 6))
    plt.plot(t, ch1, label="Frontal EEG (with blinks)", linewidth = 2)
    plt.plot(t, clean_eeg, label="True Clean EEG", linestyle="dashed")
    plt.plot(t, den_plot, label="Denoised EEG (ICA)", linewidth=2, linestyle="-.")

    plt.xlabel("Time (s)", fontsize=16)
    plt.ylabel("Amplitude", fontsize=16)
    plt.title("EEG Blink Artifact Removal using ICA", fontsize=16, fontweight="bold")
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    # Parameters
    fs = 250
    duration = 3

    # Simulate signals
    t, clean_eeg = simulate_eeg(fs, duration)
    blink = simulate_blinks(t, fs, duration, n_blinks=8)

    # Multichannel EEG with different blink weights
    ch1 = clean_eeg + 1.0 * blink + 0.01 * np.random.randn(len(t))   # frontal (blink-heavy)
    ch2 = clean_eeg + 0.5 * blink + 0.01 * np.random.randn(len(t))   # medium blink
    ch3 = clean_eeg + 0.2 * blink + 0.01 * np.random.randn(len(t))   # occipital (blink-light)
    X = np.vstack([ch1, ch2, ch3])

    # Run ICA
    den_eeg, mse_before, mse_after = run_ica(X, clean_eeg)

    # Print results
    print(f"MSE Before ICA: {mse_before:.4f}")
    print(f"MSE After ICA : {mse_after:.4f}")

    # Plot
    plot_results(t, ch1, clean_eeg, den_eeg)


if __name__ == "__main__":
    main()
