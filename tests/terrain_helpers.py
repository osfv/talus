import numpy as np


def spectral_fbm(beta: float, seed: int = 0, n: int = 64) -> np.ndarray:
    """Synthetic heightmap with power spectrum ~ k^-beta, scaled into [0.25, 0.75]."""
    rng = np.random.default_rng(seed)
    f = np.sqrt(np.fft.fftfreq(n)[:, None] ** 2 + np.fft.fftfreq(n)[None, :] ** 2) * n
    amp = np.where(f > 0, np.maximum(f, 1e-9) ** (-beta / 2), 0.0)
    field = np.real(np.fft.ifft2(amp * np.exp(2j * np.pi * rng.random((n, n)))))
    field = (field - field.min()) / (field.max() - field.min())
    return 0.25 + 0.5 * field
