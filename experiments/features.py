"""
Scalar feature functionals for reconstructed fields.

Per the identification design: M(t) = f_max(t) stays a global scalar, and
the candidate library Theta is built from spatially-aggregated functionals
of each reconstructed field (rho_h, Phi_parallel, Phi_plus, S_work,
S_current) rather than literal pointwise PDE terms. This module is that
functional library.

Every feature is a pure function: grid (n, n) array in -> scalar out (or
a pair of grids in -> scalar out, for cross-field terms). Kept
independently testable and free of any simulation/reconstruction
machinery, so the exporter and the eventual regression stage never need
to know how a feature is computed -- only that it takes a snapshot's
fields and returns a number.

All spatial derivatives use periodic finite differences (np.roll), which
is the correct convention for this periodic box and is consistent with
how the fields themselves were reconstructed.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit


# ---------------------------------------------------------------------------
# Single-field statistical functionals
# ---------------------------------------------------------------------------

def field_mean(u: np.ndarray) -> float:
    return float(np.mean(u))


def field_var(u: np.ndarray) -> float:
    return float(np.var(u))


def field_skew(u: np.ndarray) -> float:
    return float(stats.skew(u.ravel()))


def field_quantile(u: np.ndarray, q: float = 0.9) -> float:
    return float(np.quantile(u, q))


def bimodality_coefficient(u: np.ndarray) -> float:
    """
    Sarle's bimodality coefficient: BC = (skew^2 + 1) / (kurtosis_excess +
    3*(n-1)^2 / ((n-2)*(n-3))). BC > 5/9 is a common (heuristic, not a
    formal test) threshold suggesting bimodality relative to a uniform
    distribution. Used as one of the independent onset-detection signals
    for density (per the design's "density bimodality" criterion) -- NOT
    as a capacity/dissipation feature.
    """
    x = u.ravel()
    n = x.size
    if n < 4:
        return float("nan")
    skew = stats.skew(x)
    kurt = stats.kurtosis(x, fisher=True)  # excess kurtosis
    denom = kurt + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    if denom == 0:
        return float("nan")
    return float((skew ** 2 + 1.0) / denom)


# ---------------------------------------------------------------------------
# Spatial-derivative functionals (periodic finite differences)
# ---------------------------------------------------------------------------

def _gradient(u: np.ndarray, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Periodic central-difference gradient. Returns (du/dx, du/dy)."""
    gx = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2.0 * spacing)
    gy = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2.0 * spacing)
    return gx, gy


def _laplacian(u: np.ndarray, spacing: float) -> np.ndarray:
    """Periodic 5-point discrete Laplacian."""
    return (
        np.roll(u, -1, axis=1) + np.roll(u, 1, axis=1) +
        np.roll(u, -1, axis=0) + np.roll(u, 1, axis=0) - 4.0 * u
    ) / (spacing ** 2)


def gradient_squared_mean(u: np.ndarray, spacing: float) -> float:
    """<|grad u|^2> over the domain."""
    gx, gy = _gradient(u, spacing)
    return float(np.mean(gx ** 2 + gy ** 2))


def laplacian_squared_mean(u: np.ndarray, spacing: float) -> float:
    """<|laplacian u|^2> over the domain."""
    lap = _laplacian(u, spacing)
    return float(np.mean(lap ** 2))


# ---------------------------------------------------------------------------
# Fourier-space functionals: low-q structure factor and correlation length
# ---------------------------------------------------------------------------

def field_structure_factor_low_q(u: np.ndarray, L: float, n_low: int = 3) -> float:
    """
    Mean power spectral density of the field's FLUCTUATIONS (mean
    subtracted) over the lowest n_low nonzero angular-wavenumber shells.
    High values indicate large-scale spatial structure/heterogeneity in u.
    """
    n = u.shape[0]
    u0 = u - u.mean()
    uk = np.fft.fft2(u0)
    power = (np.abs(uk) ** 2) / (n * n)

    dq = 2.0 * np.pi / L
    freqs = np.fft.fftfreq(n, d=L / n) * 2.0 * np.pi
    KX, KY = np.meshgrid(freqs, freqs, indexing="ij")
    Kmag = np.sqrt(KX ** 2 + KY ** 2)

    low_mask = (Kmag > 1e-10) & (Kmag <= n_low * dq)
    if not np.any(low_mask):
        return float("nan")
    return float(power[low_mask].mean())


def correlation_length(u: np.ndarray, L: float, n_fit_points: int = 10) -> float:
    """
    Estimates a correlation length xi by fitting exp(-r/xi) to the
    radially-averaged spatial autocorrelation of u's fluctuations
    (Wiener-Khinchin: autocorrelation = IFFT(|FFT(u - mean(u))|^2)).

    Returns NaN rather than a number whenever the fit is not trustworthy:
    too few valid radial bins, a non-finite or out-of-range fit result, or
    a failed optimization. This is deliberate -- a bad xi fit silently
    passed downstream is worse than a missing feature the regression
    stage has to handle explicitly. Expect frequent NaN in dilute,
    unstructured systems, where there is no real correlation length to
    measure.
    """
    n = u.shape[0]
    spacing = L / n
    u0 = u - u.mean()

    F = np.fft.fft2(u0)
    power = np.abs(F) ** 2
    autocorr = np.fft.ifft2(power).real / (n * n)
    autocorr = np.fft.fftshift(autocorr)

    peak = autocorr.max()
    if peak <= 0:
        return float("nan")
    autocorr = autocorr / peak

    center = n // 2
    yy, xx = np.indices((n, n))
    r = np.sqrt((xx - center) ** 2 + (yy - center) ** 2) * spacing

    r_flat = r.ravel()
    c_flat = autocorr.ravel()

    r_max = L / 2.0
    n_bins = max(4, min(n // 2, 20))
    bin_edges = np.linspace(0.0, r_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    radial_c = np.full(n_bins, np.nan)
    for i in range(n_bins):
        m = (r_flat >= bin_edges[i]) & (r_flat < bin_edges[i + 1])
        if np.any(m):
            radial_c[i] = c_flat[m].mean()

    valid = ~np.isnan(radial_c) & (bin_centers > 0)
    if valid.sum() < 4:
        return float("nan")

    rc = bin_centers[valid][:n_fit_points]
    cc = np.clip(radial_c[valid][:n_fit_points], -1.0, 1.0)

    def model(r, xi):
        return np.exp(-r / xi)

    try:
        popt, _ = curve_fit(model, rc, cc, p0=[L / 8.0],
                             bounds=(1e-3, L), maxfev=2000)
        xi = float(popt[0])
    except Exception:
        return float("nan")

    if not np.isfinite(xi) or xi <= 0 or xi >= L:
        return float("nan")
    return xi


# ---------------------------------------------------------------------------
# Cross-field functionals
# ---------------------------------------------------------------------------

def field_covariance(u: np.ndarray, v: np.ndarray) -> float:
    """
    Population covariance (ddof=0), consistent with field_var's
    convention -- the grid is a complete population of cells for a given
    snapshot, not a sample drawn from a larger one.
    """
    return float(np.cov(u.ravel(), v.ravel(), ddof=0)[0, 1])


def field_mean_product(u: np.ndarray, v: np.ndarray) -> float:
    """<u * v> -- NOT the same as <u><v> unless u, v are uncorrelated."""
    return float(np.mean(u * v))


def gradient_alignment(u: np.ndarray, v: np.ndarray, spacing: float,
                        eps: float = 1e-12) -> float:
    """
    A_{u,v} = <grad u . grad v> / sqrt(<|grad u|^2> <|grad v|^2>),
    in [-1, 1]. Measures whether the two fields' spatial gradients tend to
    point the same way (near +1), opposite ways (near -1), or are
    spatially unrelated (near 0) -- the aggregated analogue of the
    pointwise coupling statistic from the earlier design discussion.
    """
    gux, guy = _gradient(u, spacing)
    gvx, gvy = _gradient(v, spacing)
    numerator = np.mean(gux * gvx + guy * gvy)
    denom = np.sqrt(np.mean(gux ** 2 + guy ** 2) * np.mean(gvx ** 2 + gvy ** 2))
    return float(numerator / (denom + eps))


# ---------------------------------------------------------------------------
# Convenience bundlers
# ---------------------------------------------------------------------------

def compute_field_features(u: np.ndarray, L: float, name: str) -> dict:
    """
    All single-field functionals for one named field, keyed like
    'phi_parallel_mean', 'phi_parallel_var', etc.
    """
    spacing = L / u.shape[0]
    return {
        f"{name}_mean": field_mean(u),
        f"{name}_var": field_var(u),
        f"{name}_skew": field_skew(u),
        f"{name}_q90": field_quantile(u, 0.9),
        f"{name}_grad2_mean": gradient_squared_mean(u, spacing),
        f"{name}_lap2_mean": laplacian_squared_mean(u, spacing),
        f"{name}_S_low_q": field_structure_factor_low_q(u, L),
        f"{name}_xi": correlation_length(u, L),
    }


def compute_cross_field_features(u: np.ndarray, v: np.ndarray, L: float,
                                  name_u: str, name_v: str) -> dict:
    """All cross-field functionals for a named pair of fields."""
    spacing = L / u.shape[0]
    key = f"{name_u}_{name_v}"
    return {
        f"cov_{key}": field_covariance(u, v),
        f"meanprod_{key}": field_mean_product(u, v),
        f"grad_align_{key}": gradient_alignment(u, v, spacing),
    }
