import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsvp_mips.fields import Grid
from rsvp_mips.features import (
    field_mean,
    field_var,
    field_skew,
    field_quantile,
    bimodality_coefficient,
    gradient_squared_mean,
    laplacian_squared_mean,
    field_structure_factor_low_q,
    correlation_length,
    field_covariance,
    field_mean_product,
    gradient_alignment,
    compute_field_features,
    compute_cross_field_features,
)


def make_sinusoid_field(L: float, n: int, A: float = 2.0, k_index: int = 1) -> np.ndarray:
    """u(x,y) = A * sin(k*x), k = k_index * 2*pi/L, independent of y."""
    grid = Grid(L=L, n=n)
    k = k_index * 2.0 * np.pi / L
    x = grid.points[:, 0]
    return (A * np.sin(k * x)).reshape(n, n)


# ---------------------------------------------------------------------------
# Basic statistical functionals
# ---------------------------------------------------------------------------

def test_field_mean_var_on_known_array():
    u = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert field_mean(u) == 2.5
    assert np.isclose(field_var(u), np.var([1.0, 2.0, 3.0, 4.0]))


def test_field_skew_zero_for_symmetric_field():
    rng = np.random.default_rng(0)
    u = rng.standard_normal((64, 64))
    assert abs(field_skew(u)) < 0.1  # should be near zero for large symmetric sample


def test_field_quantile():
    u = np.arange(100).reshape(10, 10).astype(float)
    q90 = field_quantile(u, 0.9)
    assert 85 <= q90 <= 95


def test_bimodality_coefficient_higher_for_bimodal_sample():
    rng = np.random.default_rng(1)
    bimodal = np.concatenate([
        rng.normal(-3, 0.5, 2500), rng.normal(3, 0.5, 2500)
    ]).reshape(50, 100)
    unimodal = rng.normal(0, 1, 5000).reshape(50, 100)

    bc_bimodal = bimodality_coefficient(bimodal)
    bc_unimodal = bimodality_coefficient(unimodal)
    assert bc_bimodal > bc_unimodal
    assert bc_bimodal > 5.0 / 9.0


# ---------------------------------------------------------------------------
# Spatial derivative functionals -- exact algebraic identities for a
# discrete sinusoid under periodic central differences (not an
# approximation: central-difference of a pure sinusoid has a closed form)
# ---------------------------------------------------------------------------

def test_gradient_squared_mean_matches_exact_finite_difference_identity():
    L, n, A = 20.0, 64, 2.0
    u = make_sinusoid_field(L, n, A=A, k_index=1)
    spacing = L / n
    k = 2.0 * np.pi / L

    # exact discrete identity: central diff of A*sin(kx) has amplitude
    # A * sin(k*h) / h, and <cos^2> = 0.5 exactly for uniform sampling of
    # a full period
    expected = (A * np.sin(k * spacing) / spacing) ** 2 * 0.5

    measured = gradient_squared_mean(u, spacing)
    assert np.isclose(measured, expected, rtol=1e-10)


def test_laplacian_squared_mean_matches_exact_finite_difference_identity():
    L, n, A = 20.0, 64, 2.0
    u = make_sinusoid_field(L, n, A=A, k_index=1)
    spacing = L / n
    k = 2.0 * np.pi / L

    # exact discrete identity: 5-point laplacian of A*sin(kx) (no
    # y-dependence) has amplitude 2*A*(cos(k*h)-1)/h^2
    amplitude = 2.0 * A * (np.cos(k * spacing) - 1.0) / spacing ** 2
    expected = amplitude ** 2 * 0.5

    measured = laplacian_squared_mean(u, spacing)
    assert np.isclose(measured, expected, rtol=1e-10)


def test_gradient_and_laplacian_zero_for_constant_field():
    u = np.full((32, 32), 5.0)
    assert gradient_squared_mean(u, spacing=0.5) == 0.0
    assert laplacian_squared_mean(u, spacing=0.5) == 0.0


# ---------------------------------------------------------------------------
# Fourier-space functionals
# ---------------------------------------------------------------------------

def test_structure_factor_low_q_higher_for_low_frequency_field():
    L, n = 20.0, 64
    low_freq = make_sinusoid_field(L, n, A=2.0, k_index=1)

    rng = np.random.default_rng(2)
    noise = rng.standard_normal((n, n))
    noise *= low_freq.std() / noise.std()  # match overall variance

    S_low_freq = field_structure_factor_low_q(low_freq, L)
    S_noise = field_structure_factor_low_q(noise, L)
    assert S_low_freq > S_noise


def test_correlation_length_larger_for_smoothed_field_than_white_noise():
    from scipy.ndimage import gaussian_filter

    L, n = 20.0, 64
    rng = np.random.default_rng(3)
    white_noise = rng.standard_normal((n, n))
    smoothed = gaussian_filter(white_noise, sigma=4.0, mode="wrap")

    xi_white = correlation_length(white_noise, L)
    xi_smoothed = correlation_length(smoothed, L)

    # white noise should fail to fit a meaningful correlation length (NaN)
    # or at minimum be much smaller than the smoothed field's
    assert np.isnan(xi_white) or xi_smoothed > xi_white


def test_correlation_length_nan_for_degenerate_field():
    u = np.zeros((32, 32))
    assert np.isnan(correlation_length(u, L=20.0))


# ---------------------------------------------------------------------------
# Cross-field functionals
# ---------------------------------------------------------------------------

def test_covariance_and_alignment_for_identical_fields():
    L, n = 20.0, 64
    u = make_sinusoid_field(L, n, A=2.0, k_index=1)
    spacing = L / n

    assert np.isclose(field_covariance(u, u), field_var(u))
    assert np.isclose(gradient_alignment(u, u, spacing), 1.0, atol=1e-8)


def test_alignment_negative_for_opposed_fields():
    L, n = 20.0, 64
    u = make_sinusoid_field(L, n, A=2.0, k_index=1)
    v = -u
    spacing = L / n

    assert np.isclose(gradient_alignment(u, v, spacing), -1.0, atol=1e-8)
    assert np.isclose(field_covariance(u, v), -field_var(u))


def test_mean_product_differs_from_product_of_means_when_correlated():
    L, n = 20.0, 64
    u = make_sinusoid_field(L, n, A=2.0, k_index=1)
    # u correlated with itself: <u*u> = var(u) + mean(u)^2, mean(u)~0 here
    assert not np.isclose(field_mean_product(u, u), field_mean(u) * field_mean(u))


# ---------------------------------------------------------------------------
# Bundlers
# ---------------------------------------------------------------------------

def test_compute_field_features_returns_expected_keys():
    L, n = 20.0, 32
    u = make_sinusoid_field(L, n)
    feats = compute_field_features(u, L, "phi_parallel")
    expected_keys = {
        "phi_parallel_mean", "phi_parallel_var", "phi_parallel_skew",
        "phi_parallel_q90", "phi_parallel_grad2_mean",
        "phi_parallel_lap2_mean", "phi_parallel_S_low_q", "phi_parallel_xi",
    }
    assert set(feats.keys()) == expected_keys
    assert all(isinstance(v, float) for v in feats.values())


def test_compute_cross_field_features_returns_expected_keys():
    L, n = 20.0, 32
    u = make_sinusoid_field(L, n, k_index=1)
    v = make_sinusoid_field(L, n, k_index=2)
    feats = compute_cross_field_features(u, v, L, "rho", "phi_parallel")
    expected_keys = {
        "cov_rho_phi_parallel", "meanprod_rho_phi_parallel",
        "grad_align_rho_phi_parallel",
    }
    assert set(feats.keys()) == expected_keys
