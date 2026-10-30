"""Checks of the closed forms against the selective algorithm (the claims of the paper)."""
import numpy as np
import pytest

from selective_filters import (
    apply_odd_even_bandpass,
    apply_selective_filter,
    apply_selective_filter_stream,
    odd_even_bandpass_impulse_response,
    odd_even_frequency_response,
    odd_even_impulse_response,
    odd_even_moments,
    selective_filter,
)

DESIGNS = [(lags, R, sweep)
           for lags in ([0, 1], [1, 1], [1, 3], [2, 4], [0, 3], [1, 5])
           for R in (1, 2, 3)
           for sweep in ("none", "ascending", "descending")]


def impulse(n=801):
    x = np.zeros(n)
    x[n // 2] = 1.0
    return x


def test_odd_even_step_is_three_tap_kernel():
    y = np.random.default_rng(0).standard_normal(200)
    for lag in (1, 2, 5):
        out = apply_selective_filter(y, "Odd-Even", [lag, lag], 1, "none", True)
        n = np.arange(lag, len(y) - lag)
        np.testing.assert_allclose(out[n], y[n] / 2 + (y[n - lag] + y[n + lag]) / 4, atol=1e-15)


@pytest.mark.parametrize("lags,R,sweep", DESIGNS)
def test_impulse_response_matches_closed_form(lags, R, sweep):
    x = impulse()
    n, h = odd_even_impulse_response(lags, R, sweep)
    out = apply_selective_filter(x, "Odd-Even", lags, R, sweep, True)
    np.testing.assert_allclose(out[len(x) // 2 + n], h, atol=1e-15)
    assert abs(out.sum() - 1) < 1e-12                       # nothing outside the predicted support
    assert np.all(h >= 0) and abs(h.sum() - 1) < 1e-12


@pytest.mark.parametrize("lags,R,sweep", DESIGNS)
def test_frequency_response_matches_and_lies_in_unit_interval(lags, R, sweep):
    n, h = odd_even_impulse_response(lags, R, sweep)
    w = np.linspace(0, np.pi, 257)
    H_fft = (h[None, :] * np.cos(np.outer(w, n))).sum(axis=1)       # zero-phase: real response
    H = odd_even_frequency_response(w, lags, R, sweep)
    np.testing.assert_allclose(H, H_fft, atol=1e-13)
    assert H.min() >= -1e-15 and H.max() <= 1 + 1e-15


def test_sweep_order_does_not_matter():
    y = np.random.default_rng(1).standard_normal(500)
    a = apply_selective_filter(y, "Odd-Even", [1, 6], 2, "ascending", True)
    d = apply_selective_filter(y, "Odd-Even", [1, 6], 2, "descending", True)
    np.testing.assert_allclose(a[50:-50], d[50:-50], atol=1e-14)


@pytest.mark.parametrize("lags,R,sweep", DESIGNS)
def test_variance_matches_kernel(lags, R, sweep):
    n, h = odd_even_impulse_response(lags, R, sweep)
    assert odd_even_moments(lags, R, sweep)["variance"] == pytest.approx(np.sum(n**2 * h), rel=1e-12)


def test_gaussian_limit():
    n, h = odd_even_impulse_response([0, 3], 20, "none")
    var = odd_even_moments([0, 3], 20, "none")["variance"]
    g = np.exp(-n**2 / (2 * var))
    assert var == pytest.approx(40.0)
    assert np.abs(h - g / g.sum()).sum() < 0.01


def test_data_dependent_partition_is_linear():
    """Averaging the Max, Min and remaining-sample steps gives (A + 2I)/3, although each step is nonlinear."""
    y = np.random.default_rng(2).standard_normal(300)
    lags = [1, 3]
    Ay = selective_filter(y, "All", lags, True)                      # every sample replaced: A y
    yb = np.pad(y, 1, mode="reflect")
    is_max = (y >= yb[:-2]) & (y >= yb[2:])
    is_min = (y <= yb[:-2]) & (y <= yb[2:])
    rest = ~(is_max | is_min)
    assert not np.any(is_max & is_min)                               # a partition for tie-free data
    steps = [np.where(mask, Ay, y) for mask in (is_max, is_min, rest)]
    np.testing.assert_allclose(np.mean(steps, axis=0), (Ay + 2 * y) / 3, atol=1e-15)
    np.testing.assert_allclose(steps[0], selective_filter(y, "Max", lags, True), atol=1e-15)
    np.testing.assert_allclose(steps[1], selective_filter(y, "Min", lags, True), atol=1e-15)


def test_max_min_is_not_linear():
    rng = np.random.default_rng(3)
    a, b = rng.standard_normal(200), rng.standard_normal(200)
    f = lambda v: apply_selective_filter(v, "Max-Min", [1, 3], 1, "none", True)
    assert np.abs(f(a + b) - f(a) - f(b)).max() > 1e-3


@pytest.mark.parametrize("include_borders", [True, False])
def test_odd_and_even_target_their_own_parity(include_borders):
    z = np.arange(12.0) ** 2 % 7
    odd = np.nonzero(selective_filter(z, "Odd", [1, 1], include_borders) != z)[0]
    even = np.nonzero(selective_filter(z, "Even", [1, 1], include_borders) != z)[0]
    assert np.all(odd % 2 == 1) and np.all(even % 2 == 0)


def test_single_targets_through_wrapper():
    y = np.random.default_rng(4).standard_normal(100)
    for t in ("Odd", "Even", "Max", "Min", "All"):
        apply_selective_filter(y, t, [1, 3], 2, "descending", True)


@pytest.mark.parametrize("target", ["Odd-Even", "Max-Min", "Max"])
def test_streaming_equals_batch(target):
    rng = np.random.default_rng(5)
    y = rng.standard_normal(3000)
    cuts = np.sort(rng.choice(np.arange(1, len(y)), 25, replace=False))
    blocks = np.split(y, cuts)
    stream = np.concatenate(list(apply_selective_filter_stream(blocks, target, [1, 4], 2, "descending", True)))
    np.testing.assert_allclose(stream, apply_selective_filter(y, target, [1, 4], 2, "descending", True), atol=1e-12)


@pytest.mark.parametrize("kw", [dict(method="product", r=2, s=2),
                                dict(method="product", r=3, s=1, dilation=3, image_lags=(1, 2)),
                                dict(method="difference", lags_a=(1, 2), lags_b=(1, 6), repeats=2)])
def test_bandpass_impulse_response(kw):
    x = impulse()
    n, h = odd_even_bandpass_impulse_response(**kw)
    out = apply_odd_even_bandpass(x, **kw)
    np.testing.assert_allclose(out[len(x) // 2 + n], h, atol=1e-14)
