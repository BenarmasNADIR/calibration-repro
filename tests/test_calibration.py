import numpy as np
import pytest

from baselines import HistogramBinning, IsotonicCalibration
from calibration_metrics import ece, mce, nll, nll_from_logits, softmax
from temperature_scaling import apply_temperature, fit_temperature


def sample_calibrated(n=20000, k=10, scale=3.0, seed=0):
    """Logits z, labels drawn from softmax(z): calibrated by construction."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0, scale, size=(n, k))
    p = softmax(z)
    y = (p.cumsum(1) > rng.random((n, 1))).argmax(1)
    return z, y


def test_softmax_stable_and_normalized():
    z = np.array([[1000.0, 1001.0, 999.0], [-1000.0, -1000.0, -1000.0]])
    p = softmax(z)
    assert np.all(np.isfinite(p))
    np.testing.assert_allclose(p.sum(1), 1.0)
    np.testing.assert_allclose(p[1], 1 / 3)


def test_calibrated_data_has_low_ece():
    z, y = sample_calibrated()
    assert ece(softmax(z), y) < 0.02


def test_overconfident_data_has_high_ece():
    z, y = sample_calibrated()
    assert ece(softmax(3.0 * z), y) > 0.10


def test_mce_at_least_ece():
    for t in [0.5, 1.0, 3.0]:
        z, y = sample_calibrated(seed=1)
        p = softmax(t * z)
        assert mce(p, y) >= ece(p, y) - 1e-12


def test_nll_matches_manual():
    z = np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.0]])
    y = np.array([0, 2])
    manual = -np.mean([np.log(np.exp(2.0) / np.exp([2.0, 1.0, 0.1]).sum()),
                       np.log(np.exp(0.0) / np.exp([0.5, 2.5, 0.0]).sum())])
    assert nll(softmax(z), y) == pytest.approx(manual, rel=1e-10)
    assert nll_from_logits(z, y) == pytest.approx(manual, rel=1e-10)


def test_ece_manual_small_case():
    # Two samples in the same bin: confidences 0.9 and 0.9, one correct -> |0.5 - 0.9| = 0.4
    p = np.array([[0.9, 0.1], [0.9, 0.1]])
    y = np.array([0, 1])
    assert ece(p, y) == pytest.approx(0.4)
    assert mce(p, y) == pytest.approx(0.4)


def test_temperature_keeps_argmax():
    z, _ = sample_calibrated(n=2000)
    for t in [0.3, 1.0, 4.0]:
        np.testing.assert_array_equal(apply_temperature(z, t).argmax(1), z.argmax(1))


@pytest.mark.parametrize("true_t", [0.5, 2.0, 3.0])
def test_fitted_temperature_recovers_known_t(true_t):
    # Labels come from softmax(z); the model reports z * true_t, so the best T is true_t.
    z, y = sample_calibrated(n=30000, k=20, seed=2)
    t = fit_temperature(z * true_t, y)
    assert t == pytest.approx(true_t, rel=0.05)


def test_temperature_scaling_reduces_ece():
    z, y = sample_calibrated(seed=3)
    zv, yv, zt, yt = z[:10000], y[:10000], z[10000:], y[10000:]
    t = fit_temperature(3 * zv, yv)
    assert ece(apply_temperature(3 * zt, t), yt) < 0.25 * ece(softmax(3 * zt), yt)


@pytest.mark.parametrize("cls", [HistogramBinning, IsotonicCalibration])
def test_binning_baselines_reduce_ece(cls):
    z, y = sample_calibrated(seed=4)
    zv, yv, zt, yt = 3 * z[:10000], y[:10000], 3 * z[10000:], y[10000:]
    p = cls().fit(zv, yv).predict_proba(zt)
    np.testing.assert_allclose(p.sum(1), 1.0)
    assert ece(p, yt) < 0.5 * ece(softmax(zt), yt)
