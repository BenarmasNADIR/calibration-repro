"""Calibration metrics from Guo et al. (2017), Section 2.

All functions take numpy arrays. `probs` is (n, K) and rows sum to 1; `labels` is (n,) int.
ECE and MCE use M equal-width confidence bins on [0, 1] (M = 15 in the paper).
"""
import numpy as np


def softmax(logits, axis=-1):
    """Numerically stable softmax (subtracts the row max before exponentiating)."""
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def log_softmax(logits, axis=-1):
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=axis, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=axis, keepdims=True))


def _bin_stats(probs, labels, n_bins=15):
    """Per-bin (count, accuracy, mean confidence) for equal-width bins.

    Bin m covers ((m-1)/M, m/M]; confidence 0 goes into the first bin.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(np.float64)
    idx = np.clip(np.ceil(conf * n_bins).astype(int) - 1, 0, n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins).astype(np.float64)
    acc_sum = np.bincount(idx, weights=correct, minlength=n_bins)
    conf_sum = np.bincount(idx, weights=conf, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        acc = np.where(counts > 0, acc_sum / counts, 0.0)
        mean_conf = np.where(counts > 0, conf_sum / counts, 0.0)
    return counts, acc, mean_conf


def reliability_bins(probs, labels, n_bins=15):
    """Public wrapper used by the reliability diagram."""
    return _bin_stats(probs, labels, n_bins)


def ece(probs, labels, n_bins=15):
    """Expected Calibration Error: sum_m |B_m|/n * |acc(B_m) - conf(B_m)|."""
    counts, acc, conf = _bin_stats(probs, labels, n_bins)
    return float(np.sum(counts / counts.sum() * np.abs(acc - conf)))


def mce(probs, labels, n_bins=15):
    """Maximum Calibration Error over non-empty bins."""
    counts, acc, conf = _bin_stats(probs, labels, n_bins)
    gaps = np.abs(acc - conf)[counts > 0]
    return float(gaps.max()) if gaps.size else 0.0


def nll(probs, labels, eps=1e-12):
    """Mean negative log-likelihood of the true class. Probabilities are clipped at eps."""
    probs = np.asarray(probs, dtype=np.float64)
    p_true = probs[np.arange(len(labels)), np.asarray(labels)]
    return float(-np.mean(np.log(np.clip(p_true, eps, 1.0))))


def nll_from_logits(logits, labels):
    """NLL computed directly from logits (exact, no clipping)."""
    lsm = log_softmax(logits)
    return float(-np.mean(lsm[np.arange(len(labels)), np.asarray(labels)]))


def error(probs, labels):
    return float(np.mean(np.asarray(probs).argmax(axis=1) != np.asarray(labels)))


def all_metrics(probs, labels, n_bins=15):
    return {
        "error": error(probs, labels),
        "nll": nll(probs, labels),
        "ece": ece(probs, labels, n_bins),
        "mce": mce(probs, labels, n_bins),
    }
