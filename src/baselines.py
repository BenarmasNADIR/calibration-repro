"""Binning baselines from Guo et al. (2017), Section 4, extended to K classes one-vs-all.

For each class k a binary calibrator maps the softmax score p_k to a calibrated score q_k,
fit on validation data with targets 1[y == k]. The K scores are then renormalized to sum to 1
(Section 4.2, "Extension to multiclass models").
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression

from calibration_metrics import softmax


def _normalize(q):
    s = q.sum(axis=1, keepdims=True)
    k = q.shape[1]
    # If every class score is 0, fall back to the uniform distribution.
    return np.where(s > 0, q / np.where(s > 0, s, 1.0), 1.0 / k)


class HistogramBinning:
    """One-vs-all histogram binning with M equal-width bins on [0, 1].

    Each bin predicts the fraction of positives among validation samples falling in it.
    Empty bins predict the bin centre (assumption: the paper does not specify this case).
    """

    def __init__(self, n_bins=15):
        self.n_bins = n_bins

    def _bin_index(self, p):
        return np.clip(np.ceil(p * self.n_bins).astype(int) - 1, 0, self.n_bins - 1)

    def fit(self, val_logits, val_labels):
        p = softmax(val_logits)
        labels = np.asarray(val_labels)
        k = p.shape[1]
        centres = (np.arange(self.n_bins) + 0.5) / self.n_bins
        self.values_ = np.empty((k, self.n_bins))
        for c in range(k):
            idx = self._bin_index(p[:, c])
            cnt = np.bincount(idx, minlength=self.n_bins)
            pos = np.bincount(idx, weights=(labels == c).astype(float), minlength=self.n_bins)
            self.values_[c] = np.where(cnt > 0, pos / np.maximum(cnt, 1), centres)
        return self

    def predict_proba(self, logits):
        p = softmax(logits)
        idx = self._bin_index(p)
        q = self.values_[np.arange(p.shape[1])[None, :], idx]
        return _normalize(q)


class IsotonicCalibration:
    """One-vs-all isotonic regression (scikit-learn, clipped to [0, 1])."""

    def fit(self, val_logits, val_labels):
        p = softmax(val_logits)
        labels = np.asarray(val_labels)
        self.models_ = []
        for c in range(p.shape[1]):
            m = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            m.fit(p[:, c], (labels == c).astype(float))
            self.models_.append(m)
        return self

    def predict_proba(self, logits):
        p = softmax(logits)
        q = np.stack([m.predict(p[:, c]) for c, m in enumerate(self.models_)], axis=1)
        return _normalize(q)
