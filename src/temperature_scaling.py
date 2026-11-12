"""Temperature scaling (Guo et al. 2017, Section 4.2).

A single scalar T > 0 divides the logits. T is fit on validation logits by minimizing NLL
with L-BFGS over log T (so T stays positive). The argmax, hence accuracy, is unchanged.
"""
import numpy as np
from scipy.optimize import minimize

from calibration_metrics import log_softmax, softmax


def _nll_and_grad(log_t, logits, labels):
    t = np.exp(log_t[0])
    z = logits / t
    lsm = log_softmax(z)
    n = len(labels)
    loss = -lsm[np.arange(n), labels].mean()
    # d loss / d z = softmax(z) - onehot; dz/dlogT = -z
    p = np.exp(lsm)
    g_z = p
    g_z[np.arange(n), labels] -= 1.0
    grad = np.sum(g_z * (-z)) / n
    return loss, np.array([grad])


def fit_temperature(val_logits, val_labels, init_t=1.5):
    """Return the temperature T minimizing validation NLL."""
    logits = np.asarray(val_logits, dtype=np.float64)
    labels = np.asarray(val_labels, dtype=np.int64)
    res = minimize(_nll_and_grad, x0=np.array([np.log(init_t)]), args=(logits, labels),
                   jac=True, method="L-BFGS-B", options={"maxiter": 500, "gtol": 1e-10})
    return float(np.exp(res.x[0]))


def apply_temperature(logits, t):
    return softmax(np.asarray(logits, dtype=np.float64) / t)


class TemperatureScaler:
    def fit(self, val_logits, val_labels):
        self.t_ = fit_temperature(val_logits, val_labels)
        return self

    def predict_proba(self, logits):
        return apply_temperature(logits, self.t_)
