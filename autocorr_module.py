from __future__ import annotations
import numpy as np
from math import isfinite
from scipy.stats import chi2

_EPS = 1e-12

def _autocorr(x: np.ndarray, lag: int) -> float:
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = x.size
    if lag >= n:
        return 0.0
    num = np.dot(x[:n-lag], x[lag:])
    den = np.dot(x, x)
    return float(num / den) if den > 0 else 0.0

def ljung_box_pvalue(series: np.ndarray, lags: int = 5, window: int = 200) -> float:
    x = np.asarray(series, dtype=float)
    if x.ndim != 1:
        x = x.ravel()
    n = min(x.size, int(window))
    if n < max(10, lags + 5):
        return 1.0
    y = x[-n:]
    Q = 0.0
    for k in range(1, lags + 1):
        rho_k = _autocorr(y, k)
        Q += rho_k * rho_k / (n - k)
    Q *= n * (n + 2)
    p = float(chi2.sf(Q, df=lags))
    if not isfinite(p):
        return 1.0
    return max(0.0, min(1.0, p))

def projected_simplex(v: np.ndarray, s: float = 1.0) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    if s <= 0:
        raise ValueError("s must be > 0")
    n = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - s))[0][-1]
    theta = (cssv[rho] - s) / (rho + 1.0)
    w = np.maximum(v - theta, 0.0)
    sw = w.sum()
    if sw <= 0 or not np.isfinite(sw):
        raise ValueError("projection failed; input may be too pathological")
    return w / sw

def autocorr_penalty(cum_returns: np.ndarray, penalty_scale: float = 0.05, lb_lags: int = 5, window: int = 200, min_n: int = 30) -> float:
    x = np.asarray(cum_returns, dtype=float)
    if x.size < int(min_n):
        return 0.0
    p = ljung_box_pvalue(x, lags=lb_lags, window=window)
    pen = float(penalty_scale * (1.0 - p))
    return max(0.0, min(1.0, pen))

def autocorr_adjusted_update(weights: np.ndarray, cum_returns: np.ndarray, penalty_scale: float = 0.05, lb_lags: int = 5, window: int = 200, clip_min: float = 1e-12) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    s = w.sum()
    if s <= 0 or not np.isfinite(s):
        raise ValueError("weights must have positive, finite sum")
    w = w / s
    pen = autocorr_penalty(cum_returns, penalty_scale=penalty_scale, lb_lags=lb_lags, window=window)
    new_w = np.maximum(w * (1.0 - pen), clip_min)
    return projected_simplex(new_w, s=1.0)
