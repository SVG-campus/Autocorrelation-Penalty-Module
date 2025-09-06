import numpy as np
from autocorr_module import ljung_box_pvalue, autocorr_penalty, autocorr_adjusted_update

def _ar1_series(T=800, phi=0.7, sigma=0.01, seed=0):
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, sigma, T)
    x = np.empty(T)
    x[0] = eps[0]
    for t in range(1, T):
        x[t] = phi * x[t-1] + eps[t]
    return np.cumsum(x)

def test_penalty_behaviour_ar1_vs_shuffled():
    cum = _ar1_series(T=1200, phi=0.7, sigma=0.01, seed=1)
    p_ar = ljung_box_pvalue(cum, lags=5, window=300)
    pen_ar = autocorr_penalty(cum, penalty_scale=0.05, lb_lags=5, window=300)

    rng = np.random.default_rng(1)
    cum_shuffled = np.cumsum(rng.permutation(np.diff(np.insert(cum, 0, 0.0))))
    p_sh = ljung_box_pvalue(cum_shuffled, lags=5, window=300)
    pen_sh = autocorr_penalty(cum_shuffled, penalty_scale=0.05, lb_lags=5, window=300)

    assert p_ar < p_sh or pen_ar > pen_sh

def test_update_invariants():
    cum = _ar1_series(T=1000, phi=0.6, sigma=0.01, seed=2)
    w0 = np.array([0.4, 0.3, 0.3])
    w1 = autocorr_adjusted_update(w0, cum, penalty_scale=0.05, lb_lags=5, window=200)
    assert np.all(w1 >= 0)
    assert np.isclose(w1.sum(), 1.0)

def test_short_series_zero_penalty():
    cum = np.array([0.0, 0.01, 0.02])
    w0 = np.array([0.5, 0.5])
    w1 = autocorr_adjusted_update(w0, cum, penalty_scale=0.5, lb_lags=5, window=200)
    assert np.allclose(w1, w0 / w0.sum())
