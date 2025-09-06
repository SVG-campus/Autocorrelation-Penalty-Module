import numpy as np
from autocorr_module import ljung_box_pvalue, autocorr_penalty, autocorr_adjusted_update

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T = 1000
    eps = rng.normal(0, 0.01, T)
    x = np.empty(T)
    phi = 0.6
    x[0] = eps[0]
    for t in range(1, T):
        x[t] = phi * x[t-1] + eps[t]
    cum_returns = np.cumsum(x)

    pval = ljung_box_pvalue(cum_returns, lags=5, window=200)
    pen = autocorr_penalty(cum_returns, penalty_scale=0.05, lb_lags=5, window=200)
    w0 = np.array([0.4, 0.3, 0.3])
    w1 = autocorr_adjusted_update(w0, cum_returns, penalty_scale=0.05, lb_lags=5, window=200)
    print("pval:", pval, "penalty:", pen, "w1:", w1, "sum:", w1.sum())
