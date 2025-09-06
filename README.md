# Autocorrelation Penalty Module (Paper 2)

*A per‑asset update that penalizes recent serial correlation in the signal.*

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

## Overview

This repository provides a production‑ready implementation of an **autocorrelation penalty** for portfolio updates. The method computes a Ljung–Box p‑value on a recent window of the cumulative‑return signal and converts it to a shrinkage term that reduces exposure when strong autocorrelation is detected.

**Penalty**: $\text{Penalty} = \lambda\,(1 - p_{\text{LB}})$
**Update**: project the shrunk weights back to the simplex so they remain non‑negative and sum to one.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q
python examples.py
```

## Usage

```python
import numpy as np
from autocorr_module import (
    ljung_box_pvalue, autocorr_penalty, autocorr_adjusted_update
)

# mock cumulative returns (AR(1) cum sum)
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
```

**Key properties**

* Penalizes **low p‑values** (strong autocorrelation) more than white‑noise‑like signals.
* Always returns non‑negative weights that **sum to 1**.
* Window length, lags, and penalty scale $\lambda$ are configurable.

## Files included

* `autocorr_module.py` — Ljung–Box p‑value (lightweight), penalty, simplex projection, update step.
* `tests/test_autocorr_module.py` — invariants and AR(1) vs. shuffled behavior.
* `tests/test_artifacts_exist.py` — checks for `Autocorrelation_Penalty_Module.pdf` and `Tests.zip`.
* `.github/workflows/ci.yml` — pytest on push/PR.
* `.github/workflows/release.yml` — GitHub Release on tags (for Zenodo integration).
* `CITATION.cff` — includes your ORCID.
* `.zenodo.json` — pre‑fills Zenodo deposition metadata.
* `requirements.txt`, `examples.py`, `CHANGELOG.md`, `LICENSE-CODE`, `LICENSE-DOCS`, `.gitignore`.

## ORCID & Zenodo

* Your ORCID iD: **[https://orcid.org/0009-0004-9601-5617](https://orcid.org/0009-0004-9601-5617)**.
* With GitHub↔Zenodo connected, pushing a git tag (e.g., `v0.1.0`) creates a Zenodo record and mints a DOI.

**Publish checklist**

1. Commit code, paper, and tests.
2. Update versions in `CHANGELOG.md` and `CITATION.cff`.
3. Create a tag: `git tag v0.1.0 && git push --tags`.
4. When the DOI appears on Zenodo, replace the badge DOI above and add it to `CITATION.cff -> identifiers`.
5. Check your ORCID Works; add the DOI manually if it didn’t auto‑sync.

## Citing

Use `CITATION.cff` or the BibTeX below (replace DOI after the first release).

```bibtex
@misc{autocorrpen2025,
  title        = {Autocorrelation Penalty Module},
  author       = {Villalobos-Gonzalez, Santiago de Jesus},
  year         = {2025},
  note         = {Code and preprint. DOI to be added after first Zenodo release.},
  howpublished = {GitHub + Zenodo}
}
```

## License

* **Code**: MIT (see `LICENSE-CODE`).
* **Text/figures/PDFs**: CC BY 4.0 (see `LICENSE-DOCS`).