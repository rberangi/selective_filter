# Selective filters

Reference implementation and closed-form theory for **selective filters**, a family of iterative smoothers, and for their **Odd-Even** member: a linear, zero-phase, multiplier-free smoother whose frequency response is never negative.

This repository accompanies the paper *Partition-Averaged Selective Filters: Exact Reduction to Linear Smoothers and Multiplier-Free Gaussian Design* (R. Berangi, submitted).

## The idea in three lines

A **selective step** replaces a chosen set of samples by the mean of their neighbours at prescribed lags and leaves the other samples unchanged. Choosing the local maxima, the local minima, the odd indices or the even indices gives different members of the family.

A single step is nonlinear or shift-variant. **Averaging the steps over a partition of the samples cancels the selection exactly**:

    (1/K) Σ_k F_{T_k} y = (A y + (K − 1) y) / K

where `A` averages each sample's neighbours. For the odd and even samples (K = 2) one step is

    y[n] ← y[n]/2 + (y[n−k] + y[n+k])/4

which needs two additions and a shift, with response `cos²(kω/2)` in `[0, 1]`. Repeats and lag sweeps therefore collapse into a single FIR kernel with closed-form impulse response, transfer function, moments and power spectral density, and the filter converges to Gaussian smoothing.

## Install

```bash
pip install -e .              # the library (needs only NumPy)
pip install -e ".[examples]"  # plus SciPy, Matplotlib and Jupyter for the examples
pip install -e ".[test]"      # plus pytest
```

## Quick start

```python
import numpy as np
from selective_filters import apply_selective_filter, odd_even_impulse_response, odd_even_moments

rng = np.random.default_rng(0)
t = np.arange(2000)
y = np.sin(2 * np.pi * t / 200) + 0.3 * rng.standard_normal(t.size)

# Odd-Even filter: lag sweep 5, 4, ..., 1, one repeat
smooth = apply_selective_filter(y, "Odd-Even", lags=[1, 5], repeats=1, sweep_lags="descending", include_borders=True)

# The same filter as a single FIR kernel, from the closed form
n, h = odd_even_impulse_response([1, 5], 1, "descending")
print(odd_even_moments([1, 5], 1, "descending")["sigma"])   # equivalent Gaussian width, 5.24 samples
```

## Contents

| Path | What it is |
|---|---|
| `src/selective_filters/core.py` | Selective filter, Odd-Even closed forms, bandpass filters, streaming (overlap-save) |
| `examples/demo.ipynb` | Walk-through: smoothing, exact reduction, closed forms, partitions, design, bandpass, streaming |
| `examples/quickstart.py` | Minimal script |
| `scripts/make_paper_figures.py` | Regenerates the figures of the paper |
| `tests/` | Checks every closed form against the algorithm |

### Main functions

| Function | Purpose |
|---|---|
| `selective_filter(y, target, lags, include_borders)` | One selective step. Targets: `"Max"`, `"Min"`, `"All"`, `"Odd"`, `"Even"` |
| `apply_selective_filter(y, target, lags, repeats, sweep_lags, include_borders)` | Repeats and lag sweeps; also the composite targets `"Odd-Even"` and `"Max-Min"` |
| `odd_even_impulse_response`, `odd_even_frequency_response`, `odd_even_psd`, `odd_even_moments` | Closed forms of the Odd-Even filter |
| `apply_odd_even_bandpass` and its `..._frequency_response`, `..._impulse_response`, `..._properties` | Multiplier-free bandpass filters |
| `apply_selective_filter_stream`, `selective_filter_radius` | Exact block-by-block filtering of continuous signals |

Lags are given as a window `[n1, n2]`: the neighbours of sample `n` are `n ± n1 … n ± n2`, and the centre is included when `n1 = 0`. With `sweep_lags="ascending"` or `"descending"` successive steps use the single lags `k = n1 … n2` (`k ≥ 1`). The closed forms hold away from the signal borders, which are handled by reflection.

## Reproducing the results

```bash
pytest                                # closed forms against the algorithm
python scripts/make_paper_figures.py  # writes figures/ (needs the [examples] extras)
jupyter lab examples/demo.ipynb
```

## Citation

If you use this code, please cite the paper (details will be added on publication) and this repository; see `CITATION.cff`.

## License

MIT, see `LICENSE`.
