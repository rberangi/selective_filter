"""Smooth a noisy signal with the Odd-Even filter and compare it with its closed-form FIR kernel."""
import numpy as np

from selective_filters import apply_selective_filter, odd_even_impulse_response, odd_even_moments

rng = np.random.default_rng(0)
t = np.arange(2000)
clean = np.sin(2 * np.pi * t / 200)
y = clean + 0.3 * rng.standard_normal(t.size)

lags, repeats, sweep = [1, 5], 1, "descending"
smooth = apply_selective_filter(y, "Odd-Even", lags, repeats, sweep, include_borders=True)

# The selective iteration is exactly one FIR kernel: convolving with it gives the same output.
n, h = odd_even_impulse_response(lags, repeats, sweep)
L = n[-1]
fir = np.convolve(y, h, mode="same")
print(f"kernel: {len(h)} taps, all >= 0, sum = {h.sum():.12f}")
print(f"max |selective - FIR| away from borders: {np.abs(smooth - fir)[L:-L].max():.1e}")

m = odd_even_moments(lags, repeats, sweep)
print(f"equivalent Gaussian sigma: {m['sigma']:.2f} samples, white-noise gain: {m['noise_gain']:.4f}")
print(f"RMS error against the clean signal: noisy {np.std(y - clean):.3f}, smoothed {np.std((smooth - clean)[L:-L]):.3f}")
