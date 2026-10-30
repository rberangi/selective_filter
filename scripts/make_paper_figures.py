"""Figures for the selective-filter paper.

Figure 1: neighbourhood of a selective step (lag windows, sweep steps, Odd-Even targets).
Figure 2: output power spectral density of Odd-Even lowpass designs, theory against simulation.
Figure 3: impulse response, transfer function and output PSD, closed forms against the selective algorithm.

Run from the repository root: python scripts/make_paper_figures.py (writes figures/).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from selective_filters import (apply_selective_filter, odd_even_frequency_response, odd_even_impulse_response,
                              odd_even_moments, odd_even_psd)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, AQUA, GREY, LIGHT, INK = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a85", "#d9d8d2", "#1a1a19"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GREY, "axes.linewidth": 0.6,
    "xtick.color": "#3d3d3a", "ytick.color": "#3d3d3a", "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.grid": True, "grid.color": "#ecebe6", "grid.linewidth": 0.6, "legend.frameon": False,
})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 1: neighbourhood
def row(ax, y, roles, label, span=7):
    """Draw samples n-span..n+span; roles maps offset -> 'target' | 'neigh' | 'excl' | 'kept'."""
    for d in range(-span, span + 1):
        role = roles.get(d, "other")
        style = {
            "target": dict(s=70, c=ORANGE, edgecolors=ORANGE, zorder=3),
            "neigh": dict(s=55, c=BLUE, edgecolors=BLUE, zorder=3),
            "excl": dict(s=55, c="white", edgecolors=GREY, linewidths=0.9, zorder=3),
            "kept": dict(s=30, c=LIGHT, edgecolors=GREY, linewidths=0.5, zorder=2),
            "other": dict(s=30, c=LIGHT, edgecolors=LIGHT, zorder=2),
        }[role]
        ax.scatter([d], [y], marker="o", **style)
    ax.text(-span - 0.9, y, label, ha="right", va="center", fontsize=8.5, color=INK)


fig, ax = plt.subplots(figsize=(6.3, 3.3))
ax.set_axis_off()
span = 7
rows = []
# (a) window [2, 4]: centre and inner neighbours excluded
roles = {0: "target", **{d: "neigh" for d in (-4, -3, -2, 2, 3, 4)}, **{d: "excl" for d in (-1, 1)}}
rows.append((roles, "(a)  window $[n_1,n_2]=[2,4]$"))
# (b) window [0, 3]: centre included
roles = {**{d: "neigh" for d in range(-3, 4)}}
roles[0] = "target"
rows.append((roles, "(b)  window $[0,3]$, centre included"))
# (c) sweep steps [k, k]
for k in (1, 2, 3):
    rows.append(({0: "target", -k: "neigh", k: "neigh"}, f"(c)  sweep step $k={k}$" if k == 1 else f"sweep step $k={k}$"))
# (d) Odd-Even: odd and even targets of the same input
rows.append(({d: ("target" if d % 2 else "kept") for d in range(-span, span + 1)}, "(d)  Odd step: odd targets"))
rows.append(({d: ("target" if d % 2 == 0 else "kept") for d in range(-span, span + 1)}, "Even step: even targets"))

for i, (roles, label) in enumerate(rows):
    row(ax, -i, roles, label, span)
# mark the centre column and the index axis
ax.axvline(0, color=ORANGE, lw=0.6, alpha=0.35, zorder=1)
for d in range(-span, span + 1):
    lab = "$n$" if d == 0 else (f"$n{d:+d}$" if abs(d) in (2, 4, 6) else "")
    ax.text(d, 0.75, lab, ha="center", va="bottom", fontsize=7.5, color="#3d3d3a")
# legend
handles = [
    plt.Line2D([], [], ls="", marker="o", ms=7, mfc=ORANGE, mec=ORANGE, label="target (replaced)"),
    plt.Line2D([], [], ls="", marker="o", ms=6.5, mfc=BLUE, mec=BLUE, label="neighbour in the mean"),
    plt.Line2D([], [], ls="", marker="o", ms=6.5, mfc="white", mec=GREY, label="excluded ($|k|<n_1$)"),
    plt.Line2D([], [], ls="", marker="o", ms=5, mfc=LIGHT, mec=GREY, label="kept unchanged"),
]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4, handletextpad=0.3, columnspacing=1.2)
ax.set_xlim(-span - 7.5, span + 0.8)
ax.set_ylim(-len(rows) + 0.4, 1.4)
save(fig, "fig1_neighbourhood")

# ---------------------------------------------------------------- Figure 2: lowpass PSD
designs = [
    ("sweep $[1,5]$, $R=1$", [1, 5], 1, "descending", BLUE, "-"),
    ("sweep $[1,3]$, $R=4$", [1, 3], 4, "descending", ORANGE, "--"),
    ("sweep $[1,5]$, $R=2$", [1, 5], 2, "descending", AQUA, "-."),
]
WELCH_FLOOR_DB = -135
f = np.linspace(0, 0.5, 4001)
rng = np.random.default_rng(0)
x = rng.standard_normal(2_000_000)

fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), gridspec_kw=dict(width_ratios=[1.35, 1]))
for name, lags, R, sweep, col, ls in designs:
    S_th = odd_even_psd(2 * np.pi * f, lags, R, sweep)                       # two-sided PSD for unit white input
    y = apply_selective_filter(x, "Odd-Even", lags, R, sweep, True)
    fw, S_w = welch(y, fs=1.0, nperseg=4096, detrend=False, return_onesided=True)
    S_w = S_w / 2                                                            # one-sided Welch -> two-sided density
    sig = odd_even_moments(lags, R, sweep)["sigma"]
    label = f"{name}  ($\\sigma={sig:.2f}$)"
    S_w_db = 10 * np.log10(np.maximum(S_w, 1e-30))
    S_w_db[S_w_db < WELCH_FLOOR_DB] = np.nan                                 # below this the estimate is leakage, not signal
    S_th_at_w = 10 * np.log10(np.maximum(odd_even_psd(2 * np.pi * fw, lags, R, sweep), 1e-30))
    ok = ~np.isnan(S_w_db)
    print(f"{name}: Welch vs theory where Welch > {WELCH_FLOOR_DB} dB: "
          f"median |diff| {np.median(np.abs(S_w_db[ok] - S_th_at_w[ok])):.2f} dB over {ok.sum()} bins")
    for ax in axes:
        ax.plot(fw, S_w_db, color=col, lw=2.6, alpha=0.22, solid_capstyle="butt", zorder=2)   # estimate as a soft band
        ax.plot(f, 10 * np.log10(np.maximum(S_th, 1e-30)), color=col, ls=ls, lw=1.2, label=label, zorder=3)
sig = odd_even_moments([1, 5], 1, "descending")["sigma"]
gauss = np.exp(-(sig * 2 * np.pi * f) ** 2)                                   # |H|^2 of a Gaussian of equal sigma
for ax in axes:
    ax.plot(f, 10 * np.log10(np.maximum(gauss, 1e-30)), color=GREY, ls=":", lw=1.1,
            label=f"Gaussian, $\\sigma={sig:.2f}$", zorder=2)
axes[0].set(xlim=(0, 0.5), ylim=(-200, 8), xlabel="frequency $f$ (cycles/sample)", ylabel="PSD (dB)")
axes[0].set_title("(a) full band", loc="left")
axes[1].set(xlim=(0, 0.12), ylim=(-100, 4), xlabel="frequency $f$ (cycles/sample)")
axes[1].set_title("(b) main lobe and first sidelobes", loc="left")
axes[1].axhline(-3, color=GREY, lw=0.5)
axes[1].text(0.119, -2.4, "$-3$ dB", ha="right", va="bottom", fontsize=7, color="#5c5c58")
h, l = axes[0].get_legend_handles_labels()
h.append(plt.Line2D([], [], color=GREY, lw=2.6, alpha=0.35)); l.append("Welch estimate (shaded, above $-135$ dB)")
fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=3, columnspacing=1.2, handlelength=2.4)
fig.tight_layout()
save(fig, "fig2_lowpass_psd")

# ---------------------------------------------------------------- Figure 3: closed forms against the algorithm
designs = [
    ("sweep $[1,5]$, $R=1$", [1, 5], 1, "descending", BLUE),
    ("fixed lags $[2,4]$, $R=3$", [2, 4], 3, "none", ORANGE),
]
N = 4096
impulse = np.zeros(N); impulse[N // 2] = 1.0
fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.4), gridspec_kw=dict(width_ratios=[1.15, 1, 1]))
for name, lags, R, sweep, col in designs:
    n, h = odd_even_impulse_response(lags, R, sweep)
    y = apply_selective_filter(impulse, "Odd-Even", lags, R, sweep, True)
    h_alg = y[N // 2 + n]                                                    # algorithm output on the kernel support
    print(f"{name}: max |h_alg - h_theory| = {np.abs(h_alg - h).max():.1e}, "
          f"mass outside support = {abs(y.sum() - h_alg.sum()):.1e}")
    axes[0].plot(n, h, color=col, lw=1.0, label=name, zorder=2)
    axes[0].plot(n, h_alg, ls="", marker="o", ms=2.6, mfc="white", mec=col, mew=0.8, zorder=3)
    H_alg = np.abs(np.fft.rfft(np.fft.ifftshift(y)))                         # transfer function of the measured response
    f_fft = np.fft.rfftfreq(N)
    H_th = odd_even_frequency_response(2 * np.pi * f, lags, R, sweep)
    axes[1].plot(f, 20 * np.log10(np.maximum(H_th, 1e-30)), color=col, lw=1.0, zorder=2)
    axes[1].plot(f_fft[::24], 20 * np.log10(np.maximum(H_alg[::24], 1e-30)), ls="", marker="o", ms=2.4,
                 mfc="white", mec=col, mew=0.7, zorder=3)
    yw = apply_selective_filter(x[:400_000], "Odd-Even", lags, R, sweep, True)
    fw, S_w = welch(yw, fs=1.0, nperseg=1024, detrend=False, return_onesided=True)
    S_w_db = 10 * np.log10(np.maximum(S_w / 2, 1e-30))
    S_w_db[S_w_db < -60] = np.nan                                            # 400k samples: estimate floor near -60 dB
    axes[2].plot(fw, S_w_db, color=col, lw=2.6, alpha=0.22, solid_capstyle="butt", zorder=2)
    axes[2].plot(f, 10 * np.log10(np.maximum(odd_even_psd(2 * np.pi * f, lags, R, sweep), 1e-30)),
                 color=col, lw=1.0, zorder=3)
axes[0].set(xlim=(-16, 16), xlabel="offset $n$", ylabel="$h[n]$")
axes[0].set_title("(a) impulse response", loc="left")
axes[1].set(xlim=(0, 0.5), ylim=(-120, 5), xlabel="frequency $f$", ylabel="$|H|$ (dB)")
axes[1].set_title("(b) transfer function", loc="left")
axes[2].set(xlim=(0, 0.5), ylim=(-70, 5), xlabel="frequency $f$", ylabel="PSD (dB)")
axes[2].set_title("(c) output PSD, white input", loc="left")
h, l = axes[0].get_legend_handles_labels()
h += [plt.Line2D([], [], color=GREY, lw=1.0),
      plt.Line2D([], [], ls="", marker="o", ms=4, mfc="white", mec=GREY),
      plt.Line2D([], [], color=GREY, lw=2.6, alpha=0.35)]
l += ["closed form", "selective algorithm", "Welch estimate"]
fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=5, columnspacing=1.2, handlelength=2.0)
fig.tight_layout()
save(fig, "fig3_validation")
print("saved:", sorted(os.listdir(OUT)))
