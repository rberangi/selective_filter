"""
Selective filters: reference implementation and closed-form theory of the Odd-Even member.

A selective step replaces a chosen set of samples by the mean of their neighbours at
prescribed lags and leaves the other samples unchanged. Averaging the steps over a
partition of the samples (for example odd and even indices) cancels the selection
exactly and gives a linear, zero-phase FIR smoother with a closed-form response.
"""
import numpy as np


def selective_filter(y, target, lags, include_borders, diff_val=0, diff_mode = "none"):
    """
    A localized adaptive filtering method that suppresses peaks and troughs through neighborhood averaging.
    Applies a localized neighborhood-based selective filter to a 1D signal.

    This function selectively replaces elements in the input signal `y` based on a
    specified target pattern. These patterns include local maxima, local minima,
    all samples, or specific index-based selections (odd or even). Replacement is done
    by averaging neighboring values within a specified lag window.

    Parameters
    ----------
    y : array-like
        Input 1D signal to be filtered.

    target : str
        Determines which values to replace:
        - "Max": Local maxima.
        - "Min": Local minima.
        - "All": All values.
        - "Odd": Values at odd indices.
        - "Even": Values at even indices.

    lags : list or tuple of two ints
        The lag window [n1, n2], where:
        - n1 ≥ 0 and n2 ≥ n1.
        - Neighbors are selected from:
            y[n - n2 : n - n1] and y[n + n1 : n + n2]
        - If n1 == 0, the center value is included in the averaging.

    include_borders : bool
        Whether to apply reflection padding to include edge values in detection and filtering.

    diff_val : float, optional (default=0)
        A threshold for minimum gradient magnitude used to qualify maxima/minima.
        Only applicable when `target` is "Max" or "Min".

    diff_mode : str, optional (default="none")
        Method to compute the slope condition (if `diff_val` > 0):
        - "none": Use the average of left and right slopes.
        - "Min": Use the smaller of the left and right slopes.
        - "Max": Use the larger of the left and right slopes.

    Returns
    -------
    np.ndarray
        A modified copy of the input signal with selected values replaced by
        the average of their surrounding neighbors.
    """
    n1 = lags[0]
    n2 = lags[1]
    max_lag = n2
    y = np.array(y, dtype=float)

    # Padding for extrema detection
    if include_borders:
        yb = np.pad(y, pad_width=1, mode="reflect")
        b_dist = 0
    else:
        yb = y
        b_dist = 1

    # Padding for substitution
    yd = np.pad(y, pad_width=max_lag, mode="reflect")

    # Derivatives
    dy1 = yb[1:-1] - yb[:-2]
    dy2 = yb[1:-1] - yb[2:]

    # target mask
    if target == "Max":
        n = (dy1 >= 0) & (dy2 >= 0)
        if diff_val > 0:
            if diff_mode == "Min":
                n &= (np.minimum(np.abs(dy1), np.abs(dy2)) > diff_val)
            elif diff_mode == "Max":
                n &= (np.maximum(np.abs(dy1), np.abs(dy2)) > diff_val)
            else:
                n &= (((np.abs(dy1) + np.abs(dy2)) / 2) > diff_val)
    elif target == "Min":
        n = (dy1 <= 0) & (dy2 <= 0)
        if diff_val > 0:
            if diff_mode == "Min":
                n &= (np.minimum(np.abs(dy1), np.abs(dy2)) > diff_val)
            elif diff_mode == "Max":
                n &= (np.maximum(np.abs(dy1), np.abs(dy2)) > diff_val)
            else:
                n &= (((np.abs(dy1) + np.abs(dy2)) / 2) > diff_val)
    elif target == "All":
        n = np.ones(len(yb) - 2, dtype=bool)
    elif target == "Odd":
        n = (np.arange(len(yb) - 2) + b_dist) % 2 == 1  # index 1, 3, 5, ... of y, with or without borders
    elif target == "Even":
        n = (np.arange(len(yb) - 2) + b_dist) % 2 == 0  # index 0, 2, 4, ... of y
    else:
        raise ValueError('"target" must be "Max", "Min", "All", "Odd", or "Even"')

    xn1 = np.where(n)[0] + b_dist         # Target indices in original y
    xn2 = xn1 + max_lag                   # Corresponding target indices in padded yd

    if len(xn1) == 0:
        return y  # Nothing to modify

    # Create offsets
    if n1==0:
        all_samples =  np.arange(-n2,n2+1)
    else:
        all_samples = np.concatenate((np.arange(-n2, 1-n1), np.arange(n1, n2+1)))    
    # Shape (num_centers, num_offsets)
    sample_matrix = yd[xn2[:, None] + all_samples]

    # Compute mean for each center
    avg_vals = np.mean(sample_matrix, axis=1)
    # Apply the result
    y[xn1] = avg_vals
    return y
def apply_selective_filter(y, target, lags, repeats, sweep_lags, include_borders, diff_val=0, diff_mode = "none"):
    """
    Applies repeated neighborhood-based selective filtering to a 1D signal,
    with optional lag sweeping and composite target types.

    This function performs multi-pass filtering using the `selective_filter` function.
    It supports composite targets (e.g., "Max-Min", "Odd-Even"), repeated passes, and
    dynamic lag sweeping for progressively refining the filter effect.

    Recommendations
    ---------------
    - Use target="Max-Min" with sweep_lags="descending" for effective low-pass filtering.
    - Larger maximum lag (`n2`) results in a wider smoothing window.
    - A positive `diff_val` can be used to exclude weak peaks/troughs.

    Parameters
    ----------
    y : array-like
        Input 1D signal to be filtered.

    target : str
        Type of features to filter:
        - "Max": Local maxima only.
        - "Min": Local minima only.
        - "All": All values.
        - "Max-Min": Apply both Max and Min filters, average the result.
        - "Odd-Even": Apply both Odd and Even filters, average the result.
        - "Odd": Odd indices only.
        - "Even": Even indices only.

    lags : list or tuple of two ints
        Lag window [n1, n2] specifying the averaging neighborhood.

    repeats : int
        Number of times to apply the filtering process.

    sweep_lags : str
        Defines lag sweeping behavior:
        - "ascending": Increase lag size from n1 to n2.
        - "descending": Decrease lag size from n2 to n1.
        - "none": Use fixed lag window only.

    include_borders : bool
        Whether to pad signal using reflection to allow edge processing.

    diff_val : float, optional (default=0)
        Minimum gradient magnitude threshold for qualifying extrema.
        Only used for "Max", "Min", or "Max-Min" targets.

    diff_mode : str, optional (default="none")
        Method for computing slope comparison in `selective_filter`:
        - "none": Use average of left and right slopes.
        - "Min": Use minimum slope.
        - "Max": Use maximum slope.

    Returns
    -------
    np.ndarray
        Filtered 1D signal after applying all passes and lag sweeps.
    """
    n1 = lags[0]
    n2 = lags[1]
    y = np.array(y, dtype=float)
    k1 = max(n1, 1)  # a sweep step with k = 0 replaces each target by itself, so it is skipped

    if sweep_lags == "ascending":
        sw = np.arange(k1, n2+1)[:, None].repeat(2, axis=1)
        do_sweep = True
    elif sweep_lags == "descending":
        sw = np.arange(n2,k1-1,-1)[:, None].repeat(2, axis=1)
        do_sweep = True
    elif sweep_lags == "none":
        do_sweep = False
    else:
        raise ValueError('The "sweep_lags" must be: "ascending", "descending" or "none"')   
    for repeat in range(repeats):
        if target in ["Max", "Min", "All", "Odd", "Even"]:
            if do_sweep:
                for lags0 in sw:
                    y = selective_filter(y, target, lags0, include_borders, diff_val, diff_mode)
            else:
                y = selective_filter(y, target, lags, include_borders, diff_val, diff_mode)
        elif target == "Max-Min":
            if do_sweep:
                for lags0 in sw:
                    y1 = selective_filter(y, "Max", lags0, include_borders, diff_val, diff_mode)
                    y2 = selective_filter(y, "Min", lags0, include_borders, diff_val, diff_mode)
                    y = (y1 + y2) / 2
            else:
                y1 = selective_filter(y, "Max", lags, include_borders, diff_val, diff_mode)
                y2 = selective_filter(y, "Min", lags, include_borders, diff_val, diff_mode)
                y = (y1 + y2) / 2
        elif target == "Odd-Even":
            if do_sweep:
                for lags0 in sw:
                    y1 = selective_filter(y, "Odd", lags0, include_borders, diff_val, diff_mode)
                    y2 = selective_filter(y, "Even", lags0, include_borders, diff_val, diff_mode)
                    y = (y1 + y2) / 2
            else:
                y1 = selective_filter(y, "Odd", lags, include_borders, diff_val, diff_mode)
                y2 = selective_filter(y, "Even", lags, include_borders, diff_val, diff_mode)
                y = (y1 + y2) / 2                

        else:
            raise ValueError('The "target" must be: "Max", "Min", "Max-Min", "Odd", "Even", "Odd-Even" or "All"')

    return y

# ---------------------------------------------------------------------------
# Theory of the "Odd-Even" option
#
# One Odd-Even step is the linear, shift-invariant update y <- (y + A y) / 2,
# where A averages y over the lag offsets S. Repeats and lag sweeps therefore
# collapse into a single zero-phase FIR kernel. The functions below give that
# kernel and its spectra in closed form (valid away from the signal borders).
# ---------------------------------------------------------------------------
def _odd_even_offsets(n1, n2):
    """Offset set S used by selective_filter for lags [n1, n2]."""
    if n1 == 0:
        return np.arange(-n2, n2 + 1)
    return np.concatenate((np.arange(-n2, 1 - n1), np.arange(n1, n2 + 1)))


def _odd_even_is_sweep(sweep_lags):
    if sweep_lags in ("ascending", "descending"):
        return True
    if sweep_lags == "none":
        return False
    raise ValueError('The "sweep_lags" must be: "ascending", "descending" or "none"')


def odd_even_impulse_response(lags, repeats, sweep_lags):
    """
    Theoretical impulse response of apply_selective_filter(..., target="Odd-Even").

    - sweep_lags="none":  h = [delta/2 + U_S/2]^{*R}, U_S uniform on the offsets S.
    - sweep ("ascending" or "descending", identical results):
      h = conv over k of binomial kernels C(2R, R+m) / 4^R placed at m*k, m = -R..R,
      i.e. the coefficients of prod_k ((z^(k/2) + z^(-k/2)) / 2)^(2R).

    Parameters
    ----------
    lags : list or tuple of two ints
        Lag window [n1, n2], as in apply_selective_filter.
    repeats : int
        Number of repeats R.
    sweep_lags : str
        "ascending", "descending" or "none".

    Returns
    -------
    n : np.ndarray
        Sample offsets -L..L.
    h : np.ndarray
        Kernel values at n (sums to 1, symmetric, non-negative).
    """
    from math import comb
    n1, n2 = lags
    h = np.array([1.0])
    if _odd_even_is_sweep(sweep_lags):
        binom = np.array([comb(2 * repeats, j) for j in range(2 * repeats + 1)]) / 4.0**repeats
        for k in range(max(n1, 1), n2 + 1):
            hk = np.zeros(2 * k * repeats + 1)
            hk[::k] = binom
            h = np.convolve(h, hk)
    else:
        s = _odd_even_offsets(n1, n2)
        step = np.zeros(2 * n2 + 1)
        np.add.at(step, s + n2, 0.5 / len(s))
        step[n2] += 0.5
        for _ in range(repeats):
            h = np.convolve(h, step)
    L = len(h) // 2
    return np.arange(-L, L + 1), h


def odd_even_frequency_response(w, lags, repeats, sweep_lags):
    """
    Theoretical (real, zero-phase) frequency response H(w) of the Odd-Even filter.

    - sweep_lags="none":  H = [1/2 + M(w)/2]^R, M(w) = mean over S of cos(k w).
    - sweep:              H = prod_{k=max(n1,1)}^{n2} cos^(2R)(k w / 2).

    0 <= H(w) <= 1 for all w.

    Parameters
    ----------
    w : array-like
        Angular frequency in rad/sample (0..pi). Use w = 2*pi*f/fs for f in Hz.
    lags, repeats, sweep_lags
        As in apply_selective_filter.

    Returns
    -------
    np.ndarray
        H(w).
    """
    n1, n2 = lags
    w = np.asarray(w, dtype=float)
    if _odd_even_is_sweep(sweep_lags):
        H = np.ones_like(w)
        for k in range(max(n1, 1), n2 + 1):
            H = H * np.cos(k * w / 2) ** (2 * repeats)
        return H
    s = _odd_even_offsets(n1, n2)
    M = np.cos(np.multiply.outer(w, s)).mean(axis=-1)
    return (0.5 + 0.5 * M) ** repeats


def odd_even_psd(w, lags, repeats, sweep_lags, input_psd=1.0):
    """
    Output power spectral density S_y(w) = H(w)^2 * S_x(w) of the Odd-Even filter.

    For white input of variance sigma^2 pass input_psd=sigma^2; the output
    autocorrelation is then sigma^2 * h_{2R}[lag], the impulse response with
    twice the repeats.

    Parameters
    ----------
    w : array-like
        Angular frequency in rad/sample.
    lags, repeats, sweep_lags
        As in apply_selective_filter.
    input_psd : float or array-like, optional (default=1.0)
        Input PSD S_x(w), scalar for white noise.

    Returns
    -------
    np.ndarray
        S_y(w).
    """
    return odd_even_frequency_response(w, lags, repeats, sweep_lags) ** 2 * input_psd


def odd_even_moments(lags, repeats, sweep_lags):
    """
    Closed-form statistics of the Odd-Even kernel.

    Returns
    -------
    dict
        - "variance": sigma^2 = R/2 * sum(k^2) (sweep) or R/2 * E_S[k^2] (none).
        - "sigma": equivalent Gaussian standard deviation in samples.
        - "excess_kurtosis": departure from a Gaussian kernel, decays like 1/R.
        - "noise_gain": sum(h^2) = output/input variance for white noise.
        - "noise_gain_gaussian": Gaussian approximation 1 / (2 sigma sqrt(pi)).
        - "cutoff_3db": approximate -3 dB cutoff sqrt(ln 2) / sigma in rad/sample.
    """
    n1, n2 = lags
    if _odd_even_is_sweep(sweep_lags):
        k = np.arange(max(n1, 1), n2 + 1, dtype=float)
        step_var = 0.5 * k**2
        step_k4 = 0.5 * k**4 - 3 * step_var**2       # 4th cumulant per sweep step
        var = repeats * step_var.sum()
        k4 = repeats * step_k4.sum()
    else:
        s = _odd_even_offsets(n1, n2).astype(float)
        m2, m4 = np.mean(s**2), np.mean(s**4)
        var = repeats * m2 / 2
        k4 = repeats * (m4 / 2 - 3 * m2**2 / 4)
    sigma = np.sqrt(var)
    _, h = odd_even_impulse_response(lags, repeats, sweep_lags)
    return {
        "variance": var,
        "sigma": sigma,
        "excess_kurtosis": k4 / var**2 if var > 0 else 0.0,
        "noise_gain": float(np.sum(h**2)),
        "noise_gain_gaussian": 1 / (2 * sigma * np.sqrt(np.pi)) if var > 0 else 1.0,
        "cutoff_3db": np.sqrt(np.log(2)) / sigma if var > 0 else np.pi,
    }


# ---------------------------------------------------------------------------
# Odd-Even bandpass filters
#
# Built only from Odd-Even steps, so they stay linear, zero-phase and
# multiplier-free (apart from the optional gain normalisation):
#   lowpass step  y <- y/2 + (y[n-k] + y[n+k])/4,  response cos^2(k w / 2)
#   highpass step y <- y - lowpass_step(y),         response sin^2(k w / 2)
#
# "product":    r lowpass and s highpass steps at lag d, plus an optional
#               ascending lowpass sweep (image_lags) that removes the band images
#               at f0/d + m/d. Narrow band; centre (k=1) f0 = arctan(sqrt(s/r))/pi.
# "difference": y_A - y_B for ascending sweeps A = [n1, kA], B = [n1, kB], kB > kA.
#               Wide (~1.5 octave) band, gain near 1, H >= 0.
# ---------------------------------------------------------------------------
def _odd_even_step(y, k, include_borders):
    return selective_filter(y, "Odd", [k, k], include_borders) / 2 + selective_filter(y, "Even", [k, k], include_borders) / 2


def apply_odd_even_bandpass(y, method, r=2, s=2, dilation=1, image_lags=None,
                            lags_a=(1, 2), lags_b=(1, 6), repeats=1,
                            include_borders=True, normalize=True):
    """
    Multiplier-free bandpass filter built from Odd-Even steps.

    Parameters
    ----------
    y : array-like
        Input 1D signal.

    method : str
        - "product":    H = cos^(2r)(d w/2) * sin^(2s)(d w/2) * H_image(w).
        - "difference": H = H_A^R - H_B^R for ascending Odd-Even sweeps.

    r, s : int, optional (method="product")
        Number of lowpass and highpass steps. The ratio s/r sets the centre,
        r + s the bandwidth (narrower for larger r + s).

    dilation : int, optional (default=1, method="product")
        Lag d of the steps. Moves the band to f0/d, with images at f0/d + m/d.

    image_lags : list or tuple of two ints or None, optional (method="product")
        Lags [n1, n2] of an ascending Odd-Even sweep applied afterwards to remove
        the images created by dilation > 1. None applies no sweep.

    lags_a, lags_b : list or tuple of two ints, optional (method="difference")
        Sweeps [n1, kA] and [n1, kB] with kB > kA. The band lies between the
        cutoffs of B (lower edge) and A (upper edge).

    repeats : int, optional (default=1, method="difference")
        Repeats R of both sweeps (sharper band edges).

    include_borders : bool, optional (default=True)
        As in apply_selective_filter.

    normalize : bool, optional (default=True)
        Scale so the peak gain is 1 (method="product" only; uses the theoretical
        peak). For r == s, dilation 1 and no image_lags this is exactly 4^r, a shift.

    Returns
    -------
    np.ndarray
        Bandpass-filtered signal.
    """
    y = np.array(y, dtype=float)
    if method == "product":
        for _ in range(r):
            y = _odd_even_step(y, dilation, include_borders)
        for _ in range(s):
            y = y - _odd_even_step(y, dilation, include_borders)
        if image_lags is not None:
            y = apply_selective_filter(y, "Odd-Even", image_lags, 1, "ascending", include_borders)
        if normalize:
            y = y / _odd_even_bandpass_peak("product", r, s, dilation, image_lags, lags_a, lags_b, repeats)
        return y
    if method == "difference":
        n1, ka = lags_a
        nb1, kb = lags_b
        if nb1 != n1 or kb <= ka:
            raise ValueError('"lags_b" must start at the same n1 as "lags_a" and end beyond it')
        y_a = apply_selective_filter(y, "Odd-Even", lags_a, repeats, "ascending", include_borders)
        y_b = apply_selective_filter(y_a, "Odd-Even", [ka + 1, kb], repeats, "ascending", include_borders)
        return y_a - y_b
    raise ValueError('The "method" must be: "product" or "difference"')


def odd_even_bandpass_frequency_response(w, method, r=2, s=2, dilation=1, image_lags=None,
                                         lags_a=(1, 2), lags_b=(1, 6), repeats=1, normalize=True):
    """
    Theoretical (real, zero-phase) frequency response of apply_odd_even_bandpass.

    Parameters are the same as apply_odd_even_bandpass, with w the angular
    frequency in rad/sample. Returns H(w).
    """
    w = np.asarray(w, dtype=float)
    if method == "product":
        H = np.cos(dilation * w / 2) ** (2 * r) * np.sin(dilation * w / 2) ** (2 * s)
        if image_lags is not None:
            H = H * odd_even_frequency_response(w, image_lags, 1, "ascending")
        if normalize:
            H = H / _odd_even_bandpass_peak("product", r, s, dilation, image_lags, lags_a, lags_b, repeats)
        return H
    if method == "difference":
        return (odd_even_frequency_response(w, lags_a, repeats, "ascending")
                - odd_even_frequency_response(w, lags_b, repeats, "ascending"))
    raise ValueError('The "method" must be: "product" or "difference"')


def odd_even_bandpass_impulse_response(method, r=2, s=2, dilation=1, image_lags=None,
                                       lags_a=(1, 2), lags_b=(1, 6), repeats=1, normalize=True):
    """
    Theoretical impulse response of apply_odd_even_bandpass.

    Lowpass steps are the 3-tap kernel [1/4, 1/2, 1/4] and highpass steps
    [-1/4, 1/2, -1/4], both at offsets (-d, 0, d); the response is their convolution.

    Returns
    -------
    n : np.ndarray
        Sample offsets -L..L.
    h : np.ndarray
        Kernel values at n.
    """
    if method == "product":
        def tap3(side):
            k = np.zeros(2 * dilation + 1)
            k[[0, -1]], k[dilation] = side, 0.5
            return k
        h = np.array([1.0])
        for _ in range(r):
            h = np.convolve(h, tap3(0.25))
        for _ in range(s):
            h = np.convolve(h, tap3(-0.25))
        if image_lags is not None:
            h = np.convolve(h, odd_even_impulse_response(image_lags, 1, "ascending")[1])
        if normalize:
            h = h / _odd_even_bandpass_peak("product", r, s, dilation, image_lags, lags_a, lags_b, repeats)
    elif method == "difference":
        _, h_a = odd_even_impulse_response(lags_a, repeats, "ascending")
        _, h_b = odd_even_impulse_response(lags_b, repeats, "ascending")
        pad = (len(h_b) - len(h_a)) // 2
        h = np.pad(h_a, pad) - h_b
    else:
        raise ValueError('The "method" must be: "product" or "difference"')
    L = len(h) // 2
    return np.arange(-L, L + 1), h


def odd_even_bandpass_properties(method, r=2, s=2, dilation=1, image_lags=None,
                                 lags_a=(1, 2), lags_b=(1, 6), repeats=1, normalize=True):
    """
    Centre frequency, -3 dB band edges and peak gain of the bandpass (from theory).

    Returns
    -------
    dict
        "centre", "low_3db", "high_3db" in cycles/sample (w / 2pi), and "peak_gain".
    """
    f = np.linspace(0, 0.5, 20001)
    H = odd_even_bandpass_frequency_response(2 * np.pi * f, method, r, s, dilation, image_lags,
                                             lags_a, lags_b, repeats, normalize)
    p = np.argmax(H)
    half = H[p] / np.sqrt(2)
    below = np.nonzero(H[:p] < half)[0]
    above = np.nonzero(H[p:] < half)[0]
    return {
        "centre": f[p],
        "low_3db": f[below[-1]] if len(below) else 0.0,
        "high_3db": f[p + above[0]] if len(above) else 0.5,
        "peak_gain": H[p],
    }


def _odd_even_bandpass_peak(method, r, s, dilation, image_lags, lags_a, lags_b, repeats):
    """Peak of the un-normalised "product" response (exact r^r s^s / (r+s)^(r+s) without image_lags)."""
    if image_lags is None:
        return r**r * s**s / (r + s) ** (r + s)
    w = np.linspace(0, np.pi, 20001)
    return np.max(odd_even_bandpass_frequency_response(w, method, r, s, dilation, image_lags,
                                                       lags_a, lags_b, repeats, normalize=False))


# ---------------------------------------------------------------------------
# Continuous signals
#
# Every selective filter has a finite influence radius L: an output sample
# depends only on input samples within +-L. A stream can therefore be filtered
# block by block with overlap-save, which is exact for all targets, including
# the nonlinear Max/Min ones (overlap-add would only suit the linear targets).
# ---------------------------------------------------------------------------
def selective_filter_radius(lags, repeats, sweep_lags):
    """
    Influence radius L of apply_selective_filter: L = R * sum(k) for a lag sweep, R * n2 otherwise.
    """
    n1, n2 = lags
    if sweep_lags in ("ascending", "descending"):
        return repeats * sum(range(max(n1, 1), n2 + 1))
    if sweep_lags == "none":
        return repeats * n2
    raise ValueError('The "sweep_lags" must be: "ascending", "descending" or "none"')


def apply_selective_filter_stream(blocks, target, lags, repeats, sweep_lags, include_borders, diff_val=0, diff_mode="none"):
    """
    Overlap-save version of apply_selective_filter for continuous signals.

    Consumes the signal as an iterable of blocks of any length and yields
    filtered blocks. The concatenated output equals apply_selective_filter
    applied to the whole signal, for every target. The border policy
    (reflection) is used only at the true start and end of the stream.

    Each output sample needs L future input samples (see
    selective_filter_radius), so output lags input by about L samples
    plus the block length.

    Parameters
    ----------
    blocks : iterable of array-like
        Consecutive pieces of the input signal.

    target, lags, repeats, sweep_lags, include_borders, diff_val, diff_mode
        As in apply_selective_filter.

    Yields
    ------
    np.ndarray
        Consecutive pieces of the filtered signal (possibly empty).
    """
    L = selective_filter_radius(lags, repeats, sweep_lags)
    buf = np.empty(0)
    start = 0      # absolute index of buf[0]
    emitted = 0    # absolute index of the next output sample
    for block in blocks:
        buf = np.concatenate((buf, np.asarray(block, dtype=float)))
        ready = start + len(buf) - L          # outputs before this index have their full right context
        if ready > emitted:
            y = apply_selective_filter(buf, target, lags, repeats, sweep_lags, include_borders, diff_val, diff_mode)
            yield y[emitted - start : ready - start]
            emitted = ready
            # keep L samples of left context; an even start keeps the Odd/Even parity of the indices
            keep = max(start, emitted - L)
            keep -= (keep - start) % 2
            buf, start = buf[keep - start:], keep
    if emitted < start + len(buf):
        y = apply_selective_filter(buf, target, lags, repeats, sweep_lags, include_borders, diff_val, diff_mode)
        yield y[emitted - start:]
