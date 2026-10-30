"""Selective filters and the closed-form Odd-Even smoother."""
from .core import (
    apply_odd_even_bandpass,
    apply_selective_filter,
    apply_selective_filter_stream,
    odd_even_bandpass_frequency_response,
    odd_even_bandpass_impulse_response,
    odd_even_bandpass_properties,
    odd_even_frequency_response,
    odd_even_impulse_response,
    odd_even_moments,
    odd_even_psd,
    selective_filter,
    selective_filter_radius,
)

__version__ = "0.1.0"

__all__ = [
    "selective_filter",
    "apply_selective_filter",
    "apply_selective_filter_stream",
    "selective_filter_radius",
    "odd_even_impulse_response",
    "odd_even_frequency_response",
    "odd_even_psd",
    "odd_even_moments",
    "apply_odd_even_bandpass",
    "odd_even_bandpass_frequency_response",
    "odd_even_bandpass_impulse_response",
    "odd_even_bandpass_properties",
]
