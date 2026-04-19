"""
gw_datagen — Gravitational Wave Data Generation Library

Extracted from JHPY.py. Contains all data generation, modified gravity physics,
save/load, and preprocessing utilities. Does not include neural network code.
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split, Subset
from tqdm import tqdm
from typing import Dict, List, Callable, Tuple
from functools import partial
import multiprocessing
from multiprocessing import Pool
from pycbc.waveform import get_td_waveform, get_fd_waveform
import os
import logging
from scipy.integrate import quad
from scipy.interpolate import interp1d
from pycbc.detector import Detector
from pycbc.psd import aLIGOZeroDetHighPower
from pycbc.noise import noise_from_psd
from pycbc.types import TimeSeries, FrequencySeries
from pycbc.psd import welch, interpolate
from pycbc.filter import highpass_fir, lowpass_fir, resample_to_delta_t
import pandas as pd
import warnings
from pathlib import Path
from time import perf_counter

logger = logging.getLogger(__name__)


################### O4 PSD cache utilities ###################

# O4a GPS window (2023-05-24 18:00 UTC → 2024-01-16 16:00 UTC)
_O4A_GPS_START  = 1369166418
_O4A_GPS_END    = 1389744018

_FETCH_DUR      = 32    # seconds of data per segment
_FFT_LEN        = 32    # Welch FFT length in seconds
_DEFAULT_N_SEGS = 100
_DEFAULT_CACHE  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'o4_psd_cache')


def _cache_path(detector: str, sample_rate: int, cache_dir: str) -> str:
    return os.path.join(cache_dir, f'o4_psds_{detector}_{sample_rate}Hz.npz')


def _sample_science_gps(detector: str, n: int, seed: int = 42) -> np.ndarray:
    """Return ``n`` GPS start times drawn from O4a science-mode segments.

    Uses the ``gwosc`` package to query which GPS times actually have detector
    data, so every returned time is guaranteed to be fetchable.  Falls back to
    a simple linspace if ``gwosc`` is unavailable.
    """
    _BUFFER = 21600  # 6-hour inset from run boundaries
    try:
        from gwosc.timeline import get_segments
        segs = get_segments(
            f'{detector}_DATA',
            _O4A_GPS_START + _BUFFER,
            _O4A_GPS_END   - _BUFFER,
        )
    except Exception:
        logger.warning('o4_psd: gwosc segment query failed, falling back to linspace')
        return np.linspace(
            _O4A_GPS_START + _BUFFER,
            _O4A_GPS_END   - _FETCH_DUR - _BUFFER,
            n,
        ).astype(int)

    pool = []
    for seg_start, seg_end in segs:
        t = int(seg_start)
        while t + _FETCH_DUR <= int(seg_end):
            pool.append(t)
            t += _FETCH_DUR

    if not pool:
        raise RuntimeError(f'o4_psd: no O4a science segments found for {detector}')

    rng = np.random.default_rng(seed)
    chosen = rng.choice(pool, size=min(n, len(pool)), replace=False)
    return np.sort(chosen)


def build_o4_psd_cache(
    detector: str,
    n_segments: int = _DEFAULT_N_SEGS,
    sample_rate: int = 4096,
    cache_dir: str = _DEFAULT_CACHE,
    force: bool = False,
) -> None:
    """Fetch ``n_segments`` O4a data segments and save their Welch PSDs.

    Parameters
    ----------
    detector : str
        Detector name, e.g. ``'H1'`` or ``'L1'``.
    n_segments : int
        Number of evenly-spaced segments to fetch across O4a.
    sample_rate : int
        Target sample rate in Hz.  GWOSC data will be resampled if needed.
    cache_dir : str
        Directory to store the ``.npz`` cache file.
    force : bool
        If True, re-fetch even if the cache already exists.
    """
    path = _cache_path(detector, sample_rate, cache_dir)
    if not force and os.path.exists(path):
        logger.debug('o4_psd: cache already exists at %s', path)
        return

    try:
        from gwpy.timeseries import TimeSeries as GWTimeSeries
    except ImportError as exc:
        raise ImportError(
            'gwpy is required to build the O4 PSD cache.  '
            'Install it with: pip install gwpy'
        ) from exc

    os.makedirs(cache_dir, exist_ok=True)

    gps_starts = _sample_science_gps(detector, n_segments)

    all_freqs = None
    all_psds  = []
    n_failed  = 0

    for i, gps in enumerate(gps_starts):
        try:
            data = GWTimeSeries.fetch_open_data(
                detector, gps, gps + _FETCH_DUR, verbose=False
            )
        except Exception as exc:
            logger.warning('o4_psd: skipping GPS %d (%s)', gps, exc)
            n_failed += 1
            continue

        if int(round(data.sample_rate.value)) != sample_rate:
            data = data.resample(sample_rate)

        psd_gwpy = data.psd(
            fftlength=_FFT_LEN,
            overlap=_FFT_LEN // 2,
            method='welch',
            window='hann',
        )

        freqs = np.array(psd_gwpy.frequencies.value, dtype=np.float64)
        pvals = np.array(psd_gwpy.value,             dtype=np.float64)

        valid = (freqs > 0) & (pvals > 0)
        freqs = freqs[valid]
        pvals = pvals[valid]

        if len(pvals) == 0:
            logger.warning('o4_psd: GPS %d has no valid PSD values, skipping', gps)
            n_failed += 1
            continue

        if all_freqs is None:
            all_freqs = freqs
        else:
            if not np.array_equal(freqs, all_freqs):
                interp_fn = interp1d(
                    np.log10(freqs), np.log10(pvals),
                    bounds_error=False,
                    fill_value=(np.log10(pvals[0]), np.log10(pvals[-1])),
                )
                pvals = 10 ** interp_fn(np.log10(all_freqs))

        all_psds.append(pvals)

        if (i + 1) % 10 == 0 or i == len(gps_starts) - 1:
            logger.info(
                'o4_psd: %s — %d/%d segments done (%d failed)',
                detector, i + 1, len(gps_starts), n_failed,
            )

    if not all_psds:
        raise RuntimeError(
            f'o4_psd: all {n_segments} GWOSC fetches failed for {detector}. '
            'Check network access or try a different cache_dir.'
        )

    np.savez_compressed(
        path,
        freqs=all_freqs,
        psds=np.array(all_psds, dtype=np.float64),
    )
    logger.info(
        'o4_psd: saved %d PSDs for %s to %s (%d failed)',
        len(all_psds), detector, path, n_failed,
    )


def load_random_o4_psd(
    flen: int,
    delta_f: float,
    f_lower: float,
    detector: str,
    sample_rate: int = 4096,
    cache_dir: str = _DEFAULT_CACHE,
    rng: np.random.Generator = None,
) -> FrequencySeries:
    """Return a randomly selected O4 PSD interpolated onto the required grid.

    Parameters
    ----------
    flen : int
        Number of frequency bins (``target_length // 2 + 1``).
    delta_f : float
        Frequency resolution in Hz.
    f_lower : float
        Lower frequency cutoff.  Bins below this are set to the PSD value at
        f_lower so ``noise_from_psd`` does not generate catastrophic sub-band energy.
    detector : str
        Detector name, e.g. ``'H1'``.
    sample_rate : int
        Sample rate in Hz (must match the cached PSD).
    cache_dir : str
        Directory containing the ``.npz`` cache file.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    pycbc.types.FrequencySeries
        PSD with length ``flen`` and frequency resolution ``delta_f``.
    """
    path = _cache_path(detector, sample_rate, cache_dir)

    if not os.path.exists(path):
        logger.warning(
            'o4_psd: cache not found at %s — falling back to aLIGOZeroDetHighPower',
            path,
        )
        return aLIGOZeroDetHighPower(flen, delta_f, f_lower)

    data  = np.load(path)
    freqs = data['freqs']   # (M,)
    psds  = data['psds']    # (N, M)

    if rng is None:
        rng = np.random.default_rng()

    idx     = rng.integers(0, psds.shape[0])
    psd_row = psds[idx]

    f_out = np.arange(flen) * delta_f

    safe_freqs = freqs[freqs > 0]
    safe_psd   = psd_row[freqs > 0]

    interp_fn = interp1d(
        np.log10(safe_freqs),
        np.log10(safe_psd),
        bounds_error=False,
        fill_value=(np.log10(safe_psd[0]), np.log10(safe_psd[-1])),
    )

    pos_mask = f_out > 0
    psd_out  = np.empty(flen, dtype=np.float64)
    psd_out[~pos_mask] = 1.0
    psd_out[pos_mask]  = 10 ** interp_fn(np.log10(f_out[pos_mask]))

    above = f_out >= f_lower
    wall = psd_out[above][0] if above.any() else 1e-44
    psd_out[~above & (f_out > 0)] = wall

    psd_out[psd_out <= 0] = 1e-40

    return FrequencySeries(psd_out, delta_f=delta_f)


################### Cosmological / Modified Gravity Constants ###################
_C = 2.998e8                          # speed of light, m/s
_H_PLANCK = 6.626e-34                 # Planck's constant, J·s
_G = 6.674e-11                        # gravitational constant, m^3 kg^-1 s^-2
_M_SUN = 1.989e30                     # solar mass, kg
_MPC = 3.086e22                       # megaparsec, m
_M_SUN_SEC = _G * _M_SUN / _C**3     # solar mass in seconds (~4.926e-6 s)
_H0 = 67.4e3 / _MPC                  # Hubble constant in 1/s
_OMEGA_M = 0.315
_OMEGA_LAMBDA = 0.685
_HBAR_C_EV_M = _H_PLANCK / (2.0 * np.pi) * _C / 1.602e-19   # ℏc in eV·m
_M_OMEGA_PN = 0.1       # PN breakdown: (m1+m2)·omega = 0.1 (geometric units G=c=1)
_TAPER_FRACTION = 0.50  # phase taper: smooth to zero over top 50% of f_pn_cutoff
_HIGHPASS_FC        = 35  # high-pass filter cutoff (Hz) applied to all generated signals
_RINGDOWN_TAPER_LEN = 128    # samples cosine-tapered to zero at array end before highpass
# _N_RINGDOWN removed: waveforms are now centred on coalescence (merger at target_length//2)


def _luminosity_distance_mpc(z):
    """
    Analytical flat-ΛCDM luminosity distance via Adachi & Kasai (2012), Eq. 8-9.
    Returns d_L in Mpc.
    """
    s = ((1.0 - _OMEGA_M) / _OMEGA_M) ** (1.0 / 3.0)

    def _xi(a):
        return (2.0 * np.sqrt(s**3 + 1.0) *
                (a**-4 - 0.1540*s*a**-3 + 0.4304*s**2*a**-2
                 + 0.19097*s**3*a**-1 + 0.066941*s**4) ** (-1.0/8.0))

    a = 1.0 / (1.0 + z)
    d_C_m = (_C / _H0) * (_xi(1.0) - _xi(a))
    return (1.0 + z) * d_C_m / _MPC


_Z_GRID  = np.linspace(1e-4, 3.0, 10_000)
_DL_GRID = np.array([_luminosity_distance_mpc(z) for z in _Z_GRID])
_Z_FROM_DL_INTERP = interp1d(_DL_GRID, _Z_GRID,
                              kind='linear', bounds_error=False,
                              fill_value=(_Z_GRID[0], _Z_GRID[-1]))


def _redshift_from_distance(d_L_mpc):
    """Return redshift for a flat-ΛCDM luminosity distance d_L_mpc [Mpc]."""
    return float(_Z_FROM_DL_INTERP(d_L_mpc))


def _D_alpha(alpha, z):
    """
    Compute cosmological distance D_alpha for modified dispersion relations.

    For massive graviton (alpha=0), returns the effective distance
    appearing in the phase modification.

    Parameters
    ----------
    alpha : float
        Power-law index for modified dispersion relation.
        alpha=0 for massive graviton.
    z : float
        Source redshift.

    Returns
    -------
    float
        Cosmological distance in metres.
    """
    integrand_result = quad(
        lambda z_prime: (1 + z_prime)**(alpha - 2) /
            np.sqrt(_OMEGA_M * (1 + z_prime)**3 + _OMEGA_LAMBDA),
        0, z
    )
    return _C * (1 + z) / _H0 * integrand_result[0]


def _cos_taper(freqs, f_low, f_high):
    """
    Cosine (Hann-like) spectral taper from 1 at f_low to 0 at f_high.

    Returns 1 for f <= f_low, 0 for f >= f_high, and a smooth half-cosine
    transition in between.  Used to suppress Gibbs ringing that would otherwise
    result from abrupt step discontinuities in the frequency-domain phase
    modification before IRFFT.
    """
    result = np.ones_like(freqs, dtype=float)
    mask = (freqs > f_low) & (freqs < f_high)
    xi = (freqs[mask] - f_low) / (f_high - f_low)
    result[mask] = 0.5 * (1.0 + np.cos(np.pi * xi))
    result[freqs >= f_high] = 0.0
    return result


def _apply_end_taper(arr):
    """Cosine-taper the last _RINGDOWN_TAPER_LEN samples of arr to zero.

    The ringdown does not fully decay within the n_ringdown=500 sample window,
    so the highpass FIR filter sees a hard non-zero step at the end of the array
    and produces edge-effect ringing in the post-coalescence region.  Tapering
    the tail to zero before filtering eliminates that discontinuity.
    """
    arr = arr.copy()
    xi = np.linspace(0, 1, _RINGDOWN_TAPER_LEN)
    arr[-_RINGDOWN_TAPER_LEN:] *= 0.5 * (1.0 + np.cos(np.pi * xi))
    return arr


def _additional_phase(freqs, chirp_mass, z, lambda_g, f_pn_cutoff=None):
    """
    Compute massive graviton phase shift for a frequency array.

    Implements delta_Psi = -beta * u^{-1} where beta encodes the graviton
    Compton wavelength and cosmological distance. PN correction terms are
    NOT included -- they are already in IMRPhenomD/IMRPhenomXP.

    Parameters
    ----------
    freqs : np.ndarray
        Frequency array in Hz (must not contain zeros).
    chirp_mass : float
        Chirp mass in solar masses.
    z : float
        Source redshift.
    lambda_g : float
        Graviton Compton wavelength in metres.
    f_pn_cutoff : float or None, optional
        Post-Newtonian breakdown frequency in Hz, from (m1+m2)·omega = 0.1.
        If given, the phase is forced to zero above this frequency — the PN
        dispersion formula is not valid beyond the inspiral regime.
        Default None (no cutoff applied, backward-compatible).

    Returns
    -------
    np.ndarray
        Phase shift in radians, same shape as freqs.
    """
    M = chirp_mass * _M_SUN_SEC * (1 + z)   # chirp mass in seconds
    u = np.pi * M * freqs                    # dimensionless PN parameter
    beta = np.pi**2 * _C * _D_alpha(0, z) * M / (lambda_g**2 * (1 + z))
    delta_psi = -beta * u**(-1)
    if f_pn_cutoff is not None:
        taper = _cos_taper(freqs, f_pn_cutoff * (1.0 - _TAPER_FRACTION), f_pn_cutoff)
        delta_psi = delta_psi * taper
    return delta_psi


def _additional_phase_lv(freqs, chirp_mass, z, lambda_g, alpha_lv, A_lv, f_c,
                          f_pn_cutoff=None):
    """
    Compute the generalized Lorentz-violating (LV) phase shift.

    Implements the parametrized modified dispersion relation of Mirshekari,
    Yunes & Will (2011), arXiv:1110.2720:

        E² = p²c² + m_g²c⁴ + A_physical · p^α · c^α          [Eq. 1]

    where A_physical has units [energy]^{2−α}.  The paper re-packages this
    as the LV Compton wavelength (Eq. 13):

        A  ≡  A_physical^{1/(α−2)}       [always has units of metres]

    The total phase correction (Eq. 28, α ≠ 1, 2) is:

        δΨ = −β u^{−1} − ζ u^{α−1}

    where the massive graviton part (β term) is identical to _additional_phase
    and the Lorentz-violating ζ term is computed from A_lv via Eqs. 30, 32:

        ζ = π^{2−α}/(1−α) · c^{1−α} · D_α · M^{1−α}
                          / (A^{2−α} · (1+Z)^{1−α})     [α ≠ 1, 2]
        ζ = D_1 / A                                       [α = 1, Eq. 32]

    The ppE mapping (Eq. 34):  β_ppE = −ζ,  b_ppE = α − 1.

    Special cases:
        α = 0   : degenerate with massive graviton (A → λ_g)
        α = 1   : logarithmic correction (Eq. 31)
        α = 2   : degenerate with time of coalescence — no observable effect
        α = 2.5 : non-commutative geometry
        α = 3   : doubly special relativity (DSR)
        α = 4   : extra dimensions / Horava-Lifshitz gravity

    Parameters
    ----------
    freqs : np.ndarray
        Frequency array in Hz (must not contain zeros).
    chirp_mass : float
        Chirp mass in solar masses.
    z : float
        Source redshift.
    lambda_g : float
        Graviton Compton wavelength in metres. Use np.inf to suppress the
        massive graviton term.
    alpha_lv : float
        LV dispersion exponent α in the modified dispersion relation.
    A_lv : float
        LV Compton wavelength in metres (A ≡ A_physical^{1/(α−2)}).
        Use np.inf to suppress the LV term (GR + massive-graviton limit).
    f_c : float
        Cutoff frequency in Hz (typically the maximum waveform frequency).
        The LV phase is normalised to zero at f_c. The massive graviton term
        is purely frequency-dependent and does not use f_c.
    f_pn_cutoff : float or None, optional
        Post-Newtonian breakdown frequency in Hz, from (m1+m2)·omega = 0.1.
        If given, the total phase is forced to zero above this frequency — the
        PN dispersion formula is not valid beyond the inspiral regime.
        Default None (no cutoff applied, backward-compatible).

    Returns
    -------
    np.ndarray
        Total phase shift in radians, same shape as freqs.
    """
    M   = chirp_mass * _M_SUN_SEC * (1 + z)   # detector-frame chirp mass [s]
    u   = np.pi * M * freqs                    # dimensionless PN frequency
    u_c = np.pi * M * f_c

    # ── Massive graviton term (same as _additional_phase) ─────────────────────
    if np.isfinite(lambda_g) and lambda_g > 0:
        beta_g = np.pi**2 * _C * _D_alpha(0, z) * M / (lambda_g**2 * (1 + z))
        delta_psi_mg = -beta_g * u**(-1)
    else:
        delta_psi_mg = np.zeros_like(freqs)

    # ── Lorentz-violating term (Mirshekari et al. 2011, Eqs. 28–32) ──────────
    if not np.isfinite(A_lv) or A_lv <= 0 or alpha_lv == 2.0:
        # A_lv = inf  → GR / massive-graviton-only limit
        # alpha_lv = 2 → degenerate with time of coalescence
        delta_psi_lv = np.zeros_like(freqs)
    elif alpha_lv == 1.0:
        # Eq. 32:  ζ = D_1 / A  (both in metres → dimensionless)
        # Eq. 31:  δΨ_LV = +ζ · (ln u − ln u_c)
        zeta = _D_alpha(1, z) / A_lv
        delta_psi_lv = zeta * (np.log(u) - np.log(u_c))
    else:
        # Eq. 30 (SI):  ζ = π^{2−α}/(1−α) · c^{1−α} · D_α · M^{1−α}
        #                             / (A^{2−α} · (1+Z)^{1−α})
        zeta = (
            np.pi**(2 - alpha_lv) / (1 - alpha_lv)
            * _C**(1 - alpha_lv)
            * _D_alpha(alpha_lv, z)
            * M**(1 - alpha_lv)
            / (A_lv**(2 - alpha_lv) * (1 + z)**(1 - alpha_lv))
        )
        # Eq. 28:  δΨ_LV = −ζ · u^{α−1},  normalised to zero at u_c
        delta_psi_lv = -zeta * (u**(alpha_lv - 1) - u_c**(alpha_lv - 1))

    total_phase = delta_psi_mg + delta_psi_lv
    if f_pn_cutoff is not None:
        taper = _cos_taper(freqs, f_pn_cutoff * (1.0 - _TAPER_FRACTION), f_pn_cutoff)
        total_phase = total_phase * taper
    return total_phase


def _m_g_to_lambda_g(m_g_kg):
    """Convert graviton mass m_g [kg] to Compton wavelength λ_g [m] = h/(m_g·c).
    m_g = 0 or non-finite → λ_g = inf (massless/GR limit).
    """
    if not np.isfinite(m_g_kg) or m_g_kg <= 0:
        return np.inf
    return _H_PLANCK / (m_g_kg * _C)


def _A_to_lambda_A(A_eV, alpha_lv):
    """Convert LV dispersion coefficient A [eV^{2-alpha}] to Compton wavelength λ_A [m].
    Implements Mirshekari et al. (2011), Eq. 13: λ_A = ℏc · A^{1/(α-2)}.
    A <= 0, non-finite, or alpha_lv == 2 → λ_A = inf (suppresses LV term).
    """
    if not np.isfinite(A_eV) or A_eV <= 0 or alpha_lv == 2.0:
        return np.inf
    return _HBAR_C_EV_M * float(A_eV) ** (1.0 / (alpha_lv - 2.0))


def _fd_to_td_polarisations(hp_fd_array, hc_fd_array, delta_f, target_length,
                             time_resolution, highpass_fc=_HIGHPASS_FC):
    """Convert FD polarisation arrays to anti-ringing TD TimeSeries.

    Steps: IRFFT + normalise → rearrange so coalescence sits at index
    target_length//2 (t=0, 1 s before and 1 s after at 4096 Hz)
    → cosine-taper tail → highpass FIR filter.
    """
    hp_raw = np.fft.irfft(hp_fd_array)
    hc_raw = np.fft.irfft(hc_fd_array)
    N = len(hp_raw)
    hp_raw *= delta_f * N
    hc_raw *= delta_f * N

    n_half = target_length // 2
    if target_length <= N:
        hp_arr = np.concatenate([hp_raw[N - n_half:], hp_raw[:target_length - n_half]])
        hc_arr = np.concatenate([hc_raw[N - n_half:], hc_raw[:target_length - n_half]])
    else:
        # Waveform shorter than window: centre it, zero-pad both sides
        n_post = min(N, target_length - n_half)
        n_pre  = N - n_post
        hp_arr = np.concatenate([
            np.zeros(n_half - n_pre),
            hp_raw[N - n_pre:] if n_pre else np.zeros(0),
            hp_raw[:n_post],
            np.zeros((target_length - n_half) - n_post),
        ])
        hc_arr = np.concatenate([
            np.zeros(n_half - n_pre),
            hc_raw[N - n_pre:] if n_pre else np.zeros(0),
            hc_raw[:n_post],
            np.zeros((target_length - n_half) - n_post),
        ])

    hp_arr = _apply_end_taper(hp_arr)
    hc_arr = _apply_end_taper(hc_arr)
    hp_ts = TimeSeries(hp_arr.astype(np.float64), delta_t=time_resolution)
    hc_ts = TimeSeries(hc_arr.astype(np.float64), delta_t=time_resolution)
    hp_ts = highpass_fir(hp_ts, highpass_fc, 128)
    hc_ts = highpass_fir(hc_ts, highpass_fc, 128)
    return hp_ts, hc_ts


def _generate_single_lv_waveform(params: dict, time_resolution: float,
                                  approximant: str, f_lower: float,
                                  detectors: list, target_length: int,
                                  add_noise: bool, m_g: float,
                                  alpha_lv: float, A: float,
                                  f_final: float,
                                  noise_backend: str = 'o4_psd',
                                  psd_cache_dir: str = _DEFAULT_CACHE,
                                  highpass_fc: float = _HIGHPASS_FC) -> dict:
    """
    Worker function to generate a single Lorentz-violating (LV) GW waveform.

    Identical pipeline to _generate_single_modified_waveform but applies the
    generalized LV phase from Mirshekari, Yunes & Will (2011), arXiv:1110.2720,
    instead of the pure massive graviton phase from Will (1997), arXiv:9709011.

    Parameters
    ----------
    params : dict
        Waveform parameters. Required: 'mass1', 'mass2'.
        Optional: 'redshift' (default 0.1), plus sky/orientation params.
    time_resolution : float
        Time step delta_t in seconds.
    approximant : str
        FD-capable waveform approximant (e.g. 'IMRPhenomD').
    f_lower : float
        Lower frequency cutoff in Hz.
    detectors : list of str
        Detector names, e.g. ['H1', 'L1'].
    target_length : int
        Desired number of output samples.
    add_noise : bool
        Whether to add aLIGO noise.
    m_g : float
        Graviton mass in kg. Converted to λ_g = h/(m_g·c) internally.
        Use 0 to suppress the mass term (GR limit).
    alpha_lv : float
        LV dispersion exponent (α_LV). Key values: 3 (DSR), 4 (Horava-Lifshitz).
    A : float
        LV dispersion coefficient from Mirshekari et al. (2011) Eq. 1, in units
        [eV]^{2-α}. Converted to λ_A internally via Eq. 13.
        Use np.inf to suppress the LV term.
    f_final : float
        Upper frequency cutoff in Hz.

    Returns
    -------
    dict
        Keys: 'success', 'detectors', 'params', optionally 'error'.
    """
    try:
        delta_f = 1.0 / 256

        hp_fd, hc_fd = get_fd_waveform(
            approximant=approximant,
            mass1=params['mass1'],
            mass2=params['mass2'],
            spin1z=params.get('spin1z', 0.0),
            spin2z=params.get('spin2z', 0.0),
            inclination=params.get('inclination', 0.0),
            coa_phase=params.get('coa_phase', 0.0),
            distance=params.get('distance', 410.0),
            delta_f=delta_f,
            f_lower=f_lower,
            f_final=f_final
        )

        m1 = params['mass1']
        m2 = params['mass2']
        chirp_mass = (m1 * m2)**(3.0 / 5.0) / (m1 + m2)**(1.0 / 5.0)
        z = _redshift_from_distance(params.get('distance', 410.0))

        freqs = hp_fd.sample_frequencies.numpy()[1:]
        hp_fd_amp = np.abs(hp_fd.numpy()[1:])
        f_c = float(np.max(freqs[np.nonzero(hp_fd_amp)]))

        # Per-sample overrides — convert m_g [kg] → λ_g [m]; A [eV^{2-α}] → λ_A [m]
        m_g_val = params.get('m_g', m_g)
        lg      = _m_g_to_lambda_g(m_g_val)
        A_val   = params.get('A', A)
        a_lv    = _A_to_lambda_A(A_val, alpha_lv)

        # PN breakdown frequency: phase modification is only valid in the inspiral
        f_pn = _M_OMEGA_PN / (np.pi * (m1 + m2) * _M_SUN_SEC)

        phase_shift = _additional_phase_lv(freqs, chirp_mass, z, lg,
                                            alpha_lv, a_lv, f_c,
                                            f_pn_cutoff=f_pn)

        hp_array = hp_fd.numpy().copy()
        hc_array = hc_fd.numpy().copy()
        hp_array[1:] = hp_array[1:] * np.exp(1j * phase_shift)
        hc_array[1:] = hc_array[1:] * np.exp(1j * phase_shift)

        hp_ts, hc_ts = _fd_to_td_polarisations(
            hp_array, hc_array, delta_f, target_length, time_resolution, highpass_fc)

        gps_time = params.get('gps_time', 1126259462.4)
        hp_ts.start_time += gps_time
        hc_ts.start_time += gps_time

        ra = params.get('ra', 0.0)
        dec = params.get('dec', np.pi / 2)
        polarization = params.get('polarization', 0.0)

        detector_signals = {}
        for det_name in detectors:
            detector = Detector(det_name)
            signal = detector.project_wave(hp_ts, hc_ts, ra, dec, polarization, method='lal')

            sig_array = signal.numpy()
            signal_len = len(sig_array)
            if signal_len > target_length:
                sig_array = sig_array[signal_len - target_length:]
            elif signal_len < target_length:
                sig_array = np.concatenate([
                    np.zeros(target_length - signal_len, dtype=sig_array.dtype),
                    sig_array
                ])

            detector_signals[det_name] = TimeSeries(
                sig_array, delta_t=signal.delta_t, epoch=signal.start_time)

        if add_noise:
            for det_name in detectors:
                signal = detector_signals[det_name]
                delta_t = signal.delta_t
                duration = target_length * delta_t
                delta_f_noise = 1.0 / duration
                flen = target_length // 2 + 1
                # aligo: analytic PSD — for debugging/quick tests only, not realistic noise
                if noise_backend == 'aligo':
                    psd = aLIGOZeroDetHighPower(flen, delta_f_noise, f_lower)
                else:
                    psd = load_random_o4_psd(flen, delta_f_noise, f_lower, det_name, int(round(1.0 / delta_t)), cache_dir=psd_cache_dir)
                noise = noise_from_psd(target_length, delta_t, psd)
                noise._epoch = signal._epoch
                detector_signals[det_name] = signal.inject(noise)

        return {
            'success': True,
            'detectors': detector_signals,
            'params': params
        }

    except Exception as e:
        return {'success': False, 'error': str(e), 'params': params}


################### Single-waveform utilities ###################

def normalize_waveform(waveform, scale_factor=1e21):
    """
    Normalize a single waveform by multiplying by a fixed scale factor.

    This scales very small gravitational wave strain values (typically ~10^-21)
    to order 1-10 range for easier processing and visualization.

    Parameters
    ----------
    waveform : np.ndarray or TimeSeries
        Single waveform to normalize (1D array)
    scale_factor : float, optional
        Fixed scaling factor to multiply the waveform by.
        Default: 1e21 (appropriate for typical GW strains of ~1e-21)

    Returns
    -------
    normalized : np.ndarray
        Scaled waveform as numpy array
    """
    if isinstance(waveform, TimeSeries):
        data = waveform.numpy()
    else:
        data = np.asarray(waveform)

    if data.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {data.shape}")

    normalized = data * scale_factor

    return normalized


def whiten_waveform(waveform, delta_t=1/4096, f_lower=20.0, apply_bandpass=True,
                    apply_tukey=True, tukey_alpha=0.1, tukey_side='both',
                    psd=None):
    """
    Whiten a single waveform using PSD-based whitening.

    Follows the PyCBC GW150914 tutorial approach. Pipeline order is
    ``window → whiten → bandpass``:

    1. Estimate PSD on the UNWINDOWED strain (Welch's per-segment Hann handles
       leakage internally; pre-windowing here would scale the PSD down and
       bias the whitening amplitude). Skipped if ``psd=`` is supplied.
    2. Apply the Tukey window to the strain for the forward FFT so that abrupt
       edges don't leak Gibbs ringing through the sqrt(PSD) division.
    3. Divide frequency-domain data by sqrt(PSD).
    4. Convert back to time domain.
    5. Apply bandpass filter (f_lower – 300 Hz) if ``apply_bandpass=True``.

    Parameters
    ----------
    waveform : np.ndarray or TimeSeries
        Single waveform to whiten (1D array)
    delta_t : float, optional
        Time resolution in seconds (default: 1/4096)
    f_lower : float, optional
        Lower frequency cutoff in Hz (default: 20.0)
    apply_bandpass : bool, optional
        Apply bandpass filter 35-300 Hz (default: True)
    apply_tukey : bool, optional
        Apply Tukey window before whitening to prevent edge effects (default: True)
    tukey_alpha : float, optional
        Tukey window alpha parameter - fraction of signal to taper (default: 0.1)
    tukey_side : str, optional
        Which side(s) to apply the Tukey taper (default: 'both')
        - 'both': Taper both sides — correct when merger is centred at t=0
        - 'left': Only taper the beginning
        - 'right': Only taper the end
    psd : pycbc.types.FrequencySeries, optional
        Pre-computed PSD to use directly, bypassing PSD estimation.
        Useful for simulated data where the generation PSD is known exactly.

    Returns
    -------
    whitened : np.ndarray
        Whitened waveform as numpy array (same length as input)
    psd : np.ndarray
        Power spectral density used for whitening
    freqs : np.ndarray
        Frequency array for PSD
    """
    from scipy.signal.windows import tukey
    if isinstance(waveform, TimeSeries):
        data = waveform.numpy()
    else:
        data = np.asarray(waveform)
        if data.ndim != 1:
            raise ValueError(f"Expected 1D array, got shape {data.shape}")

    # Pipeline order: window → whiten → bandpass.
    # Subtlety: the PSD is estimated from the UNWINDOWED strain (Welch applies
    # its own per-segment Hann internally, so it is already leakage-safe), but
    # the forward FFT for whitening must see a WINDOWED strain so that abrupt
    # edges do not leak a broadband Gibbs ringing through the division by
    # sqrt(PSD) that no post-filter can remove.
    strain_raw = TimeSeries(data, delta_t=delta_t)

    n_samples = len(strain_raw)
    if psd is not None:
        pass  # use caller-supplied PSD as-is
    elif n_samples >= 8192:
        seg_len = 4096
        seg_stride = 2048
        psd_welch = welch(strain_raw, seg_len=seg_len, seg_stride=seg_stride)
        psd = interpolate(psd_welch, 1.0 / strain_raw.duration)
    else:
        delta_f = 1.0 / strain_raw.duration
        flen = n_samples // 2 + 1
        psd = aLIGOZeroDetHighPower(flen, delta_f, f_lower)

    if apply_tukey:
        n = len(data)
        if tukey_side == 'both':
            window = tukey(n, alpha=tukey_alpha)
        elif tukey_side in ('left', 'right'):
            full_window = tukey(n, alpha=tukey_alpha * 2)
            window = np.ones(n)
            taper_len = int(n * tukey_alpha)
            if tukey_side == 'left':
                window[:taper_len] = full_window[:taper_len]
            else:
                window[-taper_len:] = full_window[-taper_len:]
        else:
            raise ValueError(f"tukey_side must be 'left', 'right', or 'both', got '{tukey_side}'")
        strain = TimeSeries(data * window, delta_t=delta_t)
    else:
        strain = strain_raw

    freq_series = strain.to_frequencyseries()

    psd.resize(len(freq_series))

    psd_safe = psd.copy()
    psd_array = psd_safe.numpy()
    epsilon = 1e-40
    psd_array[psd_array <= 0] = epsilon
    psd_safe = FrequencySeries(psd_array, delta_f=psd.delta_f, epoch=psd.epoch)

    white_strain = (freq_series / (psd_safe ** 0.5)).to_timeseries()

    if apply_bandpass:
        white_strain = highpass_fir(white_strain, f_lower, 128)
        white_strain = lowpass_fir(white_strain, 300, 128)

    whitened = white_strain.numpy()
    psd_array = psd.numpy()
    freqs = np.arange(len(psd)) * psd.delta_f

    return whitened, psd_array, freqs


def process_waveform(
    waveform,
    detector: str = "H1",
    delta_t: float = 1 / 4096,
    f_lower: float = 10.0,
    highpass_fc: float = 35.0,
    tukey_alpha: float = 0.1,
    apply_bandpass: bool = True,
    sample_rate: int = 4096,
    cache_dir: str = _DEFAULT_CACHE,
) -> np.ndarray:
    """Apply the full processing pipeline to a single waveform.

    Combines Tukey windowing, O4 PSD loading, and whitening into one call.
    Pipeline: window → whiten (O4 cached PSD) → bandpass.

    Parameters
    ----------
    waveform : np.ndarray
        Raw 1-D strain array.
    detector : str
        Detector name used to look up the O4 PSD cache (e.g. ``'H1'``).
    delta_t : float
        Sample spacing in seconds.
    f_lower : float
        Lower frequency for the O4 PSD floor during noise injection and PSD
        loading.  Kept separate from ``highpass_fc`` so the PSD covers low
        frequencies without driving the post-whitening highpass too low.
    highpass_fc : float
        Highpass cutoff applied after whitening (default 35 Hz).  A FIR
        highpass at very low frequencies needs many more taps than the 128
        used here, so keeping this at ≥35 Hz avoids edge ringing.
    tukey_alpha : float
        Fraction of the signal to taper at each end.
    apply_bandpass : bool
        Apply highpass (at ``highpass_fc``) + lowpass 300 Hz after whitening.
    sample_rate : int
        Sample rate in Hz — must match the cached PSD.
    cache_dir : str
        Directory containing the O4 PSD ``.npz`` cache files.

    Returns
    -------
    np.ndarray
        Processed waveform (same length as input).
    """
    from scipy.signal.windows import tukey
    n = len(waveform)
    flen = n // 2 + 1
    delta_f = 1.0 / (n * delta_t)
    psd = load_random_o4_psd(flen, delta_f, f_lower, detector, sample_rate, cache_dir=cache_dir)
    window = tukey(n, alpha=tukey_alpha)
    whitened, _, _ = whiten_waveform(
        waveform * window,
        delta_t=delta_t,
        f_lower=highpass_fc,
        apply_bandpass=apply_bandpass,
        apply_tukey=False,
        psd=psd,
    )
    return whitened


def resample_waveform(waveform, original_delta_t, target_delta_t,
                      apply_tukey=True, tukey_alpha=0.1, tukey_side='both'):
    """
    Resample a single waveform to a different sampling rate.

    Uses PyCBC's resample_to_delta_t function which applies proper
    anti-aliasing filtering.

    Parameters
    ----------
    waveform : np.ndarray or TimeSeries
        Single waveform to resample (1D array)
    original_delta_t : float
        Original time resolution in seconds
    target_delta_t : float
        Target time resolution in seconds
    apply_tukey : bool, optional
        Apply Tukey window before resampling to prevent edge effects (default: True)
    tukey_alpha : float, optional
        Tukey window alpha parameter - fraction of signal to taper (default: 0.1)
    tukey_side : str, optional
        Which side(s) to apply the Tukey taper (default: 'both')

    Returns
    -------
    resampled : np.ndarray
        Resampled waveform as numpy array
    """
    from scipy.signal.windows import tukey

    if isinstance(waveform, TimeSeries):
        data = waveform.numpy()
    else:
        data = np.asarray(waveform)
        if data.ndim != 1:
            raise ValueError(f"Expected 1D array, got shape {data.shape}")

    if apply_tukey:
        n = len(data)
        if tukey_side == 'both':
            window = tukey(n, alpha=tukey_alpha)
        elif tukey_side == 'left':
            full_window = tukey(n, alpha=tukey_alpha * 2)
            window = np.ones(n)
            taper_len = int(n * tukey_alpha)
            window[:taper_len] = full_window[:taper_len]
        elif tukey_side == 'right':
            full_window = tukey(n, alpha=tukey_alpha * 2)
            window = np.ones(n)
            taper_len = int(n * tukey_alpha)
            window[-taper_len:] = full_window[-taper_len:]
        else:
            raise ValueError(f"tukey_side must be 'left', 'right', or 'both', got '{tukey_side}'")
        data = data * window

    strain = TimeSeries(data, delta_t=original_delta_t)
    resampled_strain = resample_to_delta_t(strain, target_delta_t)
    resampled = np.array(resampled_strain)

    return resampled


################### Batch normalisation helpers ###################

def _chirp_mass(m1, m2):
    return (m1 * m2) ** (3.0 / 5.0) / (m1 + m2) ** (1.0 / 5.0)


def normalize_waveforms(signal_array: np.ndarray, method: str = 'global_standardize') -> Tuple[np.ndarray, List[float]]:
    """
    Normalize waveform data using various strategies.

    Parameters
    ----------
    signal_array : np.ndarray
        Shape: (num_samples, num_detectors, time_steps)
    method : str
        Normalization method:
        - 'per_sample_minmax': Per-sample min-max to [0, 100]
        - 'global_standardize': Global z-score (mean=0, std=1) across all data (RECOMMENDED)
        - 'per_sample_standardize': Per-sample z-score (preserves relative structure)
        - 'global_minmax': Global min-max to [0, 100] (preserves relative amplitudes)
        - 'scale_constant': Divide by scale (preserves all relationships, simple)
        - 'none': No normalization

    Returns
    -------
    normalized_array : np.ndarray
        Normalized waveforms
    amplitude_stats : list
        Average amplitude per sample (for diagnostics)
    """
    num_samples = signal_array.shape[0]
    amplitude_stats = []

    if method == 'per_sample_minmax':
        print(f"  Normalizing: per-sample min-max to [0, 100]")
        for i in range(num_samples):
            sample_min = signal_array[i].min()
            sample_max = signal_array[i].max()
            if sample_max > sample_min:
                signal_array[i] = (signal_array[i] - sample_min) / (sample_max - sample_min) * 100
            else:
                signal_array[i] = 50.0
            amplitude_stats.append(signal_array[i].mean())

    elif method == 'global_standardize':
        print(f"  Normalizing: global z-score (mean=0, std=1)")
        global_mean = signal_array.mean()
        global_std = signal_array.std()
        if global_std > 0:
            signal_array = (signal_array - global_mean) / global_std
        for i in range(num_samples):
            amplitude_stats.append(signal_array[i].std())
        print(f"  Global stats: mean={global_mean:.2e}, std={global_std:.2e}")

    elif method == 'per_sample_standardize':
        print(f"  Normalizing: per-sample z-score (mean=0, std=1)")
        for i in range(num_samples):
            sample_mean = signal_array[i].mean()
            sample_std = signal_array[i].std()
            if sample_std > 0:
                signal_array[i] = (signal_array[i] - sample_mean) / sample_std
            else:
                signal_array[i] = 0.0
            amplitude_stats.append(sample_std)

    elif method == 'global_minmax':
        print(f"  Normalizing: global min-max to [0, 100]")
        global_min = signal_array.min()
        global_max = signal_array.max()
        if global_max > global_min:
            signal_array = (signal_array - global_min) / (global_max - global_min) * 100
        else:
            signal_array = 50.0
        for i in range(num_samples):
            amplitude_stats.append(signal_array[i].mean())
        print(f"  Global range: [{global_min:.2e}, {global_max:.2e}]")

    elif method == 'scale_constant':
        print(f"  Normalizing: constant scaling by 1e-21")
        scale_factor = 1e-21
        signal_array = signal_array / scale_factor * 10
        for i in range(num_samples):
            amplitude_stats.append(np.abs(signal_array[i]).max())
        print(f"  Scale factor: {scale_factor:.2e}")

    elif method == 'none':
        print(f"  No waveform normalization applied")
        for i in range(num_samples):
            amplitude_stats.append(signal_array[i].mean())

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    if amplitude_stats:
        print(f"  Amplitude stats: mean={np.mean(amplitude_stats):.2f} ± {np.std(amplitude_stats):.2f}")

    return signal_array, amplitude_stats


def normalize_parameters(param_array: np.ndarray, param_names: List[str], method: str = 'zscore') -> Tuple[np.ndarray, Dict]:
    """
    Normalize parameter data.

    Parameters
    ----------
    param_array : np.ndarray
        Shape: (num_samples, num_params)
    param_names : list of str
        Parameter names
    method : str
        Normalization method:
        - 'zscore': Z-score normalization (mean=0, std=1) — RECOMMENDED for flows
        - 'minmax': Min-max to [-1, 1]
        - 'none': No normalization

    Returns
    -------
    normalized_array : np.ndarray
        Normalized parameters
    norm_info : dict
        Normalization statistics for each parameter
    """
    num_params = param_array.shape[1]
    param_means = param_array.mean(axis=0)
    param_stds = param_array.std(axis=0)
    param_mins = param_array.min(axis=0)
    param_maxs = param_array.max(axis=0)

    param_norm_info = {}
    for j, name in enumerate(param_names):
        param_norm_info[name] = {
            'mean': float(param_means[j]),
            'std': float(param_stds[j]),
            'min': float(param_mins[j]),
            'max': float(param_maxs[j]),
            'method': method
        }

    if method == 'zscore':
        print(f"  Normalizing parameters: z-score (mean=0, std=1)")
        for j in range(num_params):
            if param_stds[j] > 0:
                param_array[:, j] = (param_array[:, j] - param_means[j]) / param_stds[j]
            else:
                param_array[:, j] = 0
        print(f"  Parameters normalized:")
        for name, info in param_norm_info.items():
            print(f"    {name}: mean={info['mean']:.4f}, std={info['std']:.4f}")

    elif method == 'minmax':
        print(f"  Normalizing parameters: min-max to [-1, 1]")
        for j in range(num_params):
            if param_maxs[j] > param_mins[j]:
                param_array[:, j] = 2 * (param_array[:, j] - param_mins[j]) / (param_maxs[j] - param_mins[j]) - 1
            else:
                param_array[:, j] = 0
        print(f"  Parameters normalized:")
        for name, info in param_norm_info.items():
            print(f"    {name}: [{info['min']:.4f}, {info['max']:.4f}]")

    elif method == 'none':
        print(f"  No parameter normalization applied")

    else:
        raise ValueError(f"Unknown parameter normalization method: {method}")

    return param_array, param_norm_info


################### Config / parameter helpers ###################

def _validate_config(config: Dict[str, Callable]) -> None:
    """Validate configuration dictionary."""
    if not isinstance(config, dict):
        raise TypeError("config must be a dictionary")

    if not config:
        raise ValueError("config cannot be empty")

    for key, value in config.items():
        if not callable(value):
            raise TypeError(f"config['{key}'] must be callable (e.g., a distribution function)")

        try:
            test = value(size=2)
            if not isinstance(test, np.ndarray):
                raise TypeError(f"config['{key}']() must return numpy array")
        except Exception as e:
            raise ValueError(f"config['{key}'] failed test call: {e}")


def _generate_parameter_sets(config: Dict[str, Callable], num_samples: int) -> List[Dict]:
    """Generate all parameter combinations upfront."""
    param_arrays = {}
    for param_name, dist_func in config.items():
        param_arrays[param_name] = dist_func(size=num_samples)

    param_dicts = []
    for i in range(num_samples):
        param_dict = {name: float(values[i]) for name, values in param_arrays.items()}
        param_dicts.append(param_dict)

    return param_dicts


################### GR baseline waveform workers ###################

def _generate_single_waveform(params: Dict, time_resolution: float, approximant: str,
                              f_lower: float, detectors: List[str], target_length: int,
                              add_noise: bool = True, f_final: float = 2048.0,
                              noise_backend: str = 'o4_psd',
                              psd_cache_dir: str = _DEFAULT_CACHE,
                              highpass_fc: float = _HIGHPASS_FC) -> Dict:
    """Worker function to generate a single waveform and project to detectors at fixed length.

    Uses the FD→IRFFT pipeline (same as the modified-gravity workers) so that GR
    and modified-gravity datasets are generated with identical approximant, frequency
    range, and time-domain assembly: the last (target_length - n_ringdown) samples of
    the IRFFT array (late inspiral) are prepended to the first n_ringdown samples
    (merger + early ringdown), giving a physically motivated signal window.
    """
    try:
        delta_f = 1.0 / 256

        hp_fd, hc_fd = get_fd_waveform(
            approximant=approximant,
            mass1=params['mass1'],
            mass2=params['mass2'],
            spin1z=params.get('spin1z', 0.0),
            spin2z=params.get('spin2z', 0.0),
            inclination=params.get('inclination', 0.0),
            coa_phase=params.get('coa_phase', 0.0),
            distance=params.get('distance', 410.0),
            delta_f=delta_f,
            f_lower=f_lower,
            f_final=f_final,
        )

        hp_raw = np.fft.irfft(hp_fd.numpy())
        hc_raw = np.fft.irfft(hc_fd.numpy())
        N = len(hp_raw)
        hp_raw *= delta_f * N
        hc_raw *= delta_f * N

        n_half = target_length // 2
        if target_length <= N:
            hp_arr = np.concatenate([hp_raw[N - n_half:], hp_raw[:target_length - n_half]])
            hc_arr = np.concatenate([hc_raw[N - n_half:], hc_raw[:target_length - n_half]])
        else:
            n_post = min(N, target_length - n_half)
            n_pre  = N - n_post
            hp_arr = np.concatenate([
                np.zeros(n_half - n_pre),
                hp_raw[N - n_pre:] if n_pre else np.zeros(0),
                hp_raw[:n_post],
                np.zeros((target_length - n_half) - n_post),
            ])
            hc_arr = np.concatenate([
                np.zeros(n_half - n_pre),
                hc_raw[N - n_pre:] if n_pre else np.zeros(0),
                hc_raw[:n_post],
                np.zeros((target_length - n_half) - n_post),
            ])

        hp_arr = _apply_end_taper(hp_arr)
        hc_arr = _apply_end_taper(hc_arr)
        hp_ts = TimeSeries(hp_arr.astype(np.float64), delta_t=time_resolution)
        hc_ts = TimeSeries(hc_arr.astype(np.float64), delta_t=time_resolution)
        hp_ts = highpass_fir(hp_ts, highpass_fc, 128)
        hc_ts = highpass_fir(hc_ts, highpass_fc, 128)

        gps_time = params.get('gps_time', 1126259462.4)
        hp_ts.start_time += gps_time
        hc_ts.start_time += gps_time

        ra = params.get('ra', 0.0)
        dec = params.get('dec', np.pi / 2)
        polarization = params.get('polarization', 0.0)

        detector_signals = {}
        for det_name in detectors:
            detector = Detector(det_name)
            signal = detector.project_wave(hp_ts, hc_ts, ra, dec, polarization, method='lal')

            sig_array = signal.numpy()
            signal_len = len(sig_array)
            if signal_len >= target_length:
                sig_array = sig_array[signal_len - target_length:]
            else:
                sig_array = np.concatenate([
                    np.zeros(target_length - signal_len, dtype=sig_array.dtype),
                    sig_array
                ])

            detector_signals[det_name] = TimeSeries(
                sig_array, delta_t=signal.delta_t, epoch=signal.start_time)

        if add_noise:
            for det_name in detectors:
                signal = detector_signals[det_name]

                delta_t = signal.delta_t
                duration = target_length * delta_t
                delta_f = 1.0 / duration

                flen = target_length // 2 + 1

                # aligo: analytic PSD — for debugging/quick tests only, not realistic noise
                if noise_backend == 'aligo':
                    psd = aLIGOZeroDetHighPower(flen, delta_f, f_lower)
                else:
                    psd = load_random_o4_psd(flen, delta_f, f_lower, det_name, int(round(1.0 / delta_t)), cache_dir=psd_cache_dir)
                noise = noise_from_psd(target_length, delta_t, psd)
                noise._epoch = signal._epoch
                detector_signals[det_name] = signal.inject(noise)

        result = {
            'success': True,
            'detectors': detector_signals,
            'params': params
        }

        return result

    except Exception as e:
        return {'success': False, 'error': str(e), 'params': params}


def _generate_waveforms_parallel(param_dicts: List[Dict],
                                time_resolution: float,
                                approximant: str,
                                f_lower: float,
                                num_workers: int,
                                show_progress: bool,
                                detectors: List[str],
                                target_length: int,
                                add_noise: bool,
                                f_final: float = 2048.0,
                                noise_backend: str = 'o4_psd',
                                psd_cache_dir: str = _DEFAULT_CACHE,
                                highpass_fc: float = _HIGHPASS_FC) -> List[Dict]:
    """Generate waveforms in parallel using multiprocessing."""
    worker_func = partial(_generate_single_waveform,
                          time_resolution=time_resolution,
                          approximant=approximant,
                          f_lower=f_lower,
                          detectors=detectors,
                          target_length=target_length,
                          add_noise=add_noise,
                          f_final=f_final,
                          noise_backend=noise_backend,
                          psd_cache_dir=psd_cache_dir,
                          highpass_fc=highpass_fc)

    if num_workers == 1:
        iterable = map(worker_func, param_dicts)
        if show_progress:
            results = list(tqdm(iterable, total=len(param_dicts), desc="Generating waveforms"))
        else:
            results = list(iterable)
    else:
        with Pool(processes=num_workers) as pool:
            if show_progress:
                results = list(tqdm(
                    pool.imap_unordered(worker_func, param_dicts, chunksize=100),
                    total=len(param_dicts),
                    desc="Generating waveforms"
                ))
            else:
                results = list(pool.imap_unordered(worker_func, param_dicts, chunksize=100))

    return results


################### Modified (massive graviton) waveform workers ###################

def _generate_single_modified_waveform(params: Dict, time_resolution: float,
                                        approximant: str, f_lower: float,
                                        detectors: List[str], target_length: int,
                                        add_noise: bool, m_g: float,
                                        f_final: float,
                                        noise_backend: str = 'o4_psd',
                                        psd_cache_dir: str = _DEFAULT_CACHE,
                                        highpass_fc: float = _HIGHPASS_FC) -> Dict:
    """
    Worker function to generate a single modified (massive graviton) waveform.

    Generates a frequency-domain waveform, applies the massive graviton phase
    shift, IFFTs to time domain, projects onto detectors, and optionally adds noise.

    Parameters
    ----------
    params : dict
        Waveform parameters. Required: 'mass1', 'mass2'.
        Optional: 'redshift' (default 0.1), plus all standard sky/orientation params.
    time_resolution : float
        Time step delta_t in seconds.
    approximant : str
        FD-capable waveform approximant (e.g. 'IMRPhenomD').
    f_lower : float
        Lower frequency cutoff in Hz.
    detectors : list of str
        Detector names, e.g. ['H1', 'L1'].
    target_length : int
        Desired number of output samples.
    add_noise : bool
        Whether to add aLIGO noise.
    m_g : float
        Graviton mass in kg. Converted to λ_g = h/(m_g·c) internally.
        Use 0 to suppress the mass term (GR limit).
    f_final : float
        Upper frequency cutoff for FD waveform generation.

    Returns
    -------
    dict
        Keys: 'success', 'detectors', 'params', optionally 'error'.
    """
    try:
        delta_f = 1.0 / 256

        hp_fd, hc_fd = get_fd_waveform(
            approximant=approximant,
            mass1=params['mass1'],
            mass2=params['mass2'],
            spin1z=params.get('spin1z', 0.0),
            spin2z=params.get('spin2z', 0.0),
            inclination=params.get('inclination', 0.0),
            coa_phase=params.get('coa_phase', 0.0),
            distance=params.get('distance', 410.0),
            delta_f=delta_f,
            f_lower=f_lower,
            f_final=f_final
        )

        m1 = params['mass1']
        m2 = params['mass2']
        chirp_mass = (m1 * m2)**(3.0 / 5.0) / (m1 + m2)**(1.0 / 5.0)
        z = _redshift_from_distance(params.get('distance', 410.0))

        freqs = hp_fd.sample_frequencies.numpy()[1:]
        m_g_val = params.get('m_g', m_g)
        lg = _m_g_to_lambda_g(m_g_val)

        f_pn = _M_OMEGA_PN / (np.pi * (m1 + m2) * _M_SUN_SEC)

        phase_shift = _additional_phase(freqs, chirp_mass, z, lg,
                                        f_pn_cutoff=f_pn)

        hp_array = hp_fd.numpy().copy()
        hc_array = hc_fd.numpy().copy()
        hp_array[1:] = hp_array[1:] * np.exp(1j * phase_shift)
        hc_array[1:] = hc_array[1:] * np.exp(1j * phase_shift)

        hp_ts, hc_ts = _fd_to_td_polarisations(
            hp_array, hc_array, delta_f, target_length, time_resolution, highpass_fc)

        gps_time = params.get('gps_time', 1126259462.4)
        hp_ts.start_time += gps_time
        hc_ts.start_time += gps_time

        ra = params.get('ra', 0.0)
        dec = params.get('dec', np.pi / 2)
        polarization = params.get('polarization', 0.0)

        detector_signals = {}
        for det_name in detectors:
            detector = Detector(det_name)
            signal = detector.project_wave(hp_ts, hc_ts, ra, dec, polarization, method='lal')

            sig_array = signal.numpy()
            signal_len = len(sig_array)
            if signal_len > target_length:
                sig_array = sig_array[signal_len - target_length:]
            elif signal_len < target_length:
                sig_array = np.concatenate([np.zeros(target_length - signal_len, dtype=sig_array.dtype), sig_array])

            detector_signals[det_name] = TimeSeries(
                sig_array, delta_t=signal.delta_t, epoch=signal.start_time)

        if add_noise:
            for det_name in detectors:
                signal = detector_signals[det_name]
                delta_t = signal.delta_t
                duration = target_length * delta_t
                delta_f_noise = 1.0 / duration
                flen = target_length // 2 + 1
                # aligo: analytic PSD — for debugging/quick tests only, not realistic noise
                if noise_backend == 'aligo':
                    psd = aLIGOZeroDetHighPower(flen, delta_f_noise, f_lower)
                else:
                    psd = load_random_o4_psd(flen, delta_f_noise, f_lower, det_name, int(round(1.0 / delta_t)), cache_dir=psd_cache_dir)
                noise = noise_from_psd(target_length, delta_t, psd)
                noise._epoch = signal._epoch
                detector_signals[det_name] = signal.inject(noise)

        return {
            'success': True,
            'detectors': detector_signals,
            'params': params
        }

    except Exception as e:
        return {'success': False, 'error': str(e), 'params': params}


def _generate_modified_waveforms_parallel(param_dicts: List[Dict],
                                           time_resolution: float,
                                           approximant: str,
                                           f_lower: float,
                                           num_workers: int,
                                           show_progress: bool,
                                           detectors: List[str],
                                           target_length: int,
                                           add_noise: bool,
                                           m_g: float,
                                           f_final: float,
                                           noise_backend: str = 'o4_psd',
                                           psd_cache_dir: str = _DEFAULT_CACHE,
                                           highpass_fc: float = _HIGHPASS_FC) -> List[Dict]:
    """Generate modified waveforms in parallel using multiprocessing."""
    worker_func = partial(
        _generate_single_modified_waveform,
        time_resolution=time_resolution,
        approximant=approximant,
        f_lower=f_lower,
        detectors=detectors,
        target_length=target_length,
        add_noise=add_noise,
        m_g=m_g,
        f_final=f_final,
        noise_backend=noise_backend,
        psd_cache_dir=psd_cache_dir,
        highpass_fc=highpass_fc,
    )

    if num_workers == 1:
        iterable = map(worker_func, param_dicts)
        if show_progress:
            results = list(tqdm(iterable, total=len(param_dicts), desc="Generating modified waveforms"))
        else:
            results = list(iterable)
    else:
        with Pool(processes=num_workers) as pool:
            if show_progress:
                results = list(tqdm(
                    pool.imap_unordered(worker_func, param_dicts, chunksize=100),
                    total=len(param_dicts),
                    desc="Generating modified waveforms"
                ))
            else:
                results = list(pool.imap_unordered(worker_func, param_dicts, chunksize=100))

    return results


def _generate_lv_waveforms_parallel(param_dicts: List[Dict],
                                     time_resolution: float,
                                     approximant: str,
                                     f_lower: float,
                                     num_workers: int,
                                     show_progress: bool,
                                     detectors: List[str],
                                     target_length: int,
                                     add_noise: bool,
                                     m_g: float,
                                     alpha_lv: float,
                                     A: float,
                                     f_final: float,
                                     noise_backend: str = 'o4_psd',
                                     psd_cache_dir: str = _DEFAULT_CACHE,
                                     highpass_fc: float = _HIGHPASS_FC) -> List[Dict]:
    """Generate Lorentz-violating waveforms in parallel using multiprocessing."""
    worker_func = partial(
        _generate_single_lv_waveform,
        time_resolution=time_resolution,
        approximant=approximant,
        f_lower=f_lower,
        detectors=detectors,
        target_length=target_length,
        add_noise=add_noise,
        m_g=m_g,
        alpha_lv=alpha_lv,
        A=A,
        f_final=f_final,
        noise_backend=noise_backend,
        psd_cache_dir=psd_cache_dir,
        highpass_fc=highpass_fc,
    )

    if num_workers == 1:
        iterable = map(worker_func, param_dicts)
        if show_progress:
            results = list(tqdm(iterable, total=len(param_dicts), desc="Generating LV waveforms"))
        else:
            results = list(iterable)
    else:
        with Pool(processes=num_workers) as pool:
            if show_progress:
                results = list(tqdm(
                    pool.imap_unordered(worker_func, param_dicts, chunksize=100),
                    total=len(param_dicts),
                    desc="Generating LV waveforms"
                ))
            else:
                results = list(pool.imap_unordered(worker_func, param_dicts, chunksize=100))

    return results


################### Top-level data generators ###################

def pycbc_data_generator(config: Dict[str, Callable],
                        num_samples: int,
                        time_resolution: float = 1/4096,
                        approximant: str = 'IMRPhenomD',
                        f_lower: float = 10.0,
                        f_final: float = 2048.0,
                        highpass_fc: float = _HIGHPASS_FC,
                        num_workers: int = None,
                        signal_length: float = 2.0,
                        batch_size: int = 256,
                        chunk_size: int = 10000,
                        train_split: float = 0.8,
                        val_split: float = 0.1,
                        show_progress: bool = True,
                        detectors: List[str] = None,
                        add_noise: bool = True,
                        noise_backend: str = 'o4_psd',
                        psd_cache_dir: str = _DEFAULT_CACHE) -> Dict:
    """
    Generate PyCBC waveforms projected to detectors.
    Returns PyTorch DataLoaders for training, validation, and testing.

    Parameters
    ----------
    config : dict
        Dictionary mapping parameter names to numpy distribution functions.

        Required parameters:
        - 'mass1': Primary mass (solar masses)
        - 'mass2': Secondary mass (solar masses)

        Optional parameters:
        - 'distance': Luminosity distance in Mpc - default: 410.0 (GW150914)
        - 'spin1z', 'spin2z': Spin components - default: 0.0
        - 'inclination', 'coa_phase': Orientation angles - default: 0.0
        - 'ra': Right ascension (radians) - default: 0.0
        - 'dec': Declination (radians) - default: π/2 (north pole)
        - 'polarization': Polarization angle (radians) - default: 0.0
        - 'gps_time': GPS time of merger (seconds) - default: 1126259462.4 (GW150914)
        - 'tc': Coalescence time - default: 0.0

    num_samples : int
        Total number of waveforms to generate
    time_resolution : float
        Time step (delta_t). Default: 1/4096
    approximant : str
        Waveform approximant. Default: 'IMRPhenomXP'
    f_lower : float
        Lower frequency cutoff (Hz). Default: 40.0
    num_workers : int
        Parallel processes. Default: 1
    batch_size : int
        DataLoader batch size. Default: 256
    chunk_size : int
        Process in chunks for memory. Default: 10000
    signal_length : float
        Length in seconds that each signal should be. Default: 2
    train_split : float
        Training fraction. Default: 0.8
    val_split : float
        Validation fraction. Default: 0.1
    detectors : list of str
        Detector names. Default: ['H1', 'L1']
    add_noise : bool
        Whether to add detector noise to signals. Default: True

    Returns
    -------
    dict with 'train_loader', 'val_loader', 'test_loader', 'metadata'
    """
    _validate_config(config)
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if not 0 < train_split < 1 or not 0 < val_split < 1:
        raise ValueError("train_split and val_split must be between 0 and 1")
    if train_split + val_split >= 1:
        raise ValueError("train_split + val_split must be < 1")

    if num_workers is None:
        num_workers = 1

    if detectors is None:
        detectors = ['H1', 'L1']

    sky_params_provided = {
        'ra': 'ra' in config,
        'dec': 'dec' in config,
        'polarization': 'polarization' in config,
        'gps_time': 'gps_time' in config
    }

    if any(sky_params_provided.values()):
        print(f"Generating {num_samples} waveforms with projection to {detectors}")
        print(f"  Sky parameters: ra={'provided' if sky_params_provided['ra'] else 'default (0.0)'}, "
              f"dec={'provided' if sky_params_provided['dec'] else 'default (π/2)'}, "
              f"psi={'provided' if sky_params_provided['polarization'] else 'default (0.0)'}, "
              f"gps_time={'provided' if sky_params_provided['gps_time'] else 'default (1126259462.4)'}")
    else:
        print(f"Generating {num_samples} waveforms with projection to {detectors}")
        print(f"  Using default sky location: ra=0.0, dec=π/2 (north pole), psi=0.0")
        print(f"  Using default GPS time: 1126259462.4 (GW150914)")

    target_length = int(signal_length / time_resolution)
    print(f"Target signal length: {target_length} samples ({signal_length}s at {time_resolution}s resolution)")
    print(f"Noise injection: {'enabled (' + noise_backend + ')' if add_noise else 'disabled'}")

    if add_noise and noise_backend == 'o4_psd':
        _sample_rate = int(round(1.0 / time_resolution))
        for _det in detectors:
            build_o4_psd_cache(_det, n_segments=_DEFAULT_N_SEGS, sample_rate=_sample_rate, cache_dir=psd_cache_dir)

    param_dicts = _generate_parameter_sets(config, num_samples)

    all_successful = []
    all_failed = []
    num_chunks = (num_samples + chunk_size - 1) // chunk_size

    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, num_samples)
        chunk_params = param_dicts[chunk_start:chunk_end]

        if num_chunks > 1:
            print(f"\nChunk {chunk_idx + 1}/{num_chunks} ({len(chunk_params)} waveforms)...")

        chunk_results = _generate_waveforms_parallel(
            chunk_params, time_resolution, approximant, f_lower, num_workers, show_progress, detectors, target_length, add_noise, f_final, noise_backend, psd_cache_dir, highpass_fc
        )

        for r in chunk_results:
            if r['success']:
                all_successful.append(r)
            else:
                all_failed.append(r)

        if num_chunks > 1:
            print(f"  Chunk: {len([r for r in chunk_results if r['success']])} successful")

    if not all_successful:
        raise RuntimeError("No waveforms were successfully generated!")

    num_success = len(all_successful)
    num_failed = len(all_failed)
    print(f"\nGeneration complete: {num_success} successful, {num_failed} failed")

    print(f"\nProcessing {num_success} waveforms...")

    param_names = list(all_successful[0]['params'].keys())
    num_params = len(param_names)
    detector_names = list(all_successful[0]['detectors'].keys())
    num_detectors = len(detector_names)

    print(f"  Detector channels: {detector_names}")
    print(f"  All signals fixed to: {target_length} samples ({signal_length}s)")

    signal_array = np.empty((num_success, num_detectors, target_length), dtype=np.float32)
    param_array = np.empty((num_success, num_params), dtype=np.float32)

    print(f"  Extracting signals and parameters...")

    for i, waveform_data in enumerate(all_successful):
        for j, param_name in enumerate(param_names):
            param_array[i, j] = waveform_data['params'][param_name]
        for k, det_name in enumerate(detector_names):
            signal_array[i, k, :] = waveform_data['detectors'][det_name]

    print(f"  Converting to PyTorch tensors...")

    X = torch.from_numpy(signal_array)
    y = torch.from_numpy(param_array)

    print(f"  Tensors: X={X.shape}, y={y.shape}")

    dataset = TensorDataset(X, y)
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size

    train_data, val_data, test_data = random_split(
        dataset, [train_size, val_size, test_size]
    )

    print(f"  Splits: train={train_size}, val={val_size}, test={test_size}")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    print(f"\nReady! DataLoaders with batch_size={batch_size}")

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'metadata': {
            'parameter_names': param_names,
            'num_samples': num_success,
            'num_failed': num_failed,
            'waveform_shape': tuple(X.shape[1:]),
            'channels': detector_names,
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_size,
            'batch_size': batch_size,
            'time_resolution': time_resolution,
            'approximant': approximant,
            'f_lower': f_lower,
            'f_final': f_final,
            'highpass_fc': highpass_fc,
            'detectors': detector_names,
            'target_length': target_length,
            'signal_length': signal_length,
            'chunk_size': chunk_size,
            'sky_params_provided': sky_params_provided,
            'add_noise': add_noise,
            'noise_backend': noise_backend,
            'preprocessing': {}
        }
    }


def pycbc_massive_gravity_data_generator(config: Dict[str, Callable],
                                   num_samples: int,
                                   m_g: float = None,
                                   time_resolution: float = 1/4096,
                                   approximant: str = 'IMRPhenomD',
                                   f_lower: float = 10.0,
                                   f_final: float = 2048.0,
                                   highpass_fc: float = _HIGHPASS_FC,
                                   num_workers: int = None,
                                   signal_length: float = 2.0,
                                   batch_size: int = 256,
                                   chunk_size: int = 10000,
                                   train_split: float = 0.8,
                                   val_split: float = 0.1,
                                   show_progress: bool = True,
                                   detectors: List[str] = None,
                                   add_noise: bool = True,
                                   noise_backend: str = 'o4_psd',
                                   psd_cache_dir: str = _DEFAULT_CACHE) -> Dict:
    """
    Generate massive gravity waveforms projected to detectors.
    Returns PyTorch DataLoaders for training, validation, and testing.

    Mirrors pycbc_data_generator but applies a massive graviton phase
    modification in the frequency domain before converting to time domain.

    Parameters
    ----------
    config : dict
        Dictionary mapping parameter names to numpy distribution functions.
        Required: 'mass1', 'mass2'. Optional: 'redshift' (default 0.1),
        plus all standard sky/orientation params.
    num_samples : int
        Total number of waveforms to generate.
    m_g : float, optional
        Graviton mass in kg (dataset-level constant).
        If None, 'm_g' must be provided in config as a per-sample
        distribution. If both are given, the config (per-sample) takes
        precedence.
    time_resolution : float
        Time step delta_t. Default: 1/4096
    approximant : str
        FD waveform approximant. Default: 'IMRPhenomD'
    f_lower : float
        Lower frequency cutoff in Hz. Default: 30.0
    f_final : float
        Upper frequency cutoff in Hz. Default: 2048.0
    num_workers : int
        Parallel processes. Default: 1
    signal_length : float
        Duration in seconds. Default: 2.0
    batch_size : int
        DataLoader batch size. Default: 256
    chunk_size : int
        Chunk size for memory. Default: 10000
    train_split : float
        Training fraction. Default: 0.8
    val_split : float
        Validation fraction. Default: 0.1
    show_progress : bool
        Show tqdm progress bar. Default: True
    detectors : list of str
        Detector names. Default: ['H1', 'L1']
    add_noise : bool
        Whether to add detector noise. Default: True

    Returns
    -------
    dict with 'train_loader', 'val_loader', 'test_loader', 'metadata'
    """
    _validate_config(config)
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if not 0 < train_split < 1 or not 0 < val_split < 1:
        raise ValueError("train_split and val_split must be between 0 and 1")
    if train_split + val_split >= 1:
        raise ValueError("train_split + val_split must be < 1")
    m_g_in_config = 'm_g' in config
    if m_g is None and not m_g_in_config:
        raise ValueError("m_g must be provided either as an argument or in config")
    if m_g is not None and m_g <= 0:
        raise ValueError("m_g must be positive")

    if num_workers is None:
        num_workers = 1

    if detectors is None:
        detectors = ['H1', 'L1']

    sky_params_provided = {
        'ra': 'ra' in config,
        'dec': 'dec' in config,
        'polarization': 'polarization' in config,
        'gps_time': 'gps_time' in config
    }

    target_length = int(signal_length / time_resolution)
    if m_g_in_config:
        print(f"Generating {num_samples} MODIFIED waveforms (m_g=per-sample from config)")
    else:
        print(f"Generating {num_samples} MODIFIED waveforms (m_g={m_g:.2e} kg)")
    print(f"  Approximant: {approximant} (frequency domain)")
    print(f"  Frequency range: {f_lower}-{f_final} Hz")
    print(f"  Detectors: {detectors}")
    print(f"  Target signal length: {target_length} samples ({signal_length}s at {time_resolution}s resolution)")
    print(f"  Noise injection: {'enabled (' + noise_backend + ')' if add_noise else 'disabled'}")

    if add_noise and noise_backend == 'o4_psd':
        _sample_rate = int(round(1.0 / time_resolution))
        for _det in detectors:
            build_o4_psd_cache(_det, n_segments=_DEFAULT_N_SEGS, sample_rate=_sample_rate, cache_dir=psd_cache_dir)

    param_dicts = _generate_parameter_sets(config, num_samples)

    all_successful = []
    all_failed = []
    num_chunks = (num_samples + chunk_size - 1) // chunk_size

    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, num_samples)
        chunk_params = param_dicts[chunk_start:chunk_end]

        if num_chunks > 1:
            print(f"\nChunk {chunk_idx + 1}/{num_chunks} ({len(chunk_params)} waveforms)...")

        chunk_results = _generate_modified_waveforms_parallel(
            chunk_params, time_resolution, approximant, f_lower,
            num_workers, show_progress, detectors, target_length,
            add_noise, m_g, f_final, noise_backend, psd_cache_dir, highpass_fc
        )

        for r in chunk_results:
            if r['success']:
                all_successful.append(r)
            else:
                all_failed.append(r)

        if num_chunks > 1:
            print(f"  Chunk: {len([r for r in chunk_results if r['success']])} successful")

    if not all_successful:
        raise RuntimeError("No waveforms were successfully generated!")

    num_success = len(all_successful)
    num_failed = len(all_failed)
    print(f"\nGeneration complete: {num_success} successful, {num_failed} failed")

    print(f"\nProcessing {num_success} waveforms...")

    param_names = list(all_successful[0]['params'].keys())
    num_params = len(param_names)
    detector_names = list(all_successful[0]['detectors'].keys())
    num_detectors = len(detector_names)

    print(f"  Detector channels: {detector_names}")
    print(f"  All signals fixed to: {target_length} samples ({signal_length}s)")

    signal_array = np.empty((num_success, num_detectors, target_length), dtype=np.float32)
    param_array = np.empty((num_success, num_params), dtype=np.float32)

    print(f"  Extracting signals and parameters...")

    for i, waveform_data in enumerate(all_successful):
        for j, param_name in enumerate(param_names):
            param_array[i, j] = waveform_data['params'][param_name]
        for k, det_name in enumerate(detector_names):
            signal_array[i, k, :] = waveform_data['detectors'][det_name]

    print(f"  Converting to PyTorch tensors...")

    X = torch.from_numpy(signal_array)
    y = torch.from_numpy(param_array)

    print(f"  Tensors: X={X.shape}, y={y.shape}")

    dataset = TensorDataset(X, y)
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size

    train_data, val_data, test_data = random_split(
        dataset, [train_size, val_size, test_size]
    )

    print(f"  Splits: train={train_size}, val={val_size}, test={test_size}")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    print(f"\nReady! DataLoaders with batch_size={batch_size}")

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'metadata': {
            'parameter_names': param_names,
            'num_samples': num_success,
            'num_failed': num_failed,
            'waveform_shape': tuple(X.shape[1:]),
            'channels': detector_names,
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_size,
            'batch_size': batch_size,
            'time_resolution': time_resolution,
            'approximant': approximant,
            'f_lower': f_lower,
            'f_final': f_final,
            'highpass_fc': highpass_fc,
            'detectors': detector_names,
            'target_length': target_length,
            'signal_length': signal_length,
            'chunk_size': chunk_size,
            'sky_params_provided': sky_params_provided,
            'add_noise': add_noise,
            'noise_backend': noise_backend,
            'm_g': m_g,
            'm_g_varied': m_g_in_config,
            'waveform_type': 'massive_gravity',
            'preprocessing': {}
        }
    }


def pycbc_lorentz_violation_data_generator(config: Dict[str, Callable],
                                            num_samples: int,
                                            alpha_lv: float,
                                            A: float = None,
                                            m_g: float = None,
                                            time_resolution: float = 1/4096,
                                            approximant: str = 'IMRPhenomD',
                                            f_lower: float = 10.0,
                                            f_final: float = 2048.0,
                                            highpass_fc: float = _HIGHPASS_FC,
                                            num_workers: int = None,
                                            signal_length: float = 2.0,
                                            batch_size: int = 256,
                                            chunk_size: int = 10000,
                                            train_split: float = 0.8,
                                            val_split: float = 0.1,
                                            show_progress: bool = True,
                                            detectors: List[str] = None,
                                            add_noise: bool = True,
                                            noise_backend: str = 'o4_psd',
                                            psd_cache_dir: str = _DEFAULT_CACHE) -> Dict:
    """
    Generate Lorentz-violating (LV) waveforms projected to detectors.
    Returns PyTorch DataLoaders for training, validation, and testing.

    Mirrors pycbc_massive_gravity_data_generator but applies the generalised
    LV phase from Mirshekari, Yunes & Will (2011), arXiv:1110.2720, which
    combines a massive graviton term (m_g) with a power-law LV term
    (alpha_lv, A).

    Parameters
    ----------
    config : dict
        Dictionary mapping parameter names to numpy distribution functions.
        Required: 'mass1', 'mass2'. Optional: 'redshift' (default 0.1),
        'm_g' (per-sample override), 'A' (per-sample override),
        plus all standard sky/orientation params.
    num_samples : int
        Total number of waveforms to generate.
    alpha_lv : float
        LV dispersion exponent. Key values: 2.5 (non-commutative geometry),
        3.0 (doubly special relativity), 4.0 (extra dimensions / Horava-Lifshitz).
    A : float, optional
        LV dispersion coefficient in eV^{2-alpha} (dataset-level constant).
        If None, 'A' must be provided in config as a per-sample distribution.
        If both are given, the config (per-sample) takes precedence.
        Set to np.inf to suppress the LV term (pure massive-graviton limit).
    m_g : float, optional
        Graviton mass in kg. Use np.inf to suppress the mass term (pure LV
        limit). If None, 'm_g' must be in config or m_g defaults to np.inf.
    time_resolution : float
        Time step delta_t. Default: 1/4096
    approximant : str
        FD waveform approximant. Default: 'IMRPhenomD'
    f_lower : float
        Lower frequency cutoff in Hz. Default: 30.0
    f_final : float
        Upper frequency cutoff in Hz. Default: 2048.0
    num_workers : int
        Parallel processes. Default: 1
    signal_length : float
        Duration in seconds. Default: 2.0
    batch_size : int
        DataLoader batch size. Default: 256
    chunk_size : int
        Chunk size for memory. Default: 10000
    train_split : float
        Training fraction. Default: 0.8
    val_split : float
        Validation fraction. Default: 0.1
    show_progress : bool
        Show tqdm progress bar. Default: True
    detectors : list of str
        Detector names. Default: ['H1', 'L1']
    add_noise : bool
        Whether to add detector noise. Default: True

    Returns
    -------
    dict with 'train_loader', 'val_loader', 'test_loader', 'metadata'
    """
    _validate_config(config)
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if not 0 < train_split < 1 or not 0 < val_split < 1:
        raise ValueError("train_split and val_split must be between 0 and 1")
    if train_split + val_split >= 1:
        raise ValueError("train_split + val_split must be < 1")

    A_in_config = 'A' in config
    if A is None and not A_in_config:
        raise ValueError("A must be provided either as an argument or in config")
    if A is not None and A <= 0:
        raise ValueError("A must be positive")

    m_g_in_config = 'm_g' in config
    # m_g defaults to np.inf (suppress mass term) if not given
    if m_g is None and not m_g_in_config:
        m_g = np.inf
    if m_g is not None and m_g <= 0:
        raise ValueError("m_g must be positive")

    if num_workers is None:
        num_workers = 1

    if detectors is None:
        detectors = ['H1', 'L1']

    sky_params_provided = {
        'ra': 'ra' in config,
        'dec': 'dec' in config,
        'polarization': 'polarization' in config,
        'gps_time': 'gps_time' in config
    }

    target_length = int(signal_length / time_resolution)
    print(f"Generating {num_samples} LORENTZ-VIOLATING waveforms (alpha_lv={alpha_lv})")
    if A_in_config:
        print(f"  A=per-sample from config")
    else:
        print(f"  A={A:.2e}")
    if m_g_in_config:
        print(f"  m_g=per-sample from config")
    elif np.isinf(m_g):
        print(f"  m_g=inf (mass term suppressed)")
    else:
        print(f"  m_g={m_g:.2e} kg")
    print(f"  Approximant: {approximant} (frequency domain)")
    print(f"  Frequency range: {f_lower}-{f_final} Hz")
    print(f"  Detectors: {detectors}")
    print(f"  Target signal length: {target_length} samples ({signal_length}s at {time_resolution}s resolution)")
    print(f"  Noise injection: {'enabled (' + noise_backend + ')' if add_noise else 'disabled'}")

    if add_noise and noise_backend == 'o4_psd':
        _sample_rate = int(round(1.0 / time_resolution))
        for _det in detectors:
            build_o4_psd_cache(_det, n_segments=_DEFAULT_N_SEGS, sample_rate=_sample_rate, cache_dir=psd_cache_dir)

    param_dicts = _generate_parameter_sets(config, num_samples)

    all_successful = []
    all_failed = []
    num_chunks = (num_samples + chunk_size - 1) // chunk_size

    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, num_samples)
        chunk_params = param_dicts[chunk_start:chunk_end]

        if num_chunks > 1:
            print(f"\nChunk {chunk_idx + 1}/{num_chunks} ({len(chunk_params)} waveforms)...")

        chunk_results = _generate_lv_waveforms_parallel(
            chunk_params, time_resolution, approximant, f_lower,
            num_workers, show_progress, detectors, target_length,
            add_noise, m_g, alpha_lv, A, f_final, noise_backend, psd_cache_dir, highpass_fc
        )

        for r in chunk_results:
            if r['success']:
                all_successful.append(r)
            else:
                all_failed.append(r)

        if num_chunks > 1:
            print(f"  Chunk: {len([r for r in chunk_results if r['success']])} successful")

    if not all_successful:
        raise RuntimeError("No waveforms were successfully generated!")

    num_success = len(all_successful)
    num_failed = len(all_failed)
    print(f"\nGeneration complete: {num_success} successful, {num_failed} failed")

    print(f"\nProcessing {num_success} waveforms...")

    param_names = list(all_successful[0]['params'].keys())
    num_params = len(param_names)
    detector_names = list(all_successful[0]['detectors'].keys())
    num_detectors = len(detector_names)

    print(f"  Detector channels: {detector_names}")
    print(f"  All signals fixed to: {target_length} samples ({signal_length}s)")

    signal_array = np.empty((num_success, num_detectors, target_length), dtype=np.float32)
    param_array = np.empty((num_success, num_params), dtype=np.float32)

    print(f"  Extracting signals and parameters...")

    for i, waveform_data in enumerate(all_successful):
        for j, param_name in enumerate(param_names):
            param_array[i, j] = waveform_data['params'][param_name]
        for k, det_name in enumerate(detector_names):
            signal_array[i, k, :] = waveform_data['detectors'][det_name]

    print(f"  Converting to PyTorch tensors...")

    X = torch.from_numpy(signal_array)
    y = torch.from_numpy(param_array)

    print(f"  Tensors: X={X.shape}, y={y.shape}")

    dataset = TensorDataset(X, y)
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size

    train_data, val_data, test_data = random_split(
        dataset, [train_size, val_size, test_size]
    )

    print(f"  Splits: train={train_size}, val={val_size}, test={test_size}")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    print(f"\nReady! DataLoaders with batch_size={batch_size}")

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'metadata': {
            'parameter_names': param_names,
            'num_samples': num_success,
            'num_failed': num_failed,
            'waveform_shape': tuple(X.shape[1:]),
            'channels': detector_names,
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_size,
            'batch_size': batch_size,
            'time_resolution': time_resolution,
            'approximant': approximant,
            'f_lower': f_lower,
            'f_final': f_final,
            'highpass_fc': highpass_fc,
            'detectors': detector_names,
            'target_length': target_length,
            'signal_length': signal_length,
            'chunk_size': chunk_size,
            'sky_params_provided': sky_params_provided,
            'add_noise': add_noise,
            'noise_backend': noise_backend,
            'alpha_lv': alpha_lv,
            'A': A,
            'A_varied': A_in_config,
            'm_g': m_g,
            'm_g_varied': m_g_in_config,
            'waveform_type': 'lorentz_violation',
            'preprocessing': {}
        }
    }



################### Save / Load ###################

_WHITEN_SHARED: Dict = {}


def _whiten_init(flat: np.ndarray, delta_t: float) -> None:
    """Initializer for _whiten_batch worker processes.

    Fork on Linux shares the parent's memory via copy-on-write, so the
    workers read ``flat`` without duplicating the array.
    """
    _WHITEN_SHARED['flat'] = flat
    _WHITEN_SHARED['delta_t'] = delta_t


def _whiten_worker(idx: int) -> Tuple[int, np.ndarray]:
    flat = _WHITEN_SHARED['flat']
    delta_t = _WHITEN_SHARED['delta_t']
    w, _, _ = whiten_waveform(
        flat[idx], delta_t=delta_t, f_lower=20.0,
        apply_bandpass=True, apply_tukey=True,
        tukey_alpha=0.1, tukey_side='both',
    )
    return idx, w


def _whiten_batch(X: torch.Tensor, delta_t: float,
                  num_workers: int = None) -> torch.Tensor:
    """
    Apply ``whiten_waveform`` to every (sample, detector) pair of a
    (N, D, T) tensor and return a tensor of the same shape and dtype.

    Uses the same settings as the real-data pipeline (Tukey α=0.1 both sides,
    bandpass 35–300 Hz, f_lower=20 Hz) so simulated and real processed data
    share an identical transformation.

    Parallel on Linux via a fork-based ``multiprocessing.Pool``. Set
    ``num_workers`` explicitly or via ``WHITEN_WORKERS`` env var; defaults
    to ``os.cpu_count() - 1``. Set to ``1`` to force serial.
    """
    arr = X.detach().cpu().numpy().astype(np.float64)
    N, D, T = arr.shape
    flat = arr.reshape(N * D, T)
    out_flat = np.zeros_like(flat)

    if num_workers is None:
        num_workers = int(os.environ.get(
            'WHITEN_WORKERS', max(1, (os.cpu_count() or 2) - 1)))

    if num_workers <= 1 or N * D < 128:
        for i in range(N * D):
            w, _, _ = whiten_waveform(
                flat[i], delta_t=delta_t, f_lower=20.0,
                apply_bandpass=True, apply_tukey=True,
                tukey_alpha=0.1, tukey_side='both',
            )
            out_flat[i] = w
    else:
        ctx = multiprocessing.get_context('fork')
        chunksize = max(1, (N * D) // (num_workers * 8))
        with ctx.Pool(
            processes=num_workers,
            initializer=_whiten_init,
            initargs=(flat, delta_t),
        ) as pool:
            for idx, w in pool.imap_unordered(
                _whiten_worker, range(N * D), chunksize=chunksize,
            ):
                out_flat[idx] = w

    return torch.from_numpy(out_flat.reshape(N, D, T)).to(X.dtype)


def save_dataloaders(result: Dict, save_path: str) -> None:
    """
    Save the datasets from a pycbc_data_generator result.

    Saves only the whitened + bandpassed waveforms as ``X_whitened``, using the
    same whitening settings as the real-data pipeline. The raw signal+noise
    tensor is generated in memory for whitening but not written to disk.

    Parameters
    ----------
    result : dict
        The result dictionary from pycbc_data_generator containing
        train_loader, val_loader, test_loader, and metadata
    save_path : str
        Path where to save the data (e.g., 'my_data.pt')

    Examples
    --------
    >>> result = pycbc_data_generator(config, num_samples=1000)
    >>> save_dataloaders(result, 'my_waveforms.pt')
    """
    print(f"Saving datasets to {save_path}...")

    train_dataset = result['train_loader'].dataset
    val_dataset = result['val_loader'].dataset
    test_dataset = result['test_loader'].dataset

    base_dataset = train_dataset.dataset
    X = base_dataset.tensors[0]
    y = base_dataset.tensors[1]

    delta_t = result['metadata']['time_resolution']
    print(f"  Whitening {X.shape[0]} sample(s) across {X.shape[1]} detector(s) …")
    X_whitened = _whiten_batch(X, delta_t)

    save_data = {
        'train_indices': train_dataset.indices,
        'val_indices': val_dataset.indices,
        'test_indices': test_dataset.indices,
        'X_whitened': X_whitened,
        'y': y,
        'metadata': result['metadata']
    }

    parent = os.path.dirname(save_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.save(save_data, save_path)
    print(f"  Saved successfully!")


def load_dataloaders(load_path: str, batch_size: int = None, shuffle_train: bool = True) -> Dict:
    """
    Load previously saved datasets and create DataLoaders.

    Parameters
    ----------
    load_path : str
        Path to the saved data file (created with save_dataloaders)
    batch_size : int, optional
        Batch size for DataLoaders. If None, uses the batch_size from metadata.
    shuffle_train : bool
        Whether to shuffle training data. Default: True

    Returns
    -------
    dict with 'train_loader', 'val_loader', 'test_loader', 'metadata'

    Examples
    --------
    >>> result = pycbc_data_generator(config, num_samples=1000)
    >>> save_dataloaders(result, 'my_data.pt')
    >>> loaded = load_dataloaders('my_data.pt')
    """
    print(f"Loading datasets from {load_path}...")

    save_data = torch.load(load_path, weights_only=False)

    X = save_data['X_whitened']
    y = save_data['y']
    train_indices = save_data['train_indices']
    val_indices = save_data['val_indices']
    test_indices = save_data['test_indices']
    metadata = save_data['metadata']

    if batch_size is None:
        batch_size = metadata['batch_size']
    else:
        metadata = metadata.copy()
        metadata['batch_size'] = batch_size

    print(f"  Tensors: X={X.shape}, y={y.shape}")
    print(f"  Splits: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")

    full_dataset = TensorDataset(X, y)

    train_data = Subset(full_dataset, train_indices)
    val_data = Subset(full_dataset, val_indices)
    test_data = Subset(full_dataset, test_indices)

    # RandomSampler rejects empty datasets; disable shuffle when train is empty.
    train_loader = DataLoader(train_data, batch_size=batch_size,
                              shuffle=shuffle_train and len(train_data) > 0)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    print(f"\nReady! DataLoaders with batch_size={batch_size}")

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'metadata': metadata
    }


################### DataLoader post-processing ###################

def resample_dataloaders(result: Dict,
                        target_sample_rate: float,
                        batch_size: int = None,
                        preserve_splits: bool = True) -> Dict:
    """
    Resample all waveforms in a DataLoader result to a different sampling rate.

    Parameters
    ----------
    result : dict
        Result dictionary from pycbc_data_generator() or load_dataloaders()
    target_sample_rate : float
        Target sampling rate in Hz (e.g., 2048 for downsampling from 4096)
    batch_size : int, optional
        Batch size for new DataLoaders. If None, uses original batch_size
    preserve_splits : bool
        If True, maintains original train/val/test splits. Default: True

    Returns
    -------
    dict
        New DataLoader result with resampled data, same structure as input
    """
    metadata = result['metadata'].copy()
    original_delta_t = metadata['time_resolution']
    original_rate = 1.0 / original_delta_t
    target_delta_t = 1.0 / target_sample_rate

    print(f"Resampling DataLoaders from {original_rate:.0f} Hz to {target_sample_rate:.0f} Hz...")

    train_dataset = result['train_loader'].dataset
    val_dataset = result['val_loader'].dataset
    test_dataset = result['test_loader'].dataset

    base_dataset = train_dataset.dataset
    X_full = base_dataset.tensors[0]
    y_full = base_dataset.tensors[1]

    train_indices = train_dataset.indices
    val_indices = val_dataset.indices
    test_indices = test_dataset.indices

    num_samples = X_full.shape[0]
    num_detectors = X_full.shape[1]
    original_length = X_full.shape[2]
    original_duration = original_length * original_delta_t
    new_length = int(original_duration / target_delta_t)

    print(f"  Processing {num_samples} waveforms...")
    print(f"  Original: {original_length} samples/waveform")
    print(f"  New: {new_length} samples/waveform")

    X_resampled = torch.zeros(num_samples, num_detectors, new_length, dtype=torch.float32)

    for i in range(num_samples):
        for j in range(num_detectors):
            waveform = X_full[i, j, :].numpy()
            resampled = resample_waveform(
                waveform,
                original_delta_t=original_delta_t,
                target_delta_t=target_delta_t
            )
            X_resampled[i, j, :] = torch.from_numpy(resampled)

    print(f"  Resampling complete!")

    new_dataset = TensorDataset(X_resampled, y_full)

    if preserve_splits:
        train_data = Subset(new_dataset, train_indices)
        val_data = Subset(new_dataset, val_indices)
        test_data = Subset(new_dataset, test_indices)
        train_size = len(train_indices)
        val_size = len(val_indices)
        test_size = len(test_indices)
    else:
        train_size = len(train_indices)
        val_size = len(val_indices)
        test_size = len(test_indices)
        train_data, val_data, test_data = random_split(
            new_dataset, [train_size, val_size, test_size]
        )

    if batch_size is None:
        batch_size = metadata.get('batch_size', 256)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    new_metadata = metadata.copy()
    new_metadata['time_resolution'] = target_delta_t
    new_metadata['waveform_shape'] = (num_detectors, new_length)
    new_metadata['target_length'] = new_length
    new_metadata['batch_size'] = batch_size
    new_metadata['train_size'] = train_size
    new_metadata['val_size'] = val_size
    new_metadata['test_size'] = test_size

    if 'preprocessing' not in new_metadata:
        new_metadata['preprocessing'] = {}

    new_metadata['preprocessing'] = new_metadata['preprocessing'].copy()
    new_metadata['preprocessing']['resampled'] = True
    new_metadata['preprocessing']['original_rate'] = original_rate
    new_metadata['preprocessing']['resample_rate'] = target_sample_rate
    new_metadata['preprocessing']['final_length_samples'] = new_length

    print(f"\nNew DataLoaders created:")
    print(f"  Sampling rate: {target_sample_rate:.0f} Hz")
    print(f"  Waveform shape: {new_metadata['waveform_shape']}")
    print(f"  Batch size: {batch_size}")

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'metadata': new_metadata
    }


def truncate_dataloaders(result: Dict,
                         target_length: int = None,
                         target_duration: float = None,
                         keep_end: bool = True,
                         batch_size: int = None,
                         preserve_splits: bool = True) -> Dict:
    """
    Truncate all waveforms in a DataLoader result to a specified length.

    Parameters
    ----------
    result : dict
        Result dictionary from pycbc_data_generator() or load_dataloaders()
    target_length : int, optional
        Target length in samples. Either this or target_duration must be specified.
    target_duration : float, optional
        Target duration in seconds. Will be converted to samples using metadata.
    keep_end : bool
        If True (default), keeps the END of the signal (where merger is).
        If False, keeps the START of the signal.
    batch_size : int, optional
        Batch size for new DataLoaders. If None, uses original batch_size.
    preserve_splits : bool
        If True, maintains original train/val/test splits. Default: True

    Returns
    -------
    dict
        New DataLoader result with truncated data, same structure as input
    """
    metadata = result['metadata'].copy()
    delta_t = metadata['time_resolution']
    sample_rate = 1.0 / delta_t

    if target_length is None and target_duration is None:
        raise ValueError("Must specify either target_length or target_duration")
    if target_length is not None and target_duration is not None:
        raise ValueError("Specify only one of target_length or target_duration")

    if target_duration is not None:
        target_length = int(target_duration * sample_rate)

    train_dataset = result['train_loader'].dataset
    val_dataset = result['val_loader'].dataset
    test_dataset = result['test_loader'].dataset

    base_dataset = train_dataset.dataset
    X_full = base_dataset.tensors[0]
    y_full = base_dataset.tensors[1]

    train_indices = train_dataset.indices
    val_indices = val_dataset.indices
    test_indices = test_dataset.indices

    num_samples = X_full.shape[0]
    num_detectors = X_full.shape[1]
    original_length = X_full.shape[2]

    if target_length > original_length:
        raise ValueError(f"target_length ({target_length}) cannot be greater than "
                        f"original length ({original_length})")

    original_duration = original_length * delta_t
    new_duration = target_length * delta_t

    print(f"Truncating DataLoaders...")
    print(f"  Original: {original_length} samples ({original_duration:.3f}s)")
    print(f"  Target: {target_length} samples ({new_duration:.3f}s)")
    print(f"  Keeping: {'END' if keep_end else 'START'} of signal")

    if keep_end:
        X_truncated = X_full[:, :, -target_length:]
    else:
        X_truncated = X_full[:, :, :target_length]

    X_truncated = X_truncated.clone()

    print(f"  Processing {num_samples} waveforms... done!")

    new_dataset = TensorDataset(X_truncated, y_full)

    if preserve_splits:
        train_data = Subset(new_dataset, train_indices)
        val_data = Subset(new_dataset, val_indices)
        test_data = Subset(new_dataset, test_indices)
        train_size = len(train_indices)
        val_size = len(val_indices)
        test_size = len(test_indices)
    else:
        train_size = len(train_indices)
        val_size = len(val_indices)
        test_size = len(test_indices)
        train_data, val_data, test_data = random_split(
            new_dataset, [train_size, val_size, test_size]
        )

    if batch_size is None:
        batch_size = metadata.get('batch_size', 256)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    new_metadata = metadata.copy()
    new_metadata['waveform_shape'] = (num_detectors, target_length)
    new_metadata['target_length'] = target_length
    new_metadata['signal_length'] = new_duration
    new_metadata['batch_size'] = batch_size
    new_metadata['train_size'] = train_size
    new_metadata['val_size'] = val_size
    new_metadata['test_size'] = test_size

    if 'preprocessing' not in new_metadata:
        new_metadata['preprocessing'] = {}

    new_metadata['preprocessing'] = new_metadata['preprocessing'].copy()
    new_metadata['preprocessing']['truncated'] = True
    new_metadata['preprocessing']['original_length'] = original_length
    new_metadata['preprocessing']['truncated_length'] = target_length
    new_metadata['preprocessing']['keep_end'] = keep_end

    print(f"\nNew DataLoaders created:")
    print(f"  Waveform shape: {new_metadata['waveform_shape']}")
    print(f"  Duration: {new_duration:.3f}s")
    print(f"  Batch size: {batch_size}")

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'metadata': new_metadata
    }


def _whiten_single_waveform(args):
    """Worker function for parallel whitening."""
    waveform, delta_t, f_lower, apply_bandpass, apply_tukey, tukey_alpha, tukey_side, psd = args
    whitened, _, _ = whiten_waveform(
        waveform,
        delta_t=delta_t,
        f_lower=f_lower,
        apply_bandpass=apply_bandpass,
        apply_tukey=apply_tukey,
        tukey_alpha=tukey_alpha,
        tukey_side=tukey_side,
        psd=psd,
    )
    return whitened


def whiten_dataloaders(result: Dict,
                       f_lower: float = 20.0,
                       apply_bandpass: bool = True,
                       apply_tukey: bool = True,
                       tukey_alpha: float = 0.1,
                       tukey_side: str = 'both',
                       batch_size: int = None,
                       preserve_splits: bool = True,
                       num_workers: int = 1,
                       show_progress: bool = True,
                       psd_cache_dir: str = _DEFAULT_CACHE) -> Dict:
    """
    Whiten all waveforms in a DataLoader result.

    For simulated datasets (noise_backend stored in metadata) the known
    generation PSD is used directly rather than estimating it from each
    2-second waveform via Welch, which would give a poor estimate and
    introduce ringing artefacts.

    Parameters
    ----------
    result : dict
        Result dictionary from pycbc_data_generator() or load_dataloaders()
    f_lower : float
        Lower frequency cutoff in Hz. Default: 20.0
    apply_bandpass : bool
        Apply 35-300 Hz bandpass filter after whitening. Default: True
    apply_tukey : bool
        Apply Tukey window before whitening to prevent edge effects. Default: True
    tukey_alpha : float
        Tukey window alpha parameter. Default: 0.1
    tukey_side : str
        Which side(s) to apply the Tukey taper. Default: 'both'
        (correct when merger is centred at t=0)
    batch_size : int, optional
        Batch size for new DataLoaders. If None, uses original batch_size
    preserve_splits : bool
        If True, maintains original train/val/test splits. Default: True
    num_workers : int
        Number of parallel workers for whitening. Default: 1
    show_progress : bool
        Show progress bar during whitening. Default: True

    Returns
    -------
    dict
        New DataLoader result with whitened data, same structure as input
    """
    metadata = result['metadata'].copy()
    delta_t = metadata['time_resolution']

    print(f"Whitening DataLoaders...")
    print(f"  f_lower: {f_lower} Hz")
    print(f"  Bandpass: {apply_bandpass}")
    print(f"  Tukey window: {apply_tukey} (alpha={tukey_alpha}, side={tukey_side})")

    train_dataset = result['train_loader'].dataset
    val_dataset = result['val_loader'].dataset
    test_dataset = result['test_loader'].dataset

    base_dataset = train_dataset.dataset
    X_full = base_dataset.tensors[0]
    y_full = base_dataset.tensors[1]

    train_indices = train_dataset.indices
    val_indices = val_dataset.indices
    test_indices = test_dataset.indices

    num_samples = X_full.shape[0]
    num_detectors = X_full.shape[1]
    n_samples = X_full.shape[2]

    # Build the whitening PSD for each detector from the known generation PSD
    # rather than estimating via Welch on 2 s of data (too few segments →
    # spectral imbalances → ringing artefacts in the whitened output).
    # o4_psd: load a representative O4 PSD per detector from the cache.
    #         Each cached PSD was estimated from 32 s of real O4a data.
    #         A random draw is used per whitening call, consistent with how
    #         noise was injected during generation.
    # aligo:  analytic PSD — for debugging/quick tests only, not realistic noise.
    noise_backend = metadata.get('noise_backend')
    detector_names = metadata.get('channels', metadata.get('detectors', []))
    sample_rate = int(round(1.0 / delta_t))
    delta_f_white = 1.0 / (n_samples * delta_t)
    flen_white = n_samples // 2 + 1

    det_psds = {}
    if noise_backend == 'o4_psd':
        for det in detector_names:
            det_psds[det] = load_random_o4_psd(
                flen_white, delta_f_white, f_lower, det, sample_rate,
                cache_dir=psd_cache_dir,
            )
        print(f"  PSD: O4 cached per-detector ({', '.join(detector_names)})")
    elif noise_backend == 'aligo':
        # aligo: analytic PSD — for debugging/quick tests only, not realistic noise
        aligo_psd = aLIGOZeroDetHighPower(flen_white, delta_f_white, f_lower)
        for det in detector_names:
            det_psds[det] = aligo_psd
        print(f"  PSD: aLIGO analytic (DEBUG — use o4_psd for realistic data)")
    else:
        for det in detector_names:
            det_psds[det] = None   # Welch fallback inside whiten_waveform
        print(f"  PSD: Welch estimation (no noise_backend in metadata)")

    print(f"  Processing {num_samples} waveforms x {num_detectors} detectors...")

    X_whitened = torch.zeros_like(X_full)

    all_args = []
    for i in range(num_samples):
        for j, det in enumerate(detector_names):
            waveform = X_full[i, j, :].numpy()
            all_args.append((waveform, delta_t, f_lower, apply_bandpass, apply_tukey, tukey_alpha, tukey_side, det_psds.get(det)))

    if num_workers > 1:
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=num_workers) as pool:
            if show_progress:
                results = list(tqdm(
                    pool.imap(_whiten_single_waveform, all_args, chunksize=10),
                    total=len(all_args),
                    desc="Whitening"
                ))
            else:
                results = list(pool.imap(_whiten_single_waveform, all_args, chunksize=10))
    else:
        if show_progress:
            results = [_whiten_single_waveform(args) for args in tqdm(all_args, desc="Whitening")]
        else:
            results = [_whiten_single_waveform(args) for args in all_args]

    idx = 0
    for i in range(num_samples):
        for j in range(num_detectors):
            X_whitened[i, j, :] = torch.from_numpy(results[idx])
            idx += 1

    print(f"  Whitening complete!")

    new_dataset = TensorDataset(X_whitened, y_full)

    if preserve_splits:
        train_data = Subset(new_dataset, train_indices)
        val_data = Subset(new_dataset, val_indices)
        test_data = Subset(new_dataset, test_indices)
        train_size = len(train_indices)
        val_size = len(val_indices)
        test_size = len(test_indices)
    else:
        train_size = len(train_indices)
        val_size = len(val_indices)
        test_size = len(test_indices)
        train_data, val_data, test_data = random_split(
            new_dataset, [train_size, val_size, test_size]
        )

    if batch_size is None:
        batch_size = metadata.get('batch_size', 256)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    new_metadata = metadata.copy()
    new_metadata['batch_size'] = batch_size
    new_metadata['train_size'] = train_size
    new_metadata['val_size'] = val_size
    new_metadata['test_size'] = test_size

    if 'preprocessing' not in new_metadata:
        new_metadata['preprocessing'] = {}

    new_metadata['preprocessing'] = new_metadata['preprocessing'].copy()
    new_metadata['preprocessing']['whitened'] = True
    new_metadata['preprocessing']['whiten_f_lower'] = f_lower
    new_metadata['preprocessing']['whiten_bandpass'] = apply_bandpass
    new_metadata['preprocessing']['whiten_tukey'] = apply_tukey
    new_metadata['preprocessing']['whiten_tukey_alpha'] = tukey_alpha
    new_metadata['preprocessing']['whiten_tukey_side'] = tukey_side
    new_metadata['preprocessing']['whiten_psd_source'] = (
        noise_backend if noise_backend in ('aligo', 'o4_psd') else 'welch'
    )

    print(f"\nNew DataLoaders created:")
    print(f"  Waveform shape: {new_metadata['waveform_shape']}")
    print(f"  Batch size: {batch_size}")

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'metadata': new_metadata
    }


def normalize_dataloaders(result: Dict,
                          scale_factor: float = 1e21,
                          batch_size: int = None,
                          preserve_splits: bool = True) -> Dict:
    """
    Normalize all waveforms in a DataLoader result by multiplying by a scale factor.

    Parameters
    ----------
    result : dict
        Result dictionary from pycbc_data_generator() or load_dataloaders()
    scale_factor : float
        Fixed scaling factor to multiply waveforms by. Default: 1e21
    batch_size : int, optional
        Batch size for new DataLoaders. If None, uses original batch_size
    preserve_splits : bool
        If True, maintains original train/val/test splits. Default: True

    Returns
    -------
    dict
        New DataLoader result with normalized data, same structure as input
    """
    metadata = result['metadata'].copy()

    print(f"Normalizing DataLoaders...")
    print(f"  Scale factor: {scale_factor:.2e}")

    train_dataset = result['train_loader'].dataset
    val_dataset = result['val_loader'].dataset
    test_dataset = result['test_loader'].dataset

    base_dataset = train_dataset.dataset
    X_full = base_dataset.tensors[0]
    y_full = base_dataset.tensors[1]

    train_indices = train_dataset.indices
    val_indices = val_dataset.indices
    test_indices = test_dataset.indices

    num_samples = X_full.shape[0]

    print(f"  Processing {num_samples} waveforms...")

    X_normalized = X_full * scale_factor

    print(f"  Normalization complete!")

    new_dataset = TensorDataset(X_normalized, y_full)

    if preserve_splits:
        train_data = Subset(new_dataset, train_indices)
        val_data = Subset(new_dataset, val_indices)
        test_data = Subset(new_dataset, test_indices)
        train_size = len(train_indices)
        val_size = len(val_indices)
        test_size = len(test_indices)
    else:
        train_size = len(train_indices)
        val_size = len(val_indices)
        test_size = len(test_indices)
        train_data, val_data, test_data = random_split(
            new_dataset, [train_size, val_size, test_size]
        )

    if batch_size is None:
        batch_size = metadata.get('batch_size', 256)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    new_metadata = metadata.copy()
    new_metadata['batch_size'] = batch_size
    new_metadata['train_size'] = train_size
    new_metadata['val_size'] = val_size
    new_metadata['test_size'] = test_size

    if 'preprocessing' not in new_metadata:
        new_metadata['preprocessing'] = {}

    new_metadata['preprocessing'] = new_metadata['preprocessing'].copy()
    new_metadata['preprocessing']['normalized'] = True
    new_metadata['preprocessing']['normalize_scale'] = scale_factor

    print(f"\nNew DataLoaders created:")
    print(f"  Waveform shape: {new_metadata['waveform_shape']}")
    print(f"  Batch size: {batch_size}")

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'metadata': new_metadata
    }


################### Derived representations: Q-transform & SVD ###################

# These utilities produce alternative input representations of the whitened
# strain for ML training. They are mode-agnostic (work on GR, MG, LV datasets
# identically) since they consume only ``X_whitened`` and ``y``.


def detect_mode(y: np.ndarray, col: Dict[str, int]) -> str:
    """Infer physics mode from the label columns.

    Returns 'gr' if m_g and alpha_lv are identically zero across all rows,
    'mg' if m_g varies but alpha_lv is zero, otherwise 'lv'.
    """
    alpha = y[:, col['alpha_lv']]
    if np.any(alpha != 0):
        return 'lv'
    if np.any(y[:, col['m_g']] != 0):
        return 'mg'
    return 'gr'


def compute_qtransform_batch(
    X_whitened: np.ndarray,
    delta_t: float = 1.0 / 4096,
    frange: Tuple[float, float] = (20.0, 300.0),
    logfsteps: int = 50,
    qrange: Tuple[float, float] = (4.0, 16.0),
    delta_t_out: float = 0.002,
    progress: bool = False,
) -> np.ndarray:
    """Q-transform magnitude of every (sample, detector) pair.

    Parameters
    ----------
    X_whitened : (N, D, T) array of whitened strain
    delta_t    : time resolution of the input, seconds
    frange     : frequency range for the q-transform (Hz)
    logfsteps  : number of log-spaced frequency bins
    qrange     : (q_min, q_max) tile search range for the q-transform
    delta_t_out: time resolution of the output grid, seconds

    Returns
    -------
    Q : (N, D, F, T_q) float32 array of |q-coefficients|
    """
    X_whitened = np.asarray(X_whitened)
    N, D, _ = X_whitened.shape
    probe = TimeSeries(X_whitened[0, 0].astype(np.float64), delta_t=delta_t)
    _, _, qplane = probe.qtransform(
        delta_t=delta_t_out, logfsteps=logfsteps,
        qrange=qrange, frange=frange,
    )
    F_bins, T_q = qplane.shape
    out = np.empty((N, D, F_bins, T_q), dtype=np.float32)
    iterator = range(N)
    if progress:
        iterator = tqdm(iterator, desc='q-transform')
    for n in iterator:
        for d in range(D):
            ts = TimeSeries(X_whitened[n, d].astype(np.float64), delta_t=delta_t)
            _, _, q = ts.qtransform(
                delta_t=delta_t_out, logfsteps=logfsteps,
                qrange=qrange, frange=frange,
            )
            out[n, d] = np.abs(q).astype(np.float32)
    return out


def build_svd_basis(
    templates: np.ndarray,
    k_max: int = 600,
) -> Tuple[np.ndarray, np.ndarray]:
    """SVD of a (M, T) clean-template matrix.

    Returns
    -------
    Vh : (K, T) float32, K = min(k_max, rank) — right singular vectors
    s  : (K,) float32 — singular values in descending order

    ``Vh`` is the projection matrix: coeffs = Vh @ strain, strain ≈ coeffs @ Vh.
    """
    templates = np.asarray(templates, dtype=np.float32)
    if templates.ndim != 2:
        raise ValueError(f"expected 2D (M, T) template matrix, got {templates.shape}")
    _, s, Vh = np.linalg.svd(templates, full_matrices=False)
    k = min(k_max, Vh.shape[0])
    return Vh[:k].astype(np.float32), s[:k].astype(np.float32)


def apply_svd_projection(
    X_whitened: np.ndarray,
    basis: np.ndarray,
) -> np.ndarray:
    """Project (N, D, T) strain onto a (K, T) basis → (N, D, K) coefficients.

    Inverse: ``strain_reconstruction = coeffs @ basis``.
    """
    X_whitened = np.asarray(X_whitened, dtype=np.float32)
    basis = np.asarray(basis, dtype=np.float32)
    if X_whitened.shape[-1] != basis.shape[-1]:
        raise ValueError(
            f"length mismatch: X has T={X_whitened.shape[-1]}, "
            f"basis has T={basis.shape[-1]}")
    return np.einsum('kt,ndt->ndk', basis, X_whitened)


def save_svd_basis(
    path: str,
    basis: np.ndarray,
    singular_values: np.ndarray,
    detector: str,
    mode: str,
    n_templates: int,
) -> None:
    """Save an SVD basis to a .npz file."""
    np.savez(
        path,
        basis_vectors=basis.astype(np.float32),
        singular_values=singular_values.astype(np.float32),
        detector=detector,
        mode=mode,
        n_templates=n_templates,
    )


def load_svd_basis(path: str) -> Dict:
    """Load an SVD basis written by ``save_svd_basis``."""
    data = np.load(path, allow_pickle=False)
    return {
        'basis': data['basis_vectors'],
        'singular_values': data['singular_values'],
        'detector': str(data['detector']),
        'mode': str(data['mode']),
        'n_templates': int(data['n_templates']),
    }


def _average_o4_psd(detector: str, delta_f: float, flen: int,
                    cache_dir: str = _DEFAULT_CACHE) -> FrequencySeries:
    """Segment-averaged O4 PSD on the requested (delta_f, flen) grid."""
    data = np.load(os.path.join(cache_dir, f'o4_psds_{detector}_4096Hz.npz'))
    freqs_cache = data['freqs']
    psd_mean = data['psds'].mean(axis=0)
    safe = freqs_cache > 0
    interp = interp1d(
        np.log10(freqs_cache[safe]),
        np.log10(psd_mean[safe]),
        bounds_error=False,
        fill_value=(np.log10(psd_mean[safe][0]),
                    np.log10(psd_mean[safe][-1])),
    )
    f_out = np.arange(flen) * delta_f
    psd_out = np.empty(flen)
    psd_out[0] = 1.0
    psd_out[1:] = 10 ** interp(np.log10(f_out[1:]))
    return FrequencySeries(psd_out, delta_f=delta_f)


def _clean_whitened_template(
    params: Dict,
    mode: str,
    detector: str,
    psd: FrequencySeries,
    time_resolution: float = 1.0 / 4096,
    target_length: int = 8192,
    approximant: str = 'IMRPhenomD',
    f_lower: float = 10.0,
    f_final: float = 2048.0,
    f_lower_whiten: float = 20.0,
) -> np.ndarray:
    """Dispatch to the correct per-mode worker with ``add_noise=False`` and whiten."""
    if mode == 'gr':
        r = _generate_single_waveform(
            params=params, time_resolution=time_resolution,
            approximant=approximant, f_lower=f_lower,
            detectors=[detector], target_length=target_length,
            add_noise=False, f_final=f_final,
        )
    elif mode == 'mg':
        r = _generate_single_modified_waveform(
            params=params, time_resolution=time_resolution,
            approximant=approximant, f_lower=f_lower,
            detectors=[detector], target_length=target_length,
            add_noise=False, m_g=params['m_g'], f_final=f_final,
        )
    elif mode == 'lv':
        r = _generate_single_lv_waveform(
            params=params, time_resolution=time_resolution,
            approximant=approximant, f_lower=f_lower,
            detectors=[detector], target_length=target_length,
            add_noise=False, m_g=params['m_g'],
            alpha_lv=params['alpha_lv'], A=params['A'],
            f_final=f_final,
        )
    else:
        raise ValueError(f"unknown mode {mode!r}")
    if not r['success']:
        raise RuntimeError(r.get('error', 'template generation failed'))
    w, _, _ = whiten_waveform(
        r['detectors'][detector].numpy(),
        delta_t=time_resolution, f_lower=f_lower_whiten, psd=psd,
    )
    return w.astype(np.float32)


def generate_clean_templates(
    y: np.ndarray,
    col: Dict[str, int],
    mode: str,
    detector: str,
    n_templates: int,
    psd_cache_dir: str = _DEFAULT_CACHE,
    rng: np.random.Generator = None,
    progress: bool = False,
    time_resolution: float = 1.0 / 4096,
    target_length: int = 8192,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate clean whitened templates for building an SVD basis.

    Draws ``n_templates`` random rows of the dataset labels ``y``, runs each
    through the mode-specific per-sample worker with noise disabled, whitens
    the result against the segment-averaged O4 PSD for ``detector``, and
    returns a dense (M, T) matrix suitable for ``build_svd_basis``.

    Returns
    -------
    templates : (M, T) float32, M = number of successful generations ≤ n_templates
    indices   : (M,) int — source row indices in ``y``
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n = min(n_templates, len(y))
    idxs = rng.choice(len(y), size=n, replace=False)

    delta_f = 1.0 / (target_length * time_resolution)
    flen = target_length // 2 + 1
    psd = _average_o4_psd(detector, delta_f, flen, cache_dir=psd_cache_dir)

    keys = ('mass1', 'mass2', 'spin1z', 'spin2z', 'distance',
            'inclination', 'coa_phase', 'ra', 'dec', 'polarization',
            'm_g', 'alpha_lv', 'A')
    out = np.zeros((n, target_length), dtype=np.float32)
    kept = np.zeros(n, dtype=bool)
    iterator = enumerate(idxs)
    if progress:
        iterator = tqdm(list(iterator), desc=f'{detector} templates')
    for row, src in iterator:
        params = {k: float(y[src, col[k]]) for k in keys}
        try:
            out[row] = _clean_whitened_template(
                params=params, mode=mode, detector=detector, psd=psd,
                time_resolution=time_resolution, target_length=target_length,
            )
            kept[row] = True
        except Exception:
            pass
    return out[kept], idxs[kept]
