"""
o4_psd — O4 PSD cache and random-segment noise utility

Fetches ~100 short segments from GWOSC O4a data, estimates a Welch PSD for
each, and caches them.  At noise-generation time one PSD is chosen at random,
capturing the temporal variation in detector sensitivity across the run.

Typical usage
-------------
# One-off cache build (do this before parallel waveform generation):
from o4_psd import build_o4_psd_cache
build_o4_psd_cache('H1', n_segments=100, sample_rate=4096)
build_o4_psd_cache('L1', n_segments=100, sample_rate=4096)

# Inside a waveform worker:
from o4_psd import load_random_o4_psd
psd = load_random_o4_psd(flen, delta_f, f_lower, detector='H1', sample_rate=4096)
noise = noise_from_psd(target_length, delta_t, psd)
"""

import os
import logging
import numpy as np
from scipy.interpolate import interp1d
from pycbc.types import FrequencySeries
from pycbc.psd import aLIGOZeroDetHighPower

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# O4a GPS window (2023-05-24 18:00 UTC → 2024-01-16 16:00 UTC)
# ---------------------------------------------------------------------------
_O4A_GPS_START = 1369166418
_O4A_GPS_END   = 1389744018

_FETCH_DUR      = 256   # seconds of data per segment
_FFT_LEN        = 4     # Welch FFT length in seconds
_DEFAULT_N_SEGS = 100
_DEFAULT_CACHE  = os.path.join(os.path.expanduser('~'), '.cache', 'gw_datagen')


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

    # Build a pool of all valid non-overlapping start times
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

    # Query GWOSC for science-mode segments so we only fetch GPS times that
    # actually have data, avoiding slow timeouts on gaps.
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

        # Resample if needed
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

        # Sanity: drop zero / negative values (can appear at DC)
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
            # Interpolate onto the first segment's frequency grid for a
            # consistent stacked array shape
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
        Lower frequency cutoff.  Bins below this are set to a large sentinel
        value so ``noise_from_psd`` does not generate energy there.
    detector : str
        Detector name, e.g. ``'H1'``.
    sample_rate : int
        Sample rate in Hz (must match the cached PSD).
    cache_dir : str
        Directory containing the ``.npz`` cache file.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.  A fresh generator is
        created if not provided.

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

    data   = np.load(path)
    freqs  = data['freqs']   # (M,)
    psds   = data['psds']    # (N, M)

    if rng is None:
        rng = np.random.default_rng()

    idx       = rng.integers(0, psds.shape[0])
    psd_row   = psds[idx]

    # Interpolate in log-log space onto target frequency grid
    f_out = np.arange(flen) * delta_f

    # Avoid log(0) at DC
    safe_freqs = freqs[freqs > 0]
    safe_psd   = psd_row[freqs > 0]

    interp_fn = interp1d(
        np.log10(safe_freqs),
        np.log10(safe_psd),
        bounds_error=False,
        fill_value=(np.log10(safe_psd[0]), np.log10(safe_psd[-1])),
    )

    # Evaluate only at positive output frequencies; DC bin gets sentinel
    pos_mask = f_out > 0
    psd_out  = np.empty(flen, dtype=np.float64)
    psd_out[~pos_mask] = 1.0  # DC bin — unused by noise_from_psd
    psd_out[pos_mask]  = 10 ** interp_fn(np.log10(f_out[pos_mask]))

    # For sub-f_lower bins use the PSD value at f_lower as a flat wall.
    # Using a sentinel like 1e40 generates catastrophic sub-band noise
    # (RMS ~ sqrt(1e40 * bandwidth) ~ 10^20 strain) that overwhelms the signal.
    above = f_out >= f_lower
    wall = psd_out[above][0] if above.any() else 1e-44
    psd_out[~above & (f_out > 0)] = wall

    # Clamp any non-positive values that may arise from extrapolation
    psd_out[psd_out <= 0] = 1e-40

    return FrequencySeries(psd_out, delta_f=delta_f)


# ---------------------------------------------------------------------------
# Quick-look plot  (python o4_psd.py)
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import logging
    import matplotlib.pyplot as plt
    from pycbc.waveform import get_fd_waveform
    from pycbc.noise import noise_from_psd as _noise_from_psd
    from pycbc.types import TimeSeries
    from pycbc.filter import highpass_fir
    from pycbc.detector import Detector
    from gw_datagen import _apply_end_taper, _HIGHPASS_FC

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    DETECTOR    = 'H1'
    SAMPLE_RATE = 4096
    DELTA_T     = 1.0 / SAMPLE_RATE
    F_LOWER     = 20.0
    F_FINAL     = 2048.0
    TARGET_LEN  = 8192          # 2 seconds at 4096 Hz
    FLEN        = TARGET_LEN // 2 + 1
    DELTA_F     = 1.0 / (TARGET_LEN * DELTA_T)
    DELTA_F_FD  = 1.0 / 256     # FD resolution (matches gw_datagen pipeline)
    N_RINGDOWN  = 500            # merger + early ringdown samples
    GPS_TIME    = 1126259462.4   # GW150914 epoch (same default as gw_datagen)

    # Build PSD cache — use 3 segments for a quick debug plot.
    # For production, call build_o4_psd_cache() with n_segments=100 separately.
    build_o4_psd_cache(DETECTOR, n_segments=3, sample_rate=SAMPLE_RATE, force=True)

    # ── Step 1: frequency-domain waveform (mirrors _generate_single_waveform) ──
    print("Generating IMRPhenomD waveform via FD->IRFFT pipeline (m1=m2=30 M\u2609)...")
    hp_fd, hc_fd = get_fd_waveform(
        approximant='IMRPhenomD',
        mass1=30, mass2=30,
        spin1z=0.0, spin2z=0.0,
        inclination=0.0, coa_phase=0.0,
        distance=410.0,
        delta_f=DELTA_F_FD,
        f_lower=F_LOWER,
        f_final=F_FINAL,
    )

    # ── Step 2: IRFFT + normalise ─────────────────────────────────────────────
    hp_raw = np.fft.irfft(hp_fd.numpy())
    hc_raw = np.fft.irfft(hc_fd.numpy())
    N = len(hp_raw)
    hp_raw *= DELTA_F_FD * N
    hc_raw *= DELTA_F_FD * N

    # ── Step 3: assemble to TARGET_LEN (late inspiral + merger/ringdown) ──────
    if TARGET_LEN <= N:
        n_pre  = TARGET_LEN - N_RINGDOWN
        hp_arr = np.concatenate([hp_raw[N - n_pre:], hp_raw[:N_RINGDOWN]])
        hc_arr = np.concatenate([hc_raw[N - n_pre:], hc_raw[:N_RINGDOWN]])
    else:
        hp_arr = np.concatenate([np.zeros(TARGET_LEN - N), hp_raw])
        hc_arr = np.concatenate([np.zeros(TARGET_LEN - N), hc_raw])

    # ── Step 4: cosine-taper tail to suppress filter edge effects ─────────────
    hp_arr = _apply_end_taper(hp_arr)
    hc_arr = _apply_end_taper(hc_arr)

    # ── Step 5: highpass at _HIGHPASS_FC (35 Hz), 128 taps ───────────────────
    hp_ts = TimeSeries(hp_arr.astype(np.float64), delta_t=DELTA_T)
    hc_ts = TimeSeries(hc_arr.astype(np.float64), delta_t=DELTA_T)
    hp_ts.start_time += GPS_TIME
    hc_ts.start_time += GPS_TIME
    hp_ts = highpass_fir(hp_ts, _HIGHPASS_FC, 128)
    hc_ts = highpass_fir(hc_ts, _HIGHPASS_FC, 128)

    # ── Step 6: project onto H1 (face-on, directly overhead) ─────────────────
    det    = Detector(DETECTOR)
    signal = det.project_wave(hp_ts, hc_ts, ra=0.0, dec=np.pi / 2,
                               polarization=0.0, method='lal')
    sig = signal.numpy()
    sig_len = len(sig)
    if sig_len >= TARGET_LEN:
        sig = sig[sig_len - TARGET_LEN:]
    else:
        sig = np.concatenate([np.zeros(TARGET_LEN - sig_len, dtype=sig.dtype), sig])

    # Time axis: merger is at sample index (TARGET_LEN - N_RINGDOWN)
    merger_idx = TARGET_LEN - N_RINGDOWN
    t = (np.arange(TARGET_LEN) - merger_idx) * DELTA_T

    # ── O4 noise realisation ──────────────────────────────────────────────────
    psd   = load_random_o4_psd(FLEN, DELTA_F, F_LOWER, DETECTOR, SAMPLE_RATE)
    noise = _noise_from_psd(TARGET_LEN, DELTA_T, psd)
    noisy = sig + np.array(noise.data)

    # PSD frequency axis (skip DC)
    f_psd = np.array(psd.data)
    f_ax  = np.arange(FLEN) * DELTA_F

    fig, axes = plt.subplots(3, 1, figsize=(11, 8))
    fig.suptitle(f'O4 noise demo \u2014 {DETECTOR}  (m\u2081=m\u2082=30 M\u2609, d=410 Mpc)', fontsize=12)

    # Top: clean signal
    axes[0].plot(t, sig * 1e21, color='#e05c00', linewidth=0.9)
    axes[0].set_ylabel(r'Strain $\times10^{21}$')
    axes[0].set_title('Clean signal (FD\u2192IRFFT, highpassed, H1 projected)')
    axes[0].set_xlim(t[0], t[-1])
    axes[0].axvline(0, color='grey', linewidth=0.6, linestyle='--', alpha=0.6)

    # Middle: signal + O4 noise
    axes[1].plot(t, noisy * 1e21, color='#1f6fbf', linewidth=0.6, alpha=0.85)
    axes[1].plot(t, sig * 1e21,   color='#e05c00', linewidth=0.9, label='clean')
    axes[1].set_ylabel(r'Strain $\times10^{21}$')
    axes[1].set_title('Signal + O4 noise')
    axes[1].legend(fontsize=8, loc='upper left')
    axes[1].set_xlim(t[0], t[-1])
    axes[1].axvline(0, color='grey', linewidth=0.6, linestyle='--', alpha=0.6)

    # Bottom: O4 PSD
    mask = (f_ax > 0) & (f_psd < 1e30)   # exclude sentinel values
    axes[2].loglog(f_ax[mask], np.sqrt(f_psd[mask]), color='#2ca02c', linewidth=1.0)
    axes[2].set_xlabel('Frequency (Hz)')
    axes[2].set_ylabel('ASD (strain / \u221aHz)')
    axes[2].set_title('O4 PSD used for this noise realisation')
    axes[2].set_xlim(F_LOWER, SAMPLE_RATE / 2)

    plt.tight_layout()
    _out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'o4_noise_demo.png')
    plt.savefig(_out, dpi=150, bbox_inches='tight')
    print(f"Saved {_out}")
    try:
        plt.show()
    except Exception:
        pass
