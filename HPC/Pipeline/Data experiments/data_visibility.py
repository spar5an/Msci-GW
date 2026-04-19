"""
data_visibility.py — per-event preprocessing / representation comparison.

For six events picked at SNR quantiles (from snr_audit.csv if present, else by
chirp-mass quantile), write one figure showing six views of the same waveform:

  (a) whitened strain (the training input) — baseline
  (b) Q-transform magnitude — time-frequency heatmap
  (c) clean whitened template overlaid on data — signal vs noise morphology
  (d) matched-filter SNR time series — "signal extraction" view
  (e) FFT magnitude (strain vs clean template) — frequency content
  (f) last-0.25 s zoom on strain + template — merger close-up

Caveat: panels (c) and (d) use the event's own true parameters to build the
template. This is legitimate for visualising what matched filtering looks like
but is NOT a viable NPE input — it leaks θ into the representation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d

from pycbc.types import FrequencySeries, TimeSeries

HERE = Path(__file__).resolve().parent
PIPELINE = HERE.parent
sys.path.insert(0, str(PIPELINE / 'Data Generation'))
from gw_datagen import _generate_single_waveform, whiten_waveform  # noqa: E402


DATASET_PATH = PIPELINE / 'Data' / 'dataset.pt'
SNR_CSV = HERE / 'outputs' / 'snr_audit.csv'
PSD_CACHE_DIR = PIPELINE / 'Data Generation' / 'o4_psd_cache'
PLOT_DIR = HERE / 'plots'

DETECTOR_INDEX = 0   # H1
DETECTOR_NAME = 'H1'
T_REF = 1126259462.4
F_LOWER_WAVEFORM = 10.0
F_LOWER_WHITEN = 20.0
APPROXIMANT = 'IMRPhenomD'
TIME_RESOLUTION = 1.0 / 4096
TARGET_LENGTH = 8192    # 2 s × 4096 Hz
SIGNAL_LENGTH_S = 2.0
MERGER_SAMPLE = TARGET_LENGTH // 2

SELECTION_QUANTILES = [0.20, 0.40, 0.60, 0.80, 0.95, 0.99]


def build_average_psd_at(delta_f: float, flen: int, detector: str) -> FrequencySeries:
    """Average cached O4 segments for this detector, interpolate onto (delta_f, flen)."""
    data = np.load(PSD_CACHE_DIR / f'o4_psds_{detector}_4096Hz.npz')
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


def pick_events(y: np.ndarray, col: dict, snr_csv_path: Path) -> list[int]:
    """Return 6 event indices spanning the SNR range (or chirp-mass range if no csv)."""
    if snr_csv_path.exists():
        df = pd.read_csv(snr_csv_path)
        print(f"  using SNR quantiles from {snr_csv_path.name}")
        return [int(df['idx'].iloc[(df['snr_network'] - df['snr_network'].quantile(q))
                                    .abs().idxmin()]) for q in SELECTION_QUANTILES]
    # fallback: chirp-mass quantiles
    print("  no snr_audit.csv — using chirp-mass quantiles")
    m1 = y[:, col['mass1']]; m2 = y[:, col['mass2']]
    chirp = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    qs = np.quantile(chirp, SELECTION_QUANTILES)
    return [int(np.argmin(np.abs(chirp - q))) for q in qs]


def make_params(y_row: np.ndarray, col: dict) -> dict:
    keys = ('mass1', 'mass2', 'spin1z', 'spin2z', 'distance',
            'inclination', 'coa_phase', 'ra', 'dec', 'polarization')
    return {k: float(y_row[col[k]]) for k in keys}


def clean_whitened_template(params: dict, psd: FrequencySeries,
                            detector: str) -> np.ndarray:
    """Run the simulation pipeline with noise disabled, then whiten."""
    result = _generate_single_waveform(
        params=params,
        time_resolution=TIME_RESOLUTION,
        approximant=APPROXIMANT,
        f_lower=F_LOWER_WAVEFORM,
        detectors=[detector],
        target_length=TARGET_LENGTH,
        add_noise=False,
    )
    if not result['success']:
        raise RuntimeError(f"template gen failed: {result.get('error')}")
    clean = result['detectors'][detector].numpy()
    w, _, _ = whiten_waveform(clean, delta_t=TIME_RESOLUTION,
                              f_lower=F_LOWER_WHITEN, psd=psd)
    return w


def matched_filter_snr(data_w: np.ndarray, template_w: np.ndarray) -> np.ndarray:
    """Cross-correlation of whitened data with whitened template, normalised so
    peak value reads out in σ-units of the off-merger matched-filter noise
    (robust to the dataset's whitening not producing exactly unit-variance noise)."""
    n = len(data_w)
    pad = np.zeros(n)
    pad[:len(template_w)] = template_w
    D = np.fft.fft(data_w)
    T = np.fft.fft(pad)
    corr = np.fft.ifft(D * np.conj(T)).real
    corr = np.fft.fftshift(corr)
    corr /= (np.sqrt(np.sum(template_w ** 2)) + 1e-30)

    # estimate off-merger noise std: mask a 200-ms window around the centre
    centre = n // 2
    half = int(0.1 / TIME_RESOLUTION)
    mask = np.ones(n, dtype=bool)
    mask[centre - half:centre + half] = False
    sigma_rho = corr[mask].std() + 1e-30
    return corr / sigma_rho


def qtransform_magnitude(strain: np.ndarray):
    """Return (times, freqs, |q_plane|) with the merger centred at t=0."""
    ts = TimeSeries(strain.astype(np.float64), delta_t=TIME_RESOLUTION)
    ts.start_time = -MERGER_SAMPLE * TIME_RESOLUTION  # so t=0 sits at merger
    times, freqs, qplane = ts.qtransform(
        delta_t=0.002,
        logfsteps=80,
        qrange=(4, 16),
        frange=(20, 300),
    )
    return times, freqs, np.abs(qplane)


def plot_event(idx: int, x_w: np.ndarray, params: dict, psd: FrequencySeries,
               snr_meta: dict | None, out_path: Path) -> None:
    print(f"  event {idx}: m1={params['mass1']:.1f} m2={params['mass2']:.1f} "
          f"d={params['distance']:.0f} Mpc")

    # time axis: 0 at merger
    t = (np.arange(TARGET_LENGTH) - MERGER_SAMPLE) * TIME_RESOLUTION

    template_w = clean_whitened_template(params, psd, DETECTOR_NAME)
    rho_centred = matched_filter_snr(x_w, template_w)

    qt_times, qt_freqs, qt_mag = qtransform_magnitude(x_w)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # (a) whitened strain
    ax = axes[0, 0]
    ax.plot(t, x_w, lw=0.5, color='tab:blue')
    ax.set_title('(a) whitened strain — training input')
    ax.set_xlabel('time from merger [s]')
    ax.set_ylabel('whitened strain')
    ax.axvline(0, color='k', lw=0.4, alpha=0.4)
    ax.grid(alpha=0.3)

    # (b) Q-transform
    ax = axes[0, 1]
    im = ax.pcolormesh(qt_times, qt_freqs, qt_mag, cmap='viridis',
                       shading='auto')
    ax.set_yscale('log')
    ax.set_title('(b) Q-transform magnitude (20–300 Hz)')
    ax.set_xlabel('time from merger [s]')
    ax.set_ylabel('frequency [Hz]')
    fig.colorbar(im, ax=ax, label='|q-coeff|')

    # (c) clean template overlaid on strain
    ax = axes[0, 2]
    ax.plot(t, x_w, lw=0.5, color='tab:blue', alpha=0.6, label='whitened data')
    ax.plot(t, template_w, lw=0.9, color='tab:red',
            label='clean whitened template')
    ax.set_title('(c) clean template vs noisy data')
    ax.set_xlabel('time from merger [s]')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(alpha=0.3)

    # (d) matched-filter SNR timeseries
    ax = axes[1, 0]
    ax.plot(t, rho_centred, lw=0.6, color='tab:green')
    peak = np.max(np.abs(rho_centred))
    ax.set_title(f'(d) matched-filter ρ(t)   peak ≈ {peak:.1f} σ')
    ax.set_xlabel('time from merger [s]')
    ax.set_ylabel('ρ / σ_off-merger')
    ax.axvline(0, color='k', lw=0.4, alpha=0.4)
    ax.grid(alpha=0.3)

    # (e) FFT magnitude
    ax = axes[1, 1]
    freqs = np.fft.rfftfreq(TARGET_LENGTH, d=TIME_RESOLUTION)
    Xw = np.abs(np.fft.rfft(x_w))
    Tw = np.abs(np.fft.rfft(template_w))
    ax.loglog(freqs, Xw + 1e-30, lw=0.6, color='tab:blue', label='data')
    ax.loglog(freqs, Tw + 1e-30, lw=0.9, color='tab:red', label='template')
    ax.set_xlim(20, 500)
    ax.set_title('(e) |FFT| of whitened series')
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel('magnitude')
    ax.axvspan(0, 35, color='k', alpha=0.05)
    ax.axvspan(300, 500, color='k', alpha=0.05)
    ax.legend(loc='lower left', fontsize=8)
    ax.grid(alpha=0.3, which='both')

    # (f) merger zoom
    ax = axes[1, 2]
    mask = (t > -0.25) & (t < 0.05)
    ax.plot(t[mask], x_w[mask], lw=0.6, color='tab:blue',
            alpha=0.6, label='data')
    ax.plot(t[mask], template_w[mask], lw=1.0, color='tab:red',
            label='template')
    ax.axvline(0, color='k', lw=0.4, alpha=0.4)
    ax.set_title('(f) merger zoom (last 0.25 s)')
    ax.set_xlabel('time from merger [s]')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(alpha=0.3)

    # title with parameters
    snr_str = f", net SNR={snr_meta['snr_network']:.1f}" if snr_meta else ""
    chirp = (params['mass1'] * params['mass2']) ** 0.6 / \
            (params['mass1'] + params['mass2']) ** 0.2
    fig.suptitle(
        f"idx {idx}  |  m1={params['mass1']:.1f}  m2={params['mass2']:.1f}  "
        f"M_c={chirp:.1f}  d_L={params['distance']:.0f} Mpc  "
        f"ι={params['inclination']:.2f}{snr_str}  —  detector {DETECTOR_NAME}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"    saved {out_path.name}")


def main():
    print(f"Loading dataset from {DATASET_PATH} …")
    d = torch.load(DATASET_PATH, weights_only=False)
    X_w = d['X_whitened'].numpy()
    y = d['y'].numpy()
    meta = d['metadata']
    col = {n: i for i, n in enumerate(meta['parameter_names'])}

    # Average PSD for whitening — 2s waveform → delta_f = 0.5 Hz, flen = 4097
    delta_f = 1.0 / SIGNAL_LENGTH_S
    flen = TARGET_LENGTH // 2 + 1
    print(f"Building average PSD ({DETECTOR_NAME}, delta_f={delta_f}, flen={flen}) …")
    psd = build_average_psd_at(delta_f, flen, DETECTOR_NAME)

    print("Selecting events …")
    idxs = pick_events(y, col, SNR_CSV)

    snr_df = pd.read_csv(SNR_CSV) if SNR_CSV.exists() else None
    snr_lookup = (snr_df.set_index('idx').to_dict('index')
                  if snr_df is not None else {})

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    for rank, i in enumerate(idxs):
        params = make_params(y[i], col)
        out = PLOT_DIR / f'representation_comparison_q{int(SELECTION_QUANTILES[rank]*100):02d}_idx{i}.png'
        try:
            plot_event(i, X_w[i, DETECTOR_INDEX], params, psd,
                       snr_lookup.get(i), out)
        except Exception as e:
            print(f"  event {i} failed: {e}")


if __name__ == '__main__':
    main()
