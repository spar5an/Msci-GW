"""
test6.py — Download, whiten (with signal-excised PSD), compute SNR, and plot GW events.

For each event:
1. Download 1024s of H1+L1 data for PSD estimation (signal ±0.5s excised)
2. Download 32s centred on merger for whitening
3. Apply Tukey window, whiten using the long PSD, bandpass 35-350 Hz
4. Compute time-domain SNR (signal RMS / noise RMS in whitened data)
5. Plot whitened strain + PSDs
"""

import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from scipy.signal.windows import tukey
from scipy.signal import welch as scipy_welch, sosfiltfilt, butter

# ── Events sorted by expected loudness ──
EVENTS = [
    {'name': 'GW150914', 'gps': 1126259462.4, 'zoom': 0.6},
    {'name': 'GW170814', 'gps': 1186741861.5, 'zoom': 0.6},
    {'name': 'GW170608', 'gps': 1180922494.5, 'zoom': 0.8},
    {'name': 'GW170104', 'gps': 1167559936.6, 'zoom': 0.6},
    {'name': 'GW151226', 'gps': 1135136350.6, 'zoom': 0.8},
    {'name': 'GW190412', 'gps': 1239082262.2, 'zoom': 0.6},
    {'name': 'GW170729', 'gps': 1185389807.3, 'zoom': 0.6},
    {'name': 'GW151012', 'gps': 1128678900.4, 'zoom': 0.6},
]

SAMPLE_RATE = 4096
WINDOW = 16          # seconds each side of merger for whitening (32s)
PSD_WINDOW = 512     # seconds each side of merger for PSD (1024s)
SIGNAL_HALF = 0.5    # seconds each side of merger to excise from PSD
CROP = 1.0           # seconds to crop from each edge after filtering

# ── Bandpass (35-350 Hz, no notches) ──
sos_bp = butter(8, [35, 350], btype='bandpass', fs=SAMPLE_RATE, output='sos')


def estimate_psd(psd_data, merger_gps):
    """Estimate PSD from long data segment with signal excised."""
    arr = psd_data.value.astype(np.float64)
    n = len(arr)
    data_start_gps = psd_data.t0.value

    # Excise ±0.5s around merger
    merger_idx = int((merger_gps - data_start_gps) * SAMPLE_RATE)
    exc_half = int(SIGNAL_HALF * SAMPLE_RATE)
    exc_start = max(merger_idx - exc_half, 0)
    exc_end = min(merger_idx + exc_half, n)
    arr_excised = np.concatenate([arr[:exc_start], arr[exc_end:]])

    # Welch PSD (4s segments, 2s overlap)
    seg_len = 4 * SAMPLE_RATE
    seg_overlap = 2 * SAMPLE_RATE
    freqs_psd, psd = scipy_welch(arr_excised, fs=SAMPLE_RATE,
                                  nperseg=seg_len, noverlap=seg_overlap)
    return freqs_psd, psd


def whiten_detector(data, merger_gps, freqs_psd, psd):
    """Whiten 32s segment using pre-computed PSD, then bandpass and crop."""
    arr = data.value.astype(np.float64)
    n = len(arr)
    dt = 1.0 / SAMPLE_RATE
    data_start_gps = data.t0.value

    merger_idx = int((merger_gps - data_start_gps) * SAMPLE_RATE)

    # 1. Tukey window
    win = tukey(n, alpha=0.3)
    arr_windowed = arr * win

    # 2. Whiten in frequency domain using long PSD
    freq_data = np.fft.rfft(arr_windowed)
    freqs_fft = np.fft.rfftfreq(n, d=dt)
    psd_interp = np.interp(freqs_fft, freqs_psd, psd)
    psd_interp = np.maximum(psd_interp, 1e-60)  # clamp negatives and NaNs
    psd_interp = np.nan_to_num(psd_interp, nan=1e-60)
    whitened_freq = freq_data / np.sqrt(psd_interp)
    whitened = np.fft.irfft(whitened_freq, n=n)

    # 3. Bandpass
    whitened_bp = sosfiltfilt(sos_bp, whitened)

    # 4. Crop edges and build time array relative to merger
    crop_samp = int(CROP * SAMPLE_RATE)
    whitened_cropped = whitened_bp[crop_samp:-crop_samp]
    t_cropped = (np.arange(len(whitened_cropped)) + crop_samp) / SAMPLE_RATE
    t_rel = t_cropped - (merger_idx / SAMPLE_RATE)

    return whitened_cropped, t_rel


def compute_snr(whitened, t_rel, signal_window=0.3):
    """Compute time-domain SNR: peak |signal| / noise_std.

    signal_window: seconds each side of t=0 to consider as signal region.
    """
    signal_mask = np.abs(t_rel) <= signal_window
    noise_mask = np.abs(t_rel) > signal_window + 0.2  # gap to avoid leakage

    signal = whitened[signal_mask]
    noise = whitened[noise_mask]

    noise_std = np.std(noise)
    if noise_std == 0:
        return 0.0

    # Peak SNR: max |signal| / noise_std
    peak_snr = np.max(np.abs(signal)) / noise_std

    return peak_snr


# ── Process each event ──
results = []
for ev in EVENTS:
    name, gps, zoom = ev['name'], ev['gps'], ev['zoom']
    print(f'Downloading {name} (GPS {gps:.1f})...')

    # Long segment for PSD
    psd_start = int(gps) - PSD_WINDOW
    psd_end = int(gps) + PSD_WINDOW
    print(f'  Fetching 1024s for PSD...')
    h_psd_data = TimeSeries.fetch_open_data('H1', psd_start, psd_end, sample_rate=SAMPLE_RATE)
    l_psd_data = TimeSeries.fetch_open_data('L1', psd_start, psd_end, sample_rate=SAMPLE_RATE)

    h_freqs, h_psd = estimate_psd(h_psd_data, gps)
    l_freqs, l_psd = estimate_psd(l_psd_data, gps)

    # Short segment for whitening
    wh_start = int(gps) - WINDOW
    wh_end = int(gps) + WINDOW
    print(f'  Fetching 32s for whitening...')
    hdata = TimeSeries.fetch_open_data('H1', wh_start, wh_end, sample_rate=SAMPLE_RATE)
    ldata = TimeSeries.fetch_open_data('L1', wh_start, wh_end, sample_rate=SAMPLE_RATE)

    h_white, h_t = whiten_detector(hdata, gps, h_freqs, h_psd)
    l_white, l_t = whiten_detector(ldata, gps, l_freqs, l_psd)

    # Compute SNR
    h_snr = compute_snr(h_white, h_t)
    l_snr = compute_snr(l_white, l_t)
    print(f'  SNR: H1={h_snr:.1f}  L1={l_snr:.1f}')

    results.append({
        'name': name, 'zoom': zoom,
        'h_white': h_white, 'h_t': h_t,
        'l_white': l_white, 'l_t': l_t,
        'h_freqs': h_freqs, 'h_psd': h_psd,
        'l_freqs': l_freqs, 'l_psd': l_psd,
        'h_snr': h_snr, 'l_snr': l_snr,
    })
    print(f'  {name} done\n')

# ── Print SNR summary ──
print('=' * 60)
print(f'{"Event":20s}  {"H1 SNR":>8s}  {"L1 SNR":>8s}')
print('-' * 60)
for r in results:
    print(f'{r["name"]:20s}  {r["h_snr"]:8.1f}  {r["l_snr"]:8.1f}')
print('=' * 60)

# ── Figure 1: Whitened strain zoomed around merger ──
n_events = len(results)
fig, axes = plt.subplots(n_events, 1, figsize=(14, 3 * n_events))
fig.suptitle('Whitened LIGO strain (1024s PSD, 35-350 Hz bandpass)', fontsize=14)

for ax, r in zip(axes, results):
    zoom = r['zoom']
    ax.plot(r['h_t'], r['h_white'], label='H1', color='gwpy:ligo-hanford', linewidth=0.8)
    ax.plot(r['l_t'], r['l_white'], label='L1', color='gwpy:ligo-livingston',
            linewidth=0.8, alpha=0.7)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlim(-zoom / 2, zoom / 2)
    ax.set_title(f'{r["name"]}  (H1 SNR={r["h_snr"]:.1f}, L1 SNR={r["l_snr"]:.1f})',
                 fontsize=11)
    ax.set_ylabel('Whitened strain')
    ax.legend(loc='upper right', fontsize=8)

axes[-1].set_xlabel('Time relative to merger (s)')
plt.tight_layout()
plt.savefig('test6_three_events.png', dpi=150)
print('Saved test6_three_events.png')

# ── Figure 2: PSDs ──
fig, axes = plt.subplots(n_events, 1, figsize=(14, 3 * n_events))
fig.suptitle('Amplitude Spectral Density (1024s, signal-excised, Welch method)', fontsize=14)

for ax, r in zip(axes, results):
    ax.loglog(r['h_freqs'], np.sqrt(r['h_psd']), label='H1',
              color='gwpy:ligo-hanford', linewidth=0.8)
    ax.loglog(r['l_freqs'], np.sqrt(r['l_psd']), label='L1',
              color='gwpy:ligo-livingston', linewidth=0.8)
    ax.set_xlim(10, 2000)
    ax.set_ylim(1e-24, 1e-19)
    ax.set_title(r['name'], fontsize=11)
    ax.set_ylabel(r'ASD [strain/$\sqrt{\mathrm{Hz}}$]')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, which='both', alpha=0.3)

axes[-1].set_xlabel('Frequency (Hz)')
plt.tight_layout()
plt.savefig('test6_psds.png', dpi=150)
print('Saved test6_psds.png')
