"""
test7.py — Download, whiten, compute SNR, and plot all O4 GW events.

For each event:
1. Try to download 512s for PSD (fallback to 32s if unavailable)
2. Download 32s centred on merger for whitening
3. Tukey window, whiten using the PSD, bandpass 35-350 Hz
4. Compute time-domain SNR
5. Plot whitened strain (zoomed) + PSDs
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from gwpy.timeseries import TimeSeries
from scipy.signal.windows import tukey
from scipy.signal import welch as scipy_welch, sosfiltfilt, butter
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Load O4 events from CSV (O4 started May 2023, GPS ~1368000000) ──
df = pd.read_csv('/opt/pycbc/real_data/gw_events_stats.csv')
o4_all = df[df['geocent_time'] > 1368000000].copy()
o4_all['loudness'] = o4_all['chirp_mass_source'] / o4_all['luminosity_distance']
o4 = o4_all.sort_values('loudness', ascending=False).reset_index(drop=True)
print(f'All {len(o4)} O4 events (sorted by loudness):')

SAMPLE_RATE = 4096
WINDOW = 16          # seconds each side for whitening (32s)
PSD_WINDOW = 256     # seconds each side for PSD (512s)
SIGNAL_HALF = 0.5
CROP = 1.0
DETECTORS = ['H1', 'L1']

# ── Bandpass (35-350 Hz) ──
sos_bp = butter(8, [35, 350], btype='bandpass', fs=SAMPLE_RATE, output='sos')


def estimate_psd(psd_data, merger_gps):
    arr = psd_data.value.astype(np.float64)
    n = len(arr)
    data_start_gps = psd_data.t0.value
    merger_idx = int((merger_gps - data_start_gps) * SAMPLE_RATE)
    exc_half = int(SIGNAL_HALF * SAMPLE_RATE)
    exc_start = max(merger_idx - exc_half, 0)
    exc_end = min(merger_idx + exc_half, n)
    arr_excised = np.concatenate([arr[:exc_start], arr[exc_end:]])
    seg_len = 4 * SAMPLE_RATE
    seg_overlap = 2 * SAMPLE_RATE
    freqs_psd, psd = scipy_welch(arr_excised, fs=SAMPLE_RATE,
                                  nperseg=seg_len, noverlap=seg_overlap)
    return freqs_psd, psd


def whiten_detector(data, merger_gps, freqs_psd, psd):
    arr = data.value.astype(np.float64)
    n = len(arr)
    dt = 1.0 / SAMPLE_RATE
    data_start_gps = data.t0.value
    merger_idx = int((merger_gps - data_start_gps) * SAMPLE_RATE)

    win = tukey(n, alpha=0.3)
    arr_windowed = arr * win

    freq_data = np.fft.rfft(arr_windowed)
    freqs_fft = np.fft.rfftfreq(n, d=dt)
    psd_interp = np.interp(freqs_fft, freqs_psd, psd)
    psd_interp = np.maximum(psd_interp, 1e-60)
    psd_interp = np.nan_to_num(psd_interp, nan=1e-60)
    whitened_freq = freq_data / np.sqrt(psd_interp)
    whitened = np.fft.irfft(whitened_freq, n=n)

    whitened_bp = sosfiltfilt(sos_bp, whitened)

    crop_samp = int(CROP * SAMPLE_RATE)
    whitened_cropped = whitened_bp[crop_samp:-crop_samp]
    t_cropped = (np.arange(len(whitened_cropped)) + crop_samp) / SAMPLE_RATE
    t_rel = t_cropped - (merger_idx / SAMPLE_RATE)

    return whitened_cropped, t_rel


def compute_snr(whitened, t_rel, signal_window=0.3):
    signal_mask = np.abs(t_rel) <= signal_window
    noise_mask = np.abs(t_rel) > signal_window + 0.2
    noise = whitened[noise_mask]
    if len(noise) == 0:
        return 0.0
    noise_std = np.std(noise)
    if noise_std == 0:
        return 0.0
    return np.max(np.abs(whitened[signal_mask])) / noise_std


def has_valid_data(ts):
    """Check that a TimeSeries has real (non-NaN) data."""
    arr = ts.value
    nan_frac = np.sum(np.isnan(arr)) / len(arr)
    return nan_frac < 0.1  # allow up to 10% NaN


def fetch_psd_data(det, gps):
    """Try 512s PSD window, fall back to shorter windows if unavailable."""
    for win in [PSD_WINDOW, 128, 64, WINDOW]:
        try:
            data = TimeSeries.fetch_open_data(det, int(gps) - win, int(gps) + win,
                                               sample_rate=SAMPLE_RATE)
            if not has_valid_data(data):
                continue
            # Replace any remaining NaNs with zeros for PSD estimation
            arr = data.value
            if np.any(np.isnan(arr)):
                arr[np.isnan(arr)] = 0.0
            return data, win
        except Exception:
            continue
    return None, 0


def process_event(name, gps):
    """Process a single event: fetch data, whiten, compute SNR for all detectors."""
    log = [f'{name} (GPS {gps:.1f})...']
    det_data = {}
    for det in DETECTORS:
        # Fetch PSD data
        psd_data, psd_win = fetch_psd_data(det, gps)
        if psd_data is None:
            log.append(f'  {det}: no valid data available — skipping detector')
            continue

        freqs, psd = estimate_psd(psd_data, gps)

        # Fetch whitening data
        try:
            data = TimeSeries.fetch_open_data(det, int(gps) - WINDOW,
                                               int(gps) + WINDOW,
                                               sample_rate=SAMPLE_RATE)
            if not has_valid_data(data):
                log.append(f'  {det}: 32s data is all NaN (detector off) — skipping detector')
                continue
        except Exception:
            log.append(f'  {det}: no 32s data — skipping detector')
            continue

        white, t_rel = whiten_detector(data, gps, freqs, psd)
        snr = compute_snr(white, t_rel)

        det_data[det] = {
            'white': white, 't': t_rel,
            'freqs': freqs, 'psd': psd, 'snr': snr,
            'psd_window': psd_win * 2,
        }
        log.append(f'  {det}: SNR={snr:.1f} (PSD from {psd_win*2}s)')

    if len(det_data) == 0:
        log.append(f'  No valid detectors — skipping event')
        return {'name': name, 'gps': gps, 'detectors': None, 'log': log}

    return {'name': name, 'gps': gps, 'detectors': det_data, 'log': log}


# ── Process all O4 events (multithreaded) ──
results = []
skipped = []

with ThreadPoolExecutor(max_workers=min(20, len(o4))) as executor:
    futures = {
        executor.submit(process_event, row['event'], row['geocent_time']): idx
        for idx, row in o4.iterrows()
    }
    completed = {}
    for future in as_completed(futures):
        idx = futures[future]
        completed[idx] = future.result()

# Print logs and collect results in original order
for idx in sorted(completed):
    r = completed[idx]
    print(f'[{idx+1}/{len(o4)}] ' + '\n'.join(r['log']))
    if r['detectors'] is None:
        skipped.append(r['name'])
    else:
        results.append({'name': r['name'], 'gps': r['gps'], 'detectors': r['detectors']})

# ── SNR summary ──
print('\n' + '=' * 70)
print(f'{"Event":25s}  {"H1 SNR":>8s}  {"L1 SNR":>8s}  {"PSD window":>10s}')
print('-' * 70)
for r in results:
    h_snr = r['detectors']['H1']['snr'] if 'H1' in r['detectors'] else float('nan')
    l_snr = r['detectors']['L1']['snr'] if 'L1' in r['detectors'] else float('nan')
    dets = list(r['detectors'].keys())
    psd_w = r['detectors'][dets[0]]['psd_window']
    print(f'{r["name"]:25s}  {h_snr:8.1f}  {l_snr:8.1f}  {psd_w:8.0f}s')
print('=' * 70)
print(f'\nProcessed: {len(results)}  Skipped: {len(skipped)}')
if skipped:
    print(f'Skipped events: {", ".join(skipped)}')

# ── Figure 1: Whitened strain (zoomed ±0.3s around merger) ──
n_events = len(results)
n_cols = 10
n_rows = int(np.ceil(n_events / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2 * n_rows))
fig.suptitle('O4 events — Whitened strain (zoomed ±0.3s around merger)', fontsize=14)
axes_flat = axes.flatten()

for i, r in enumerate(results):
    ax = axes_flat[i]
    for det in DETECTORS:
        if det not in r['detectors']:
            continue
        d = r['detectors'][det]
        colour = 'gwpy:ligo-hanford' if det == 'H1' else 'gwpy:ligo-livingston'
        ax.plot(d['t'], d['white'], color=colour, linewidth=0.4, alpha=0.8)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.4, linewidth=0.5)
    ax.set_xlim(-0.3, 0.3)
    h_snr = r['detectors']['H1']['snr'] if 'H1' in r['detectors'] else float('nan')
    l_snr = r['detectors']['L1']['snr'] if 'L1' in r['detectors'] else float('nan')
    ax.set_title(f'{r["name"]}\nH1={h_snr:.1f} L1={l_snr:.1f}', fontsize=5)
    ax.tick_params(labelsize=4)
    if i % n_cols == 0:
        ax.set_ylabel('Whitened', fontsize=5)
    if i >= n_events - n_cols:
        ax.set_xlabel('t (s)', fontsize=5)

# Hide unused subplots
for i in range(n_events, len(axes_flat)):
    axes_flat[i].set_visible(False)

plt.tight_layout()
plt.savefig('test7_o4_whitened.png', dpi=250)
print('Saved test7_o4_whitened.png')

# ── Figure 2: PSDs ──
fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2 * n_rows))
fig.suptitle('O4 events — Amplitude Spectral Density', fontsize=14)
axes_flat = axes.flatten()

for i, r in enumerate(results):
    ax = axes_flat[i]
    for det in DETECTORS:
        if det not in r['detectors']:
            continue
        d = r['detectors'][det]
        colour = 'gwpy:ligo-hanford' if det == 'H1' else 'gwpy:ligo-livingston'
        ax.loglog(d['freqs'], np.sqrt(d['psd']), color=colour, linewidth=0.4)
    ax.set_xlim(10, 2000)
    ax.set_ylim(1e-24, 1e-19)
    ax.set_title(r['name'], fontsize=5)
    ax.tick_params(labelsize=4)
    ax.grid(True, which='both', alpha=0.2)
    if i % n_cols == 0:
        ax.set_ylabel(r'ASD', fontsize=5)
    if i >= n_events - n_cols:
        ax.set_xlabel('Freq (Hz)', fontsize=5)

for i in range(n_events, len(axes_flat)):
    axes_flat[i].set_visible(False)

plt.tight_layout()
plt.savefig('test7_o4_psds.png', dpi=250)
print('Saved test7_o4_psds.png')
