from gwpy.timeseries import TimeSeries as GWpyTimeSeries
from JHPY import save_dataloaders, whiten_dataloaders, normalize_dataloaders, truncate_dataloaders
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import pandas as pd

# ── Config ──
SAMPLE_RATE = 4096
DOWNLOAD_WINDOW = 16  # download 16s before merger for good PSD estimation
PRE_MERGER = 1.5      # keep 1.5s before event in final output
POST_MERGER = 1.0     # keep 1.0s after event in final output
EDGE_BUFFER = 3       # extra seconds after POST_MERGER to absorb FIR edge artifacts
FINAL_DURATION = PRE_MERGER + POST_MERGER  # 2.5s total
TOTAL_DOWNLOAD = DOWNLOAD_WINDOW + POST_MERGER + EDGE_BUFFER  # 20s
DOWNLOAD_LENGTH = int(TOTAL_DOWNLOAD * SAMPLE_RATE)  # 81920 samples
DETECTORS = ['H1', 'L1']
BATCH_SIZE = 32
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1

# ── 1. Read event catalogue ──
df = pd.read_csv("gw_events_stats.csv")

# Rename pesummary columns to PyCBC/JHPY convention
rename_map = {
    'psi': 'polarization',
    'mass_1_source': 'mass1',
    'mass_2_source': 'mass2',
    'chirp_mass_source': 'chirp_mass',
    'luminosity_distance': 'distance',
    'a_1': 'spin1z',
    'a_2': 'spin2z',
    'geocent_time': 'tc',
}
df.rename(columns=rename_map, inplace=True)

# Parameter columns (everything except 'event' and 'gps_time')
skip_cols = {'event', 'gps_time'}
param_names = [c for c in df.columns if c not in skip_cols]
print(f"Parameters ({len(param_names)}): {param_names}")

# ── 2. Download 16s strain data ──
waveforms = []
params = []
events_ok = []

for idx, row in df.head(3).iterrows():
    event = row['event']
    gps = row['tc']
    start = gps - DOWNLOAD_WINDOW
    end = gps + POST_MERGER + EDGE_BUFFER

    print(f"[{idx+1}/3] {event} (GPS {gps}) — downloading {TOTAL_DOWNLOAD}s...")

    det_data = {}
    ok = True
    for det in DETECTORS:
        try:
            ts = GWpyTimeSeries.fetch_open_data(det, start, end, sample_rate=SAMPLE_RATE)
            arr = ts.value.astype(np.float32)
            if len(arr) >= DOWNLOAD_LENGTH:
                arr = arr[:DOWNLOAD_LENGTH]
            else:
                arr = np.pad(arr, (0, DOWNLOAD_LENGTH - len(arr)), mode='constant')
            det_data[det] = arr
            print(f"  {det}: OK ({len(arr)} samples)")
        except Exception as e:
            print(f"  {det}: FAILED — {e}")
            ok = False
            break

    if ok:
        waveforms.append(np.stack([det_data[d] for d in DETECTORS]))
        params.append([0.0 if p == 'tc' else (float(row[p]) if pd.notna(row[p]) else 0.0) for p in param_names])
        events_ok.append(event)

print(f"\nDownloaded {len(waveforms)}/{len(df)} events successfully.")

# ── 3. Build tensors (full 20s: 16s before + 1s after + 3s buffer) ──
X = torch.from_numpy(np.stack(waveforms))
y = torch.from_numpy(np.array(params, dtype=np.float32))
print(f"Tensors: X={X.shape}, y={y.shape}")

# ── 4. Split & create DataLoaders ──
N = len(X)
train_size = int(TRAIN_SPLIT * N)
val_size = int(VAL_SPLIT * N)
test_size = N - train_size - val_size

dataset = TensorDataset(X, y)
train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

result = {
    'train_loader': train_loader,
    'val_loader': val_loader,
    'test_loader': test_loader,
    'metadata': {
        'parameter_names': param_names,
        'num_samples': N,
        'waveform_shape': (len(DETECTORS), DOWNLOAD_LENGTH),
        'channels': DETECTORS,
        'train_size': train_size,
        'val_size': val_size,
        'test_size': test_size,
        'batch_size': BATCH_SIZE,
        'time_resolution': 1 / SAMPLE_RATE,
        'detectors': DETECTORS,
        'target_length': DOWNLOAD_LENGTH,
        'signal_length': TOTAL_DOWNLOAD,
        'source': 'GWOSC_real_events',
        'events': events_ok,
        'preprocessing': {}
    }
}

# ── 5. Whiten full signal, crop away buffer, then keep final window, then normalize ──
print("\n--- Signal Processing ---")
print("Step 1: Whiten full signal (both-side taper to prevent FIR edge artifacts)")
result_whitened = whiten_dataloaders(result, f_lower=40.0, apply_bandpass=True,
                                     apply_tukey=True, tukey_alpha=0.1, tukey_side='both')

print(f"Step 2a: Drop {EDGE_BUFFER}s right buffer (keep first {DOWNLOAD_WINDOW + POST_MERGER}s)")
result_no_buffer = truncate_dataloaders(result_whitened,
    target_duration=DOWNLOAD_WINDOW + POST_MERGER, keep_end=False)

print(f"Step 2b: Crop to last {FINAL_DURATION}s [{-PRE_MERGER}s, +{POST_MERGER}s] around merger")
result_cropped = truncate_dataloaders(result_no_buffer,
    target_duration=FINAL_DURATION, keep_end=True)

print("Step 3: Normalize")
result_processed = normalize_dataloaders(result_cropped, scale_factor=1e21)

# ── 6. Save in JHPY format ──
save_dataloaders(result_processed, 'real_events.pt')
print(f"Saved real_events.pt")

# ── 7. Extract processed waveforms for plotting ──
base_processed = result_processed['train_loader'].dataset.dataset
X_proc = base_processed.tensors[0].numpy()
FINAL_LENGTH = X_proc.shape[-1]

# ── 8. Plot: raw vs whitened+normalized ([-1.5s, +0.5s] around merger) ──
n_plot = min(3, len(waveforms))
fig, axes = plt.subplots(n_plot, 3, figsize=(18, 3 * n_plot))
if n_plot == 1:
    axes = axes[np.newaxis, :]

t_final = np.linspace(-PRE_MERGER, POST_MERGER, FINAL_LENGTH)
raw_crop = int(FINAL_DURATION * SAMPLE_RATE)
t_raw = np.linspace(-PRE_MERGER, POST_MERGER, raw_crop)

for i in range(n_plot):
    # Raw signal around merger [-1.5s, +1.0s]
    # Merger is at DOWNLOAD_WINDOW seconds into the signal
    merger_idx = int(DOWNLOAD_WINDOW * SAMPLE_RATE)
    raw_start = merger_idx - int(PRE_MERGER * SAMPLE_RATE)
    raw_end = merger_idx + int(POST_MERGER * SAMPLE_RATE)
    ax = axes[i, 0]
    for k, (det, color) in enumerate(zip(DETECTORS, ['red', 'blue'])):
        ax.plot(t_raw, waveforms[i][k][raw_start:raw_end], label=det, color=color, alpha=0.7)
    ax.set_title(f'{events_ok[i]} — Raw [{-PRE_MERGER}s, +{POST_MERGER}s]')
    ax.set_ylabel('Strain')
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5, label='merger')
    ax.legend(fontsize=8)

    # Whitened + normalized (full 2s)
    ax = axes[i, 1]
    for k, (det, color) in enumerate(zip(DETECTORS, ['red', 'blue'])):
        ax.plot(t_final, X_proc[i][k], label=det, color=color, alpha=0.7)
    ax.set_title(f'{events_ok[i]} — Whitened + Normalized')
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.legend(fontsize=8)

    # Zoomed around merger (±0.25s)
    ax = axes[i, 2]
    merger_idx = int(PRE_MERGER * SAMPLE_RATE)
    zoom_half = int(0.25 * SAMPLE_RATE)
    t_zoom = t_final[merger_idx - zoom_half:merger_idx + zoom_half]
    for k, (det, color) in enumerate(zip(DETECTORS, ['red', 'blue'])):
        ax.plot(t_zoom, X_proc[i][k][merger_idx - zoom_half:merger_idx + zoom_half],
                label=det, color=color, alpha=0.7)
    ax.set_title(f'{events_ok[i]} — Zoom (±0.25s around merger)')
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.legend(fontsize=8)

for ax in axes[-1, :]:
    ax.set_xlabel('Time relative to merger (s)')

plt.tight_layout()
plt.savefig('waveform_check.png', dpi=150)
plt.show()
print("Saved waveform_check.png")
