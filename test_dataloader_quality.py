"""
Diagnostic: compare standard pycbc_data_generator vs modified (GR limit)
to check data quality from dataloaders.
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(__file__))
from JHPY import pycbc_data_generator, pycbc_modified_data_generator

PLOT_DIR = os.path.join(os.path.dirname(__file__), 'test_plots')
os.makedirs(PLOT_DIR, exist_ok=True)

# Fixed parameters (no randomness in mass/distance)
fixed_config = {
    'mass1': lambda size: np.full(size, 30.0),
    'mass2': lambda size: np.full(size, 30.0),
    'distance': lambda size: np.full(size, 410.0),
}

print("=== Generating STANDARD (TD) waveforms ===")
np.random.seed(42)
std_result = pycbc_data_generator(
    fixed_config, num_samples=5,
    approximant='IMRPhenomXP', f_lower=30.0,
    signal_length=2.0, batch_size=5, add_noise=False,
    show_progress=False
)

print("\n=== Generating MODIFIED (FD, GR limit lambda_g=1e30) waveforms ===")
np.random.seed(42)
mod_result = pycbc_modified_data_generator(
    fixed_config, num_samples=5, lambda_g=1e30,
    signal_length=2.0, batch_size=5, add_noise=False,
    show_progress=False
)

# Extract ALL data (not from shuffled loader — from base dataset)
std_base = std_result['train_loader'].dataset.dataset
mod_base = mod_result['train_loader'].dataset.dataset
X_std = std_base.tensors[0]  # (N, 2, 8192)
y_std = std_base.tensors[1]
X_mod = mod_base.tensors[0]
y_mod = mod_base.tensors[1]

print(f"\n=== Data shapes ===")
print(f"  Standard: X={X_std.shape}, y={y_std.shape}")
print(f"  Modified: X={X_mod.shape}, y={y_mod.shape}")
print(f"  Std param names: {std_result['metadata']['parameter_names']}")
print(f"  Mod param names: {mod_result['metadata']['parameter_names']}")

print(f"\n=== Waveform statistics ===")
for i in range(min(3, X_std.shape[0])):
    for ch, det in enumerate(['H1', 'L1']):
        s_std = X_std[i, ch, :].numpy()
        s_mod = X_mod[i, ch, :].numpy()
        std_peak = np.max(np.abs(s_std))
        mod_peak = np.max(np.abs(s_mod))
        std_peak_idx = np.argmax(np.abs(s_std))
        mod_peak_idx = np.argmax(np.abs(s_mod))

        # Check if signal is essentially zero
        std_nonzero = np.count_nonzero(np.abs(s_std) > std_peak * 0.001)
        mod_nonzero = np.count_nonzero(np.abs(s_mod) > mod_peak * 0.001)

        corr = np.corrcoef(s_std, s_mod)[0, 1] if std_peak > 0 and mod_peak > 0 else 0

        print(f"  Sample {i}, {det}:")
        print(f"    STD: peak={std_peak:.3e} at idx {std_peak_idx}, active_samples={std_nonzero}")
        print(f"    MOD: peak={mod_peak:.3e} at idx {mod_peak_idx}, active_samples={mod_nonzero}")
        print(f"    Correlation: {corr:.4f}, amplitude ratio: {mod_peak/std_peak:.4f}" if std_peak > 0 else "")

# Plot comparison
fig, axes = plt.subplots(4, 2, figsize=(16, 14))
dt = 1 / 4096

for col, (det, ch) in enumerate(zip(['H1', 'L1'], [0, 1])):
    s_std = X_std[0, ch, :].numpy()
    s_mod = X_mod[0, ch, :].numpy()
    time = np.arange(len(s_std)) * dt

    peak_std = np.argmax(np.abs(s_std))
    peak_mod = np.argmax(np.abs(s_mod))

    # Row 0: Full waveform overlay
    axes[0, col].plot(time, s_std, 'k-', linewidth=0.5, label='Standard (TD)')
    axes[0, col].plot(time, s_mod, 'r--', linewidth=0.5, alpha=0.7, label='Modified GR limit (FD)')
    axes[0, col].set_title(f'{det} — Full waveform')
    axes[0, col].legend(fontsize=8)
    axes[0, col].set_ylabel('Strain')

    # Row 1: Zoom on merger
    w = int(0.1 / dt)
    s1 = max(0, peak_std - w)
    e1 = min(len(s_std), peak_std + w)
    s2 = max(0, peak_mod - w)
    e2 = min(len(s_mod), peak_mod + w)

    axes[1, col].plot(time[s1:e1] - time[peak_std], s_std[s1:e1], 'k-', linewidth=1, label='Standard')
    axes[1, col].plot(time[s2:e2] - time[peak_mod], s_mod[s2:e2], 'r--', linewidth=1, alpha=0.7, label='Modified')
    axes[1, col].set_title(f'{det} — Merger zoom (aligned by peak)')
    axes[1, col].legend(fontsize=8)
    axes[1, col].set_ylabel('Strain')

    # Row 2: Ringdown
    r_start_std = peak_std
    r_end_std = min(len(s_std), peak_std + int(0.03 / dt))
    r_start_mod = peak_mod
    r_end_mod = min(len(s_mod), peak_mod + int(0.03 / dt))

    axes[2, col].plot((np.arange(r_start_std, r_end_std) - peak_std) * dt,
                      s_std[r_start_std:r_end_std], 'k-', linewidth=1, label='Standard')
    axes[2, col].plot((np.arange(r_start_mod, r_end_mod) - peak_mod) * dt,
                      s_mod[r_start_mod:r_end_mod], 'r--', linewidth=1, alpha=0.7, label='Modified')
    axes[2, col].set_title(f'{det} — Ringdown')
    axes[2, col].legend(fontsize=8)
    axes[2, col].set_ylabel('Strain')
    axes[2, col].set_xlabel('Time from merger (s)')

    # Row 3: Amplitude envelope (log scale)
    axes[3, col].semilogy(time, np.abs(s_std), 'k-', linewidth=0.5, label='Standard')
    axes[3, col].semilogy(time, np.abs(s_mod), 'r-', linewidth=0.5, alpha=0.5, label='Modified')
    axes[3, col].set_title(f'{det} — Amplitude envelope')
    axes[3, col].legend(fontsize=8)
    axes[3, col].set_ylabel('|Strain|')
    axes[3, col].set_xlabel('Time (s)')

fig.suptitle('DataLoader Output: Standard (TD) vs Modified GR-limit (FD)', fontsize=14)
plt.tight_layout()
path = os.path.join(PLOT_DIR, 'dataloader_std_vs_mod.png')
plt.savefig(path, dpi=150)
plt.close()
print(f"\nSaved: {path}")
