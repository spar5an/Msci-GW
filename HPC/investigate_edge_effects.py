"""
Investigate and demonstrate fixes for edge effects in whitened waveforms.

The spike at the start is caused by filter transients from the bandpass filtering.
"""

from JHPY import pycbc_data_generator
import numpy as np
import matplotlib.pyplot as plt
from pycbc.types import TimeSeries
from pycbc.psd import welch, interpolate
from pycbc.filter import highpass_fir, lowpass_fir

print("=" * 80)
print("Edge Effects Investigation")
print("=" * 80)

# Generate a test waveform
config = {
    'mass1': lambda size: np.random.uniform(20, 50, size=size),
    'mass2': lambda size: np.random.uniform(20, 50, size=size),
}

result = pycbc_data_generator(config, num_samples=5, add_noise=True, show_progress=False)
waveforms, _ = next(iter(result['train_loader']))
waveform = waveforms[0, 0, :].numpy()
delta_t = result['metadata']['time_resolution']

print(f"\nOriginal waveform: {len(waveform)} samples")

# Create multiple versions with different edge handling strategies

# Strategy 1: Current approach (with edge effects)
print("\n" + "=" * 80)
print("Strategy 1: Current Approach (shows edge effects)")
print("=" * 80)

strain = TimeSeries(waveform, delta_t=delta_t)
psd_welch = welch(strain)
psd = interpolate(psd_welch, 1.0 / strain.duration)
freq_series = strain.to_frequencyseries()
psd.resize(len(freq_series))

psd_array = np.array(psd)
epsilon = 1e-40
psd_array[psd_array <= 0] = epsilon
from pycbc.types import FrequencySeries
psd_safe = FrequencySeries(psd_array, delta_f=psd.delta_f, epoch=psd.epoch)

white_strain = (freq_series / (psd_safe ** 0.5)).to_timeseries()
white_strain_filtered = highpass_fir(white_strain, 35, 8)
white_strain_filtered = lowpass_fir(white_strain_filtered, 300, 8)

whitened_v1 = np.array(white_strain_filtered)
print(f"Peak at start (first 100 samples): {np.abs(whitened_v1[:100]).max():.2f}")
print(f"Peak in middle (samples 4000-4100): {np.abs(whitened_v1[4000:4100]).max():.2f}")

# Strategy 2: Crop edge samples
print("\n" + "=" * 80)
print("Strategy 2: Crop Edge Samples")
print("=" * 80)

crop_samples = 100  # Remove first and last 100 samples
whitened_v2 = whitened_v1[crop_samples:-crop_samples]
print(f"Cropped {crop_samples} samples from each edge")
print(f"New length: {len(whitened_v2)} samples")
print(f"Peak after cropping: {np.abs(whitened_v2[:100]).max():.2f}")

# Strategy 3: Taper/window the signal before filtering
print("\n" + "=" * 80)
print("Strategy 3: Apply Tukey Window Before Filtering")
print("=" * 80)

# Apply Tukey window (cosine taper) to reduce edge discontinuities
from scipy.signal import tukey
window_length = len(waveform)
alpha = 0.1  # Fraction of window that's tapered (10% on each side)
window = tukey(window_length, alpha=alpha)

waveform_windowed = waveform * window
strain_windowed = TimeSeries(waveform_windowed, delta_t=delta_t)

# Same whitening process
psd_welch_w = welch(strain_windowed)
psd_w = interpolate(psd_welch_w, 1.0 / strain_windowed.duration)
freq_series_w = strain_windowed.to_frequencyseries()
psd_w.resize(len(freq_series_w))

psd_array_w = np.array(psd_w)
psd_array_w[psd_array_w <= 0] = epsilon
psd_safe_w = FrequencySeries(psd_array_w, delta_f=psd_w.delta_f, epoch=psd_w.epoch)

white_strain_w = (freq_series_w / (psd_safe_w ** 0.5)).to_timeseries()
white_strain_w = highpass_fir(white_strain_w, 35, 8)
white_strain_w = lowpass_fir(white_strain_w, 300, 8)

whitened_v3 = np.array(white_strain_w)
print(f"Window: Tukey with alpha={alpha}")
print(f"Peak at start (first 100 samples): {np.abs(whitened_v3[:100]).max():.2f}")
print(f"Peak in middle (samples 4000-4100): {np.abs(whitened_v3[4000:4100]).max():.2f}")

# Strategy 4: Use longer filter order (smoother transition)
print("\n" + "=" * 80)
print("Strategy 4: Reduce Filter Order (Less Ringing)")
print("=" * 80)

white_strain_v4 = (freq_series / (psd_safe ** 0.5)).to_timeseries()
white_strain_v4 = highpass_fir(white_strain_v4, 35, 4)  # Order 4 instead of 8
white_strain_v4 = lowpass_fir(white_strain_v4, 300, 4)

whitened_v4 = np.array(white_strain_v4)
print(f"Filter order reduced from 8 to 4")
print(f"Peak at start (first 100 samples): {np.abs(whitened_v4[:100]).max():.2f}")
print(f"Peak in middle (samples 4000-4100): {np.abs(whitened_v4[4000:4100]).max():.2f}")

# Strategy 5: Disable bandpass filtering entirely
print("\n" + "=" * 80)
print("Strategy 5: No Bandpass Filtering (Only Whitening)")
print("=" * 80)

white_strain_v5 = (freq_series / (psd_safe ** 0.5)).to_timeseries()
whitened_v5 = np.array(white_strain_v5)
print(f"No bandpass applied")
print(f"Peak at start (first 100 samples): {np.abs(whitened_v5[:100]).max():.2f}")
print(f"Peak in middle (samples 4000-4100): {np.abs(whitened_v5[4000:4100]).max():.2f}")

# Create comparison plots
print("\n" + "=" * 80)
print("Creating Comparison Plots")
print("=" * 80)

fig, axes = plt.subplots(3, 2, figsize=(14, 10))
time = np.arange(len(waveform)) * delta_t
time_v2 = np.arange(len(whitened_v2)) * delta_t + (crop_samples * delta_t)

# Plot full waveforms
axes[0, 0].plot(time, whitened_v1, linewidth=0.8, alpha=0.7, label='V1: Current')
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Strain')
axes[0, 0].set_title('Strategy 1: Current (with edge effects)', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(time_v2, whitened_v2, linewidth=0.8, alpha=0.7, color='orange', label='V2: Cropped')
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('Strain')
axes[0, 1].set_title('Strategy 2: Crop Edges', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(time, whitened_v3, linewidth=0.8, alpha=0.7, color='green', label='V3: Windowed')
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('Strain')
axes[1, 0].set_title('Strategy 3: Tukey Window', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(time, whitened_v4, linewidth=0.8, alpha=0.7, color='purple', label='V4: Lower order')
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_ylabel('Strain')
axes[1, 1].set_title('Strategy 4: Filter Order 4', fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

axes[2, 0].plot(time, whitened_v5, linewidth=0.8, alpha=0.7, color='red', label='V5: No bandpass')
axes[2, 0].set_xlabel('Time (s)')
axes[2, 0].set_ylabel('Strain')
axes[2, 0].set_title('Strategy 5: No Bandpass', fontweight='bold')
axes[2, 0].legend()
axes[2, 0].grid(True, alpha=0.3)

# Zoom into first 0.2 seconds to see edge effects clearly
zoom_samples = int(0.2 / delta_t)
axes[2, 1].plot(time[:zoom_samples], whitened_v1[:zoom_samples], linewidth=1, alpha=0.7, label='Current')
axes[2, 1].plot(time[:zoom_samples], whitened_v3[:zoom_samples], linewidth=1, alpha=0.7, label='Windowed')
axes[2, 1].plot(time[:zoom_samples], whitened_v4[:zoom_samples], linewidth=1, alpha=0.7, label='Order 4')
axes[2, 1].plot(time[:zoom_samples], whitened_v5[:zoom_samples], linewidth=1, alpha=0.7, label='No bandpass')
axes[2, 1].set_xlabel('Time (s)')
axes[2, 1].set_ylabel('Strain')
axes[2, 1].set_title('Zoomed: First 0.2s Comparison', fontweight='bold')
axes[2, 1].legend()
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('edge_effects_comparison.png', dpi=150, bbox_inches='tight')
print("\nSaved: edge_effects_comparison.png")

# Recommendations
print("\n" + "=" * 80)
print("Recommendations")
print("=" * 80)

print("\n1. **Best for preserving signal**: Strategy 2 (Crop edges)")
print("   - Simple and effective")
print("   - Just remove first/last ~100 samples after whitening")
print("   - Minimal impact on merger signal (usually at center)")

print("\n2. **Best for preventing artifacts**: Strategy 3 (Tukey window)")
print("   - Prevents edge effects before they happen")
print("   - Smoothly tapers signal to zero at edges")
print("   - Good for Fourier analysis")

print("\n3. **Best for preserving frequency content**: Strategy 4 (Lower filter order)")
print("   - Less aggressive filtering = less ringing")
print("   - Still gets bandpass benefits with reduced artifacts")

print("\n4. **If spike is severe**: Strategy 5 (Disable bandpass)")
print("   - Only whiten, no bandpass filtering")
print("   - Whitening alone usually sufficient for analysis")
print("   - Set apply_bandpass=False in whiten_waveform()")

print("\n" + "=" * 80)
print("Suggested Fix")
print("=" * 80)
print("\nFor LSTM training, I recommend **Strategy 2 (Crop edges)**:")
print("  - Add crop_samples parameter to whiten_waveform()")
print("  - Remove first/last N samples after filtering")
print("  - Typically crop 50-100 samples (0.01-0.02 seconds at 4096 Hz)")
print("  - Merger signal is usually centered, so no loss of important data")
print("\nAlternatively, **Strategy 3 (Tukey window)** is cleaner but requires")
print("windowing before whitening, which changes the pipeline slightly.")
