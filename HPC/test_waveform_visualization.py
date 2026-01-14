"""
Test script to generate a small number of waveforms and visualize them.
Creates a figure showing both detector signals for a single waveform.
"""

from JHPY import *
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for HPC
import matplotlib.pyplot as plt

print("Starting waveform generation test...")

# Configuration for test waveforms
config = {
    "mass1": lambda size: np.random.uniform(10, 50, size),
    "mass2": lambda size: np.random.uniform(10, 50, size),
    "spin1z": lambda size: np.random.uniform(-0.5, 0.5, size=size)
}

# Generate small number of waveforms for testing
print("\nGenerating 10 test waveforms...")
output = pycbc_data_generator(
    config,
    num_samples=10,
    signal_length=2.0,  # 2 seconds
    time_resolution=1/4096,  # Standard LIGO sampling
    show_progress=True,
    num_workers=4,
    detectors=['H1', 'L1']  # Hanford and Livingston
)

print("\nGeneration complete!")
print(f"Metadata: {output['metadata']}")

# Extract one sample from the training loader
train_loader = output['train_loader']
X_batch, y_batch = next(iter(train_loader))

print(f"\nBatch shapes:")
print(f"  X (signals): {X_batch.shape}")  # (batch_size, num_detectors, time_length)
print(f"  y (parameters): {y_batch.shape}")  # (batch_size, num_params)

# Get the first waveform
waveform = X_batch[0]  # Shape: (num_detectors, time_length)
params = y_batch[0]

print(f"\nFirst waveform:")
print(f"  Shape: {waveform.shape}")
print(f"  Parameters: {params}")
print(f"  Parameter names: {output['metadata']['parameter_names']}")

# Create time array
time_resolution = output['metadata']['time_resolution']
signal_length = output['metadata']['signal_length']
num_samples = output['metadata']['target_length']
time = np.arange(num_samples) * time_resolution

# Create visualization
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Plot H1 detector
axes[0].plot(time, waveform[0].numpy(), 'b-', linewidth=0.5)
axes[0].set_ylabel('Strain', fontsize=12)
axes[0].set_title('LIGO Hanford (H1)', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(0, signal_length)

# Plot L1 detector
axes[1].plot(time, waveform[1].numpy(), 'r-', linewidth=0.5)
axes[1].set_xlabel('Time (s)', fontsize=12)
axes[1].set_ylabel('Strain', fontsize=12)
axes[1].set_title('LIGO Livingston (L1)', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(0, signal_length)

# Add parameter information
param_text = '\n'.join([
    f"{name}: {params[i].item():.3f}"
    for i, name in enumerate(output['metadata']['parameter_names'])
])
fig.text(0.02, 0.98, param_text,
         transform=fig.transFigure,
         fontsize=10,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('test_waveform.png', dpi=150, bbox_inches='tight')
print("\nFigure saved as 'test_waveform.png'")

# Also create a zoomed-in view around the merger
fig2, axes2 = plt.subplots(2, 1, figsize=(12, 8))

# Zoom to last 0.2 seconds (where merger typically occurs)
zoom_time = 0.2
zoom_samples = int(zoom_time / time_resolution)
time_zoom = time[-zoom_samples:]
h1_zoom = waveform[0, -zoom_samples:].numpy()
l1_zoom = waveform[1, -zoom_samples:].numpy()

axes2[0].plot(time_zoom, h1_zoom, 'b-', linewidth=0.8)
axes2[0].set_ylabel('Strain', fontsize=12)
axes2[0].set_title('LIGO Hanford (H1) - Zoomed to Merger', fontsize=14, fontweight='bold')
axes2[0].grid(True, alpha=0.3)

axes2[1].plot(time_zoom, l1_zoom, 'r-', linewidth=0.8)
axes2[1].set_xlabel('Time (s)', fontsize=12)
axes2[1].set_ylabel('Strain', fontsize=12)
axes2[1].set_title('LIGO Livingston (L1) - Zoomed to Merger', fontsize=14, fontweight='bold')
axes2[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('test_waveform_zoom.png', dpi=150, bbox_inches='tight')
print("Zoomed figure saved as 'test_waveform_zoom.png'")

print("\n=== Test completed successfully! ===")
