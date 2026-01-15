# The lstm models are currently having issues, I am going to rebuild it in order to try and fix this

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from JHPY import *
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC
import matplotlib.pyplot as plt

# Load data without noise
output = load_dataloaders("data_noise.pt")

train_loader = output["train_loader"]
val_loader = output["val_loader"]
test_loader = output["test_loader"]
metadata = output["metadata"]  # Fixed typo: should be "metadata" not "meta_data"

# Just going to test this on a dingo model before starting on lstm
model = create_dingo_from_data(metadata)

optim = torch.optim.Adam(model.parameters(), lr=5e-4)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optim, mode='max', factor=0.5, patience=6
)
training_stuff = train_npe_model(model, optim, 10, train_loader, val_loader, patience=15, scheduler=sched, model_path='best_dingo_model_noise.pt')

print("\n=== Training complete ===")
print(f"Best validation log prob: {training_stuff['best_val_log_prob']:.4f}")
print(f"Best epoch: {training_stuff['best_val_epoch']}")

# Plot training history
fig_history, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(training_stuff['train_log_probs'], label='Training', linewidth=2)
ax1.plot(training_stuff['val_log_probs'], label='Validation', linewidth=2)
ax1.axvline(training_stuff['best_val_epoch'], color='red', linestyle='--', label='Best epoch', linewidth=1.5)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Log Probability', fontsize=12)
ax1.set_title('Training History', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot validation log prob only (zoomed)
ax2.plot(training_stuff['val_log_probs'], label='Validation', linewidth=2, color='orange')
ax2.axvline(training_stuff['best_val_epoch'], color='red', linestyle='--', label='Best epoch', linewidth=1.5)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Validation Log Probability', fontsize=12)
ax2.set_title('Validation Performance', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history_no_noise.png', dpi=150, bbox_inches='tight')
print("\nTraining history saved as 'training_history_no_noise.png'")
plt.close()

from matplotlib.patches import Rectangle  # Rectangle where both axes' 1-sigma regions overlap

# Load best model
model, _ = load_npe("best_dingo_model_noise.pt")

# Get parameter names from metadata
param_names = metadata['parameter_names']
num_params = len(param_names)

print(f"\n=== Testing NPE model ===")
print(f"Parameters in dataset: {param_names}")

# Extract test samples from test_loader
test_iter = iter(test_loader)
X_test, y_test = next(test_iter)

# Take first few samples for testing
num_test_samples = min(3, len(X_test))

fig, axes = plt.subplots(num_params + 1, num_test_samples, figsize=(8*num_test_samples, 6*(num_params+1)))
if num_test_samples == 1:
    axes = axes.reshape(-1, 1)  # Ensure 2D array

print(f"Testing on {num_test_samples} samples from test set:\n")

for idx in range(num_test_samples):
    # Get true parameters
    true_params = y_test[idx].numpy()
    observed_signal = X_test[idx]  # Shape: (num_detectors, time_length)

    print(f"\nTest {idx+1}:")
    for i, name in enumerate(param_names):
        print(f"  True {name}: {true_params[i]:.3f}")

    # Run inference using the model directly
    model.eval()
    with torch.no_grad():
        # observed_signal shape: (num_detectors, time_length)
        # Need to add batch dimension: (1, num_detectors, time_length)
        data_tensor = observed_signal.unsqueeze(0)
        posterior_samples = model.sample_posterior(data_tensor, num_samples=5000)
        posterior_samples = posterior_samples.numpy()  # Shape: (5000, num_params)

    # Compute statistics for each parameter
    stats_list = []
    samples_list = []

    for i in range(num_params):
        param_samples = posterior_samples[:, i]
        samples_list.append(param_samples)

        stats = {
            'mean': np.mean(param_samples),
            'median': np.median(param_samples),
            'std': np.std(param_samples),
            'q05': np.percentile(param_samples, 5),
            'q95': np.percentile(param_samples, 95),
        }
        stats_list.append(stats)

        print(f"  {param_names[i]} posterior: mean={stats['mean']:.3f} ± {stats['std']:.3f}")

    # Plot waveform for both detectors
    detector_names = metadata['detectors']
    time_resolution = metadata['time_resolution']
    signal_length = metadata['signal_length']

    time_array = np.arange(observed_signal.shape[1]) * time_resolution

    for det_idx in range(len(detector_names)):
        if det_idx == 0:
            ax = axes[0, idx]
        else:
            # If more than one detector, could add additional rows, for now just plot first detector
            ax = axes[0, idx]

        ax.plot(time_array, observed_signal[det_idx].numpy(), 'b-', alpha=0.7, linewidth=0.5, label=f'{detector_names[det_idx]}')

    axes[0, idx].set_title(f'Test {idx+1}: ' + ', '.join([f'{name}={true_params[i]:.2f}' for i, name in enumerate(param_names)]),
                          fontsize=10, fontweight='bold')
    axes[0, idx].set_xlabel('Time (s)', fontsize=10)
    axes[0, idx].set_ylabel('Strain', fontsize=10)
    axes[0, idx].legend(fontsize=8)
    axes[0, idx].grid(True, alpha=0.3)

    # Plot 1D posteriors for each parameter
    for param_idx in range(num_params):
        ax = axes[param_idx + 1, idx]

        ax.hist(samples_list[param_idx], bins=50, density=True,
                alpha=0.6, edgecolor='black', label='Posterior')
        ax.axvline(true_params[param_idx], color='red', linestyle='--',
                   linewidth=2.5, label=f'True: {true_params[param_idx]:.2f}', zorder=10)
        ax.axvline(stats_list[param_idx]['mean'], color='green', linestyle='-',
                   linewidth=2.5, label=f"Mean: {stats_list[param_idx]['mean']:.2f}", zorder=10)
        ax.axvspan(stats_list[param_idx]['q05'], stats_list[param_idx]['q95'],
                   alpha=0.2, color='gray', label='90% CI')
        ax.set_title(f'p({param_names[param_idx]} | data)', fontsize=10, fontweight='bold')
        ax.set_xlabel(param_names[param_idx], fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('posterior_analysis_no_noise.png', dpi=150, bbox_inches='tight')
print("\n\nPosterior analysis saved as 'posterior_analysis_no_noise.png'")
plt.close()

print("\n=== All plots saved successfully ===")