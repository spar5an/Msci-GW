"""
QUICK DIAGNOSTIC: Test NPE on SINGLE parameter (mass1 only) with small dataset.
If this works, we know the architecture is fundamentally sound.
If this fails, we have a core bug to fix.
"""
import torch
import torch.nn as nn
import numpy as np
import sys
import importlib.util
from torch.utils.data import DataLoader, TensorDataset, random_split
import data_generator

# Load module with hyphen in name
spec = importlib.util.spec_from_file_location("jhpy", "JHPY_NEW_NEW_NEW-Ryzen.py")
jhpy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jhpy)

DINGOModel = jhpy.DINGOModel
train_npe_pycbc = jhpy.train_npe_pycbc

print("\n" + "="*60)
print("QUICK TEST: Single Parameter (Mass1 Only)")
print("="*60)

# Generate SMALL dataset: only 2000 samples
print("\nGenerating 2000 waveforms (SMALL test set)...")
config = {
    'mass1': lambda size: np.random.uniform(10, 50, size=size),
    'mass2': lambda size: np.random.uniform(10, 50, size=size),
    'spin1z': lambda size: np.random.uniform(-0.5, 0.5, size=size)
}

result = data_generator.pycbc_data_generator(
    config, num_samples=2000, batch_size=32, num_workers=2, allow_padding=True,
    waveform_normalization='global_standardize', parameter_normalization='zscore',
    detectors=['H1', 'L1']
)

train_loader = result['train_loader']
val_loader = result['val_loader']

# Extract ONLY mass1 from params
train_data_with_m1_only = []
for waveforms, params in train_loader:
    m1_only = params[:, :1]  # Just mass1, shape [batch, 1]
    train_data_with_m1_only.append((waveforms, m1_only))

val_data_with_m1_only = []
for waveforms, params in val_loader:
    m1_only = params[:, :1]
    val_data_with_m1_only.append((waveforms, m1_only))

# Create new loaders with mass1-only
train_loader_m1 = DataLoader(
    [(w, p) for w, p in train_data_with_m1_only],
    batch_size=32, shuffle=True
)
val_loader_m1 = DataLoader(
    [(w, p) for w, p in val_data_with_m1_only],
    batch_size=32, shuffle=False
)

# Create model for mass1 ONLY
waveform_length = result['metadata']['waveform_shape'][1]
model = DINGOModel(
    data_dim=waveform_length,
    param_dim=1,  # <-- ONLY 1 PARAMETER
    context_dim=64,
    num_flow_layers=8,  # <-- INCREASED from 6
    hidden_dim=128,
    embedding='conv1d',
    num_detectors=2,
    use_conv_preprocessing=False
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nTraining on device: {device}")
print(f"Waveform length: {waveform_length}")
print(f"Parameters: 1 (mass1 only)")
print(f"Flow layers: 8")

# Train for just 30 epochs on small dataset
history = train_npe_pycbc(
    model, train_loader_m1, val_loader=val_loader_m1,
    n_epochs=30, lr=0.0003, optimizer='adamw', device=device,
    grad_clip_norm=1.0, model_path='quick_test_m1_only.pt',
    use_mixed_precision=False, verbose=True, diagnostic_interval=5
)

# Test results
print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"Final train loss: {history['train_log_probs'][-1]:.4f}")
print(f"Best val loss: {max(history['val_log_probs']):.4f}")

# Check if loss stayed negative
if all(lp < 0.5 for lp in history['train_log_probs']):
    print("✅ GOOD: Loss stayed negative (physically reasonable)")
else:
    print("❌ BAD: Loss went positive (unphysical)")

# Check final loss
final_loss = history['train_log_probs'][-1]
if final_loss < -1.5:
    print("✅ GOOD: Model learned something (loss < -1.5)")
elif final_loss < -0.5:
    print("⚠️  PARTIAL: Some learning occurred (loss < -0.5)")
else:
    print("❌ BAD: Minimal learning (loss ≈ 0)")

print("\nInterpretation:")
print("- If you see ✅ GOOD on learning: architecture works! We need more parameters/layers")
print("- If you see ❌ BAD on learning: core bug needs fixing")
print("\n" + "="*60)
