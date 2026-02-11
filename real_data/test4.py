"""
test4.py — Verify that test3-style outputs are compatible with JHPY's load_dataloaders.

Creates synthetic data mimicking test3's exact output structure, runs it through
save_dataloaders / load_dataloaders, and verifies round-trip compatibility.
Also tests the truncate → normalize pipeline and optionally whitening.
"""

import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
from JHPY import (save_dataloaders, load_dataloaders,
                  truncate_dataloaders, normalize_dataloaders,
                  whiten_dataloaders)

# ── 1. Constants (mirror test3.py exactly) ──
SAMPLE_RATE = 4096
DOWNLOAD_WINDOW = 16
PRE_MERGER = 1.5
POST_MERGER = 1.0
EDGE_BUFFER = 3
FINAL_DURATION = PRE_MERGER + POST_MERGER                              # 2.5
TOTAL_DOWNLOAD = DOWNLOAD_WINDOW + POST_MERGER + EDGE_BUFFER           # 20
DOWNLOAD_LENGTH = int(TOTAL_DOWNLOAD * SAMPLE_RATE)                    # 81920
FINAL_LENGTH = int(FINAL_DURATION * SAMPLE_RATE)                       # 10240
DETECTORS = ['H1', 'L1']
BATCH_SIZE = 32
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
N_EVENTS = 3
NUM_DETECTORS = 2
PARAM_NAMES = ['ra', 'dec', 'mass1', 'mass2', 'chirp_mass',
               'distance', 'chi_eff', 'redshift', 'tc']
NUM_PARAMS = len(PARAM_NAMES)
EVENTS = ['SYN150914', 'SYN151012', 'SYN151226']
RUN_WHITEN_TEST = True
SAVE_PATH = 'test_compat_synthetic.pt'
PROCESSED_PATH = 'test_compat_processed.pt'
PIPELINE_PATH = 'test_compat_full_pipeline.pt'

print("=" * 60)
print("test4.py — JHPY load_dataloaders compatibility test")
print("=" * 60)

# ── 2. Create synthetic data ──
print("\n[1] Creating synthetic data...")
torch.manual_seed(42)
X = torch.randn(N_EVENTS, NUM_DETECTORS, DOWNLOAD_LENGTH, dtype=torch.float32) * 1e-21
y = torch.tensor([
    [1.05, -1.26, 33.7, 30.4, 27.7, 477.0, -0.04, 0.10, 0.0],
    [0.97, -0.95, 21.6, 13.2, 15.5, 916.0,  0.07, 0.20, 0.0],
    [3.57,  0.94, 11.3,  7.7,  8.8, 493.0,  0.16, 0.10, 0.0],
], dtype=torch.float32)
print(f"  X={X.shape}, y={y.shape}")

# ── 3. Build result dict (replicate test3.py lines 84-117) ──
print("\n[2] Building result dict (same structure as test3)...")
N = len(X)
train_size = int(TRAIN_SPLIT * N)
val_size = int(VAL_SPLIT * N)
test_size = N - train_size - val_size
print(f"  Splits: train={train_size}, val={val_size}, test={test_size}")

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
        'parameter_names': PARAM_NAMES,
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
        'events': EVENTS,
        'preprocessing': {}
    }
}

# ── 4. Save / Load round-trip test ──
print("\n[3] Testing save_dataloaders -> load_dataloaders round-trip...")
save_dataloaders(result, SAVE_PATH)
loaded = load_dataloaders(SAVE_PATH)

# 4a. Top-level keys
assert set(loaded.keys()) == {'train_loader', 'val_loader', 'test_loader', 'metadata'}, \
    f"Unexpected keys: {loaded.keys()}"
print("  PASS: Top-level keys correct")

# 4b. Metadata round-trip
orig_meta = result['metadata']
load_meta = loaded['metadata']
for key in ['parameter_names', 'num_samples', 'waveform_shape', 'channels',
            'train_size', 'val_size', 'test_size', 'batch_size',
            'time_resolution', 'detectors', 'target_length', 'signal_length',
            'source', 'events', 'preprocessing']:
    assert key in load_meta, f"Missing metadata key: {key}"
    assert orig_meta[key] == load_meta[key], \
        f"Metadata mismatch for '{key}': {orig_meta[key]} != {load_meta[key]}"
print("  PASS: All metadata fields preserved")

# 4c. Tensor shapes and dtypes
loaded_base = loaded['train_loader'].dataset.dataset
X_loaded = loaded_base.tensors[0]
y_loaded = loaded_base.tensors[1]

assert X_loaded.shape == (N_EVENTS, NUM_DETECTORS, DOWNLOAD_LENGTH), \
    f"X shape mismatch: {X_loaded.shape}"
assert y_loaded.shape == (N_EVENTS, NUM_PARAMS), \
    f"y shape mismatch: {y_loaded.shape}"
assert X_loaded.dtype == torch.float32, f"X dtype: {X_loaded.dtype}"
assert y_loaded.dtype == torch.float32, f"y dtype: {y_loaded.dtype}"
print(f"  PASS: Tensor shapes X={X_loaded.shape}, y={y_loaded.shape}, dtype=float32")

# 4d. Tensor values match exactly
assert torch.equal(X_loaded, X), "X tensors not bit-identical after round-trip"
assert torch.equal(y_loaded, y), "y tensors not bit-identical after round-trip"
print("  PASS: Tensor values bit-identical")

# 4e. Split indices preserved
orig_train_idx = set(result['train_loader'].dataset.indices)
orig_val_idx = set(result['val_loader'].dataset.indices)
orig_test_idx = set(result['test_loader'].dataset.indices)

load_train_idx = set(loaded['train_loader'].dataset.indices)
load_val_idx = set(loaded['val_loader'].dataset.indices)
load_test_idx = set(loaded['test_loader'].dataset.indices)

assert orig_train_idx == load_train_idx, "Train indices changed"
assert orig_val_idx == load_val_idx, "Val indices changed"
assert orig_test_idx == load_test_idx, "Test indices changed"

all_idx = load_train_idx | load_val_idx | load_test_idx
assert all_idx == set(range(N_EVENTS)), f"Indices don't cover all samples: {all_idx}"
assert len(load_train_idx & load_val_idx) == 0, "Train/val overlap"
assert len(load_train_idx & load_test_idx) == 0, "Train/test overlap"
assert len(load_val_idx & load_test_idx) == 0, "Val/test overlap"
print(f"  PASS: Split indices preserved (train={load_train_idx}, val={load_val_idx}, test={load_test_idx})")

# 4f. DataLoader iteration works
for name, loader in [('train', loaded['train_loader']),
                     ('val', loaded['val_loader']),
                     ('test', loaded['test_loader'])]:
    total = 0
    for X_batch, y_batch in loader:
        assert X_batch.shape[1] == NUM_DETECTORS, f"{name}: wrong detector dim"
        assert X_batch.shape[2] == DOWNLOAD_LENGTH, f"{name}: wrong time dim"
        assert y_batch.shape[1] == NUM_PARAMS, f"{name}: wrong param dim"
        total += X_batch.shape[0]
    expected = {'train': train_size, 'val': val_size, 'test': test_size}[name]
    assert total == expected, f"{name} loader: got {total} samples, expected {expected}"
print("  PASS: All DataLoaders iterate correctly")

# 4g. batch_size override
loaded_custom = load_dataloaders(SAVE_PATH, batch_size=1)
assert loaded_custom['metadata']['batch_size'] == 1, "batch_size override failed"
print("  PASS: batch_size override works")

# ── 5. Pipeline compatibility test (two-step truncate + normalize) ──
print("\n[4] Testing two-step truncate -> normalize pipeline...")

# Step 1: Drop right buffer (keep first DOWNLOAD_WINDOW + POST_MERGER seconds)
NO_BUFFER_LENGTH = int((DOWNLOAD_WINDOW + POST_MERGER) * SAMPLE_RATE)
result_no_buffer = truncate_dataloaders(result,
    target_duration=DOWNLOAD_WINDOW + POST_MERGER, keep_end=False)
nb_base = result_no_buffer['train_loader'].dataset.dataset
X_no_buffer = nb_base.tensors[0]

assert X_no_buffer.shape == (N_EVENTS, NUM_DETECTORS, NO_BUFFER_LENGTH), \
    f"No-buffer shape: {X_no_buffer.shape}"
assert torch.equal(X_no_buffer, X[:, :, :NO_BUFFER_LENGTH]), "Buffer drop didn't keep the START"
print(f"  PASS: Buffer dropped, shape={X_no_buffer.shape}")

# Step 2: Keep final window (last FINAL_DURATION seconds)
result_cropped = truncate_dataloaders(result_no_buffer,
    target_duration=FINAL_DURATION, keep_end=True)
cropped_base = result_cropped['train_loader'].dataset.dataset
X_cropped = cropped_base.tensors[0]

assert X_cropped.shape == (N_EVENTS, NUM_DETECTORS, FINAL_LENGTH), \
    f"Cropped shape: {X_cropped.shape}, expected ({N_EVENTS}, {NUM_DETECTORS}, {FINAL_LENGTH})"
assert result_cropped['metadata']['waveform_shape'] == (NUM_DETECTORS, FINAL_LENGTH)
assert result_cropped['metadata']['preprocessing']['truncated'] is True
assert torch.equal(X_cropped, X_no_buffer[:, :, -FINAL_LENGTH:]), "Final crop didn't keep the END"
print(f"  PASS: Cropped to {X_cropped.shape}, metadata correct")

SCALE = 1e21
result_normed = normalize_dataloaders(result_cropped, scale_factor=SCALE)
normed_base = result_normed['train_loader'].dataset.dataset
X_normed = normed_base.tensors[0]

assert X_normed.shape == (N_EVENTS, NUM_DETECTORS, FINAL_LENGTH)
assert torch.allclose(X_normed, X_cropped * SCALE, rtol=1e-5), "Normalization values wrong"
assert result_normed['metadata']['preprocessing']['normalized'] is True
assert result_normed['metadata']['preprocessing']['normalize_scale'] == SCALE
print(f"  PASS: Normalized, metadata correct")

# Save/load processed data
save_dataloaders(result_normed, PROCESSED_PATH)
loaded_processed = load_dataloaders(PROCESSED_PATH)
lp_base = loaded_processed['train_loader'].dataset.dataset
X_lp = lp_base.tensors[0]

assert X_lp.shape == (N_EVENTS, NUM_DETECTORS, FINAL_LENGTH)
assert torch.equal(X_lp, X_normed), "Processed data changed after save/load"
assert loaded_processed['metadata']['preprocessing']['truncated'] is True
assert loaded_processed['metadata']['preprocessing']['normalized'] is True
print("  PASS: Processed data round-trips correctly")

# ── 6. Optional full pipeline test (whiten → two-step truncate → normalize) ──
if RUN_WHITEN_TEST:
    print("\n[5] Testing full pipeline (whiten -> drop buffer -> crop -> normalize)...")
    try:
        result_whitened = whiten_dataloaders(
            result, f_lower=40.0, apply_bandpass=True,
            apply_tukey=True, tukey_alpha=0.1, tukey_side='both',
            show_progress=False
        )

        wb = result_whitened['train_loader'].dataset.dataset
        X_w = wb.tensors[0]
        assert X_w.shape == (N_EVENTS, NUM_DETECTORS, DOWNLOAD_LENGTH)
        assert not torch.isnan(X_w).any(), "Whitening produced NaN"
        assert not torch.isinf(X_w).any(), "Whitening produced Inf"
        assert result_whitened['metadata']['preprocessing']['whitened'] is True
        print(f"  PASS: Whitened, shape={X_w.shape}, no NaN/Inf")

        # Two-step truncation (matching test3.py)
        r_nb = truncate_dataloaders(result_whitened,
            target_duration=DOWNLOAD_WINDOW + POST_MERGER, keep_end=False)
        r2 = truncate_dataloaders(r_nb,
            target_duration=FINAL_DURATION, keep_end=True)
        r3 = normalize_dataloaders(r2, scale_factor=SCALE)
        save_dataloaders(r3, PIPELINE_PATH)
        r4 = load_dataloaders(PIPELINE_PATH)

        fp_base = r4['train_loader'].dataset.dataset
        assert fp_base.tensors[0].shape == (N_EVENTS, NUM_DETECTORS, FINAL_LENGTH)
        assert torch.equal(fp_base.tensors[0],
                           r3['train_loader'].dataset.dataset.tensors[0])
        print("  PASS: Full pipeline (whiten -> drop buffer -> crop -> normalize -> save -> load)")

    except Exception as e:
        print(f"  SKIP: Whitening test failed: {e}")
else:
    print("\n[5] Skipping whitening test (RUN_WHITEN_TEST=False)")

# ── 7. Cleanup ──
print("\n[6] Cleaning up temp files...")
for f in [SAVE_PATH, PROCESSED_PATH, PIPELINE_PATH]:
    if os.path.exists(f):
        os.remove(f)
        print(f"  Removed {f}")

print("\n" + "=" * 60)
print("ALL COMPATIBILITY TESTS PASSED")
print("=" * 60)
