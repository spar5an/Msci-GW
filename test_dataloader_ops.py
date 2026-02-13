"""
Test save/load cycle and downstream processing operations
(resample, truncate, whiten, normalize) on modified waveform DataLoaders.
"""
import sys
import os
import tempfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import torch

sys.path.insert(0, os.path.dirname(__file__))
from JHPY import (
    pycbc_modified_data_generator,
    save_dataloaders,
    load_dataloaders,
    truncate_dataloaders,
    whiten_dataloaders,
    normalize_dataloaders,
)

PASS = 0
FAIL = 0


def report(name, passed, detail=""):
    global PASS, FAIL
    if passed:
        PASS += 1
        print(f"  [PASS] {name}" + (f" -- {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f" -- {detail}" if detail else ""))


def extract_all_data(result):
    """Extract full X, y tensors from a dataloader result."""
    train_ds = result['train_loader'].dataset
    base = train_ds.dataset if hasattr(train_ds, 'dataset') else train_ds
    return base.tensors[0], base.tensors[1]


# ─── Generate once, reuse across tests ────────────────────────────────────────
print("=" * 60)
print("DataLoader Operations Tests (Modified Waveforms)")
print("=" * 60)

print("\n--- Generating test data ---")
config = {
    'mass1': lambda size: np.random.uniform(25, 35, size),
    'mass2': lambda size: np.random.uniform(25, 35, size),
    'distance': lambda size: np.random.uniform(300, 600, size),
}
result = pycbc_modified_data_generator(
    config, num_samples=10, lambda_g=1e16,
    signal_length=2.0, batch_size=4, add_noise=False,
    show_progress=False
)


# ─── Test 1: Save / Load round-trip ──────────────────────────────────────────
print("\n=== Test 1: Save / Load round-trip ===")
try:
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        save_path = f.name

    save_dataloaders(result, save_path)
    report("save_dataloaders succeeded", True)

    loaded = load_dataloaders(save_path, batch_size=4)
    report("load_dataloaders succeeded", True)

    # Compare data
    X_orig, y_orig = extract_all_data(result)
    X_load, y_load = extract_all_data(loaded)

    report("X shapes match", X_orig.shape == X_load.shape,
           f"orig={X_orig.shape}, loaded={X_load.shape}")
    report("y shapes match", y_orig.shape == y_load.shape,
           f"orig={y_orig.shape}, loaded={y_load.shape}")
    report("X data matches", torch.allclose(X_orig, X_load),
           f"max_diff={torch.abs(X_orig - X_load).max().item():.2e}")
    report("y data matches", torch.allclose(y_orig, y_load),
           f"max_diff={torch.abs(y_orig - y_load).max().item():.2e}")

    # Check metadata preserved
    for key in ['lambda_g', 'modified', 'f_final', 'approximant', 'parameter_names']:
        orig_val = result['metadata'].get(key)
        load_val = loaded['metadata'].get(key)
        report(f"metadata['{key}'] preserved", orig_val == load_val,
               f"orig={orig_val}, loaded={load_val}")

    # Check splits preserved
    orig_train = result['train_loader'].dataset.indices
    load_train = loaded['train_loader'].dataset.indices
    report("Train indices preserved", list(orig_train) == list(load_train))

    # Iterate loaded loader
    batch_X, batch_y = next(iter(loaded['train_loader']))
    report("Can iterate loaded train_loader",
           batch_X.shape[1] == 2 and batch_X.shape[2] == 8192,
           f"shape={batch_X.shape}")

    os.unlink(save_path)
except Exception as e:
    report(f"Save/Load round-trip", False, str(e))


# ─── Test 2: Truncate ────────────────────────────────────────────────────────
print("\n=== Test 2: Truncate DataLoaders ===")
try:
    truncated = truncate_dataloaders(result, target_length=4096)
    report("truncate_dataloaders succeeded", True)

    X_trunc, _ = extract_all_data(truncated)
    report("Truncated length == 4096", X_trunc.shape[2] == 4096,
           f"got {X_trunc.shape[2]}")
    report("Num detectors preserved", X_trunc.shape[1] == 2)
    report("Num samples preserved", X_trunc.shape[0] == 10,
           f"got {X_trunc.shape[0]}")

    # Truncated metadata updated
    report("metadata.target_length updated",
           truncated['metadata']['target_length'] == 4096)

    # Should be non-zero (merger is at the end, truncation keeps the end)
    report("Truncated data non-zero", X_trunc.abs().max().item() > 0,
           f"max={X_trunc.abs().max().item():.2e}")

    # Iterate
    batch_X, batch_y = next(iter(truncated['train_loader']))
    report("Can iterate truncated loader", batch_X.shape[2] == 4096,
           f"shape={batch_X.shape}")
except Exception as e:
    report(f"Truncate", False, str(e))


# ─── Test 3: Normalize ───────────────────────────────────────────────────────
print("\n=== Test 3: Normalize DataLoaders ===")
try:
    normalized = normalize_dataloaders(result, scale_factor=1e21)
    report("normalize_dataloaders succeeded", True)

    X_norm, _ = extract_all_data(normalized)
    X_orig, _ = extract_all_data(result)

    # Check scaling
    ratio = X_norm.abs().max().item() / X_orig.abs().max().item()
    report("Scaling factor applied correctly", abs(ratio - 1e21) / 1e21 < 0.01,
           f"ratio={ratio:.2e}, expected=1e21")

    # metadata updated
    report("metadata.preprocessing has normalize info",
           'normalize' in normalized['metadata'].get('preprocessing', {}),
           f"preprocessing={normalized['metadata'].get('preprocessing', {})}")

    # Iterate
    batch_X, batch_y = next(iter(normalized['train_loader']))
    report("Can iterate normalized loader",
           batch_X.shape == (min(4, len(result['train_loader'].dataset)), 2, 8192),
           f"shape={batch_X.shape}")
except Exception as e:
    report(f"Normalize", False, str(e))


# ─── Test 4: Whiten ──────────────────────────────────────────────────────────
print("\n=== Test 4: Whiten DataLoaders ===")
try:
    whitened = whiten_dataloaders(result, f_lower=30.0, show_progress=False)
    report("whiten_dataloaders succeeded", True)

    X_white, _ = extract_all_data(whitened)
    report("Whitened shape matches original",
           X_white.shape == X_orig.shape,
           f"shape={X_white.shape}")

    # Whitened data should differ from original
    diff = torch.abs(X_white - X_orig).max().item()
    report("Whitened != original", diff > 0, f"max_diff={diff:.2e}")

    # metadata updated
    report("metadata.preprocessing has whiten info",
           'whiten' in whitened['metadata'].get('preprocessing', {}))

    # Iterate
    batch_X, batch_y = next(iter(whitened['train_loader']))
    report("Can iterate whitened loader", batch_X.dim() == 3,
           f"shape={batch_X.shape}")
except Exception as e:
    report(f"Whiten", False, str(e))


# ─── Test 5: Chained operations ──────────────────────────────────────────────
print("\n=== Test 5: Chained operations (truncate -> normalize) ===")
try:
    step1 = truncate_dataloaders(result, target_length=4096)
    step2 = normalize_dataloaders(step1, scale_factor=1e21)
    report("Chained truncate -> normalize succeeded", True)

    X_chained, _ = extract_all_data(step2)
    report("Final shape correct", X_chained.shape[2] == 4096,
           f"shape={X_chained.shape}")

    # Iterate
    batch_X, batch_y = next(iter(step2['train_loader']))
    report("Can iterate chained loader", batch_X.shape[2] == 4096,
           f"shape={batch_X.shape}")
except Exception as e:
    report(f"Chained operations", False, str(e))


# ─── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"Results: {PASS} passed, {FAIL} failed")
print("=" * 60)
