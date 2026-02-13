"""
Test: generate a dataloader with per-sample lambda_g (1e16 to 1e17),
save/reload, and plot to verify round-trip integrity.
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(__file__))
from JHPY import pycbc_modified_data_generator, save_dataloaders, load_dataloaders

PLOT_DIR = os.path.join(os.path.dirname(__file__), 'test_plots')
os.makedirs(PLOT_DIR, exist_ok=True)

SAVE_PATH = os.path.join(os.path.dirname(__file__), 'test_plots', 'lambda_varied.pt')

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


# ─── Config ──────────────────────────────────────────────────────────────────
CONFIG = {
    'mass1': lambda size: np.random.uniform(25, 35, size),
    'mass2': lambda size: np.random.uniform(25, 35, size),
    'distance': lambda size: np.random.uniform(300, 600, size),
    'redshift': lambda size: np.random.uniform(0.05, 0.2, size),
    'lambda_g': lambda size: np.random.uniform(1e16, 1e17, size),
}

NUM_SAMPLES = 20
BATCH_SIZE = NUM_SAMPLES  # single batch for easy inspection


# ─── Test 1: Generate with per-sample lambda_g ──────────────────────────────
def test_generate():
    print("\n=== Test 1: Generate dataloader with per-sample lambda_g ===")

    result = pycbc_modified_data_generator(
        CONFIG, num_samples=NUM_SAMPLES,
        signal_length=2.0, batch_size=BATCH_SIZE,
        add_noise=False, show_progress=False
    )

    meta = result['metadata']
    report("lambda_g_varied == True", meta.get('lambda_g_varied') is True)
    report("lambda_g arg is None in metadata", meta.get('lambda_g') is None)
    report("'lambda_g' in parameter_names",
           'lambda_g' in meta['parameter_names'])

    X, y = next(iter(result['train_loader']))
    report("X is 3D", X.dim() == 3, f"shape={X.shape}")
    report("X time dim == 8192", X.shape[2] == 8192)

    lg_idx = meta['parameter_names'].index('lambda_g')
    lg_vals = y[:, lg_idx].numpy()
    report("lambda_g values in [1e16, 1e17]",
           np.all((lg_vals >= 1e16) & (lg_vals <= 1e17)),
           f"range [{lg_vals.min():.2e}, {lg_vals.max():.2e}]")
    report("lambda_g has variation", np.std(lg_vals) > 1e14,
           f"std={np.std(lg_vals):.2e}")

    return result


# ─── Test 2: Save and reload ────────────────────────────────────────────────
def test_save_reload(result):
    print("\n=== Test 2: Save and reload dataloader ===")

    save_dataloaders(result, SAVE_PATH)
    report("File saved", os.path.exists(SAVE_PATH))

    loaded = load_dataloaders(SAVE_PATH, batch_size=BATCH_SIZE)
    report("Loaded has train_loader", 'train_loader' in loaded)
    report("Loaded has metadata", 'metadata' in loaded)

    meta_orig = result['metadata']
    meta_load = loaded['metadata']
    report("parameter_names match",
           meta_orig['parameter_names'] == meta_load['parameter_names'])
    report("lambda_g_varied preserved",
           meta_load.get('lambda_g_varied') is True)

    # Compare tensors from original and loaded
    X_orig, y_orig = next(iter(result['train_loader']))
    # Collect all samples from loaded (order may differ due to shuffling)
    # Use the underlying dataset directly for exact comparison
    orig_base = result['train_loader'].dataset.dataset
    load_base = loaded['train_loader'].dataset.dataset
    X_orig_all = orig_base.tensors[0]
    X_load_all = load_base.tensors[0]
    y_orig_all = orig_base.tensors[1]
    y_load_all = load_base.tensors[1]

    report("X tensors match exactly",
           torch.equal(X_orig_all, X_load_all),
           f"orig={X_orig_all.shape}, loaded={X_load_all.shape}")
    report("y tensors match exactly",
           torch.equal(y_orig_all, y_load_all),
           f"orig={y_orig_all.shape}, loaded={y_load_all.shape}")

    return loaded


# ─── Test 3: Plot waveforms from loaded dataloader ──────────────────────────
def test_plot(loaded):
    print("\n=== Test 3: Plot waveforms from reloaded dataloader ===")

    meta = loaded['metadata']
    lg_idx = meta['parameter_names'].index('lambda_g')

    # Gather all samples across splits
    all_X, all_y = [], []
    for loader in [loaded['train_loader'], loaded['val_loader'],
                   loaded['test_loader']]:
        for X_b, y_b in loader:
            all_X.append(X_b)
            all_y.append(y_b)
    X = torch.cat(all_X)
    y = torch.cat(all_y)

    lg_vals = y[:, lg_idx].numpy()

    # Sort by lambda_g for a nice progression
    order = np.argsort(lg_vals)
    n_show = min(6, len(order))
    # Pick evenly spaced indices from sorted order
    pick = order[np.linspace(0, len(order) - 1, n_show, dtype=int)]

    dt = 1 / 4096
    fig, axes = plt.subplots(n_show, 2, figsize=(14, 3 * n_show), sharex=True)
    if n_show == 1:
        axes = axes.reshape(1, -1)

    for row, idx in enumerate(pick):
        lg_val = lg_vals[idx]
        for col, det in enumerate(['H1', 'L1']):
            sig = X[idx, col, :].numpy()
            time = np.arange(len(sig)) * dt

            # Show full waveform
            axes[row, col].plot(time, sig, linewidth=0.6)
            axes[row, col].set_ylabel("Strain")
            if row == 0:
                axes[row, col].set_title(f"{det} Detector")
            axes[row, col].annotate(
                f"$\\lambda_g = {lg_val:.2e}$ m",
                xy=(0.02, 0.92), xycoords='axes fraction',
                fontsize=9, va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.7))

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    fig.suptitle("Reloaded DataLoader: Per-Sample $\\lambda_g$ (1e16 – 1e17 m)",
                 fontsize=14)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "lambda_varied_reloaded.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")
    report("Plot saved", os.path.exists(path))

    # Also plot a zoomed view around merger
    fig2, axes2 = plt.subplots(n_show, 2, figsize=(14, 3 * n_show), sharex=True)
    if n_show == 1:
        axes2 = axes2.reshape(1, -1)

    for row, idx in enumerate(pick):
        lg_val = lg_vals[idx]
        for col, det in enumerate(['H1', 'L1']):
            sig = X[idx, col, :].numpy()
            time = np.arange(len(sig)) * dt

            peak_idx = np.argmax(np.abs(sig))
            window = int(0.3 / dt)
            t_start = max(0, peak_idx - window)
            t_end = min(len(sig), peak_idx + window)

            axes2[row, col].plot(time[t_start:t_end], sig[t_start:t_end],
                                 linewidth=0.8)
            axes2[row, col].set_ylabel("Strain")
            if row == 0:
                axes2[row, col].set_title(f"{det} Detector")
            axes2[row, col].annotate(
                f"$\\lambda_g = {lg_val:.2e}$ m",
                xy=(0.02, 0.92), xycoords='axes fraction',
                fontsize=9, va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.7))

    axes2[-1, 0].set_xlabel("Time (s)")
    axes2[-1, 1].set_xlabel("Time (s)")
    fig2.suptitle("Reloaded DataLoader: Merger Zoom ($\\lambda_g$ varied)",
                  fontsize=14)
    plt.tight_layout()
    path2 = os.path.join(PLOT_DIR, "lambda_varied_reloaded_zoom.png")
    plt.savefig(path2, dpi=150)
    plt.close()
    print(f"  Saved: {path2}")
    report("Zoom plot saved", os.path.exists(path2))


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("Varied lambda_g DataLoader: Generate → Save → Reload → Plot")
    print("=" * 60)

    result = test_generate()
    loaded = test_save_reload(result)
    test_plot(loaded)

    # Cleanup
    if os.path.exists(SAVE_PATH):
        os.remove(SAVE_PATH)
        print(f"\n  Cleaned up {SAVE_PATH}")

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed")
    print("=" * 60)

    if FAIL > 0:
        sys.exit(1)
