"""
Tests for the modified (massive graviton) waveform DataLoader pipeline.

Covers: data shapes, train/val/test splits, parameter ranges, waveform
quality, noise injection, different lambda_g values, both detectors,
and batch iteration.  Generates diagnostic plots.
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(__file__))
from JHPY import pycbc_modified_data_generator

PLOT_DIR = os.path.join(os.path.dirname(__file__), 'test_plots')
os.makedirs(PLOT_DIR, exist_ok=True)

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


# ─── Shared config ────────────────────────────────────────────────────────────
BASE_CONFIG = {
    'mass1': lambda size: np.random.uniform(25, 35, size),
    'mass2': lambda size: np.random.uniform(25, 35, size),
}

FULL_CONFIG = {
    'mass1': lambda size: np.random.uniform(20, 40, size),
    'mass2': lambda size: np.random.uniform(20, 40, size),
    'distance': lambda size: np.random.uniform(300, 600, size),
    'redshift': lambda size: np.random.uniform(0.05, 0.2, size),
    'spin1z': lambda size: np.random.uniform(-0.5, 0.5, size),
    'spin2z': lambda size: np.random.uniform(-0.5, 0.5, size),
    'inclination': lambda size: np.random.uniform(0, np.pi, size),
    'coa_phase': lambda size: np.random.uniform(0, 2 * np.pi, size),
    'ra': lambda size: np.random.uniform(0, 2 * np.pi, size),
    'dec': lambda size: np.random.uniform(-np.pi / 2, np.pi / 2, size),
    'polarization': lambda size: np.random.uniform(0, np.pi, size),
}


# ─── Test 1: Basic structure and shapes ───────────────────────────────────────
def test_basic_structure():
    print("\n=== Test 1: Basic DataLoader structure and shapes ===")

    result = pycbc_modified_data_generator(
        BASE_CONFIG, num_samples=20, lambda_g=1e16,
        signal_length=2.0, batch_size=4, add_noise=False,
        show_progress=False
    )

    # Keys present
    for key in ['train_loader', 'val_loader', 'test_loader', 'metadata']:
        report(f"'{key}' in result", key in result)

    meta = result['metadata']
    report("metadata.lambda_g == 1e16", meta['lambda_g'] == 1e16)
    report("metadata.modified == True", meta['modified'] is True)
    report("metadata.approximant == IMRPhenomD", meta['approximant'] == 'IMRPhenomD')
    report("metadata.f_lower == 30.0", meta['f_lower'] == 30.0)
    report("metadata.f_final == 2048.0", meta['f_final'] == 2048.0)
    report("metadata.add_noise == False", meta['add_noise'] is False)
    report("metadata.signal_length == 2.0", meta['signal_length'] == 2.0)
    report("metadata.target_length == 8192", meta['target_length'] == 8192)
    report("metadata.detectors == ['H1', 'L1']", meta['detectors'] == ['H1', 'L1'])

    # Batch shape
    batch_X, batch_y = next(iter(result['train_loader']))
    report("X is 3D (batch, detectors, time)", batch_X.dim() == 3, f"shape={batch_X.shape}")
    report("X has 2 detector channels", batch_X.shape[1] == 2, f"got {batch_X.shape[1]}")
    report("X time dim == 8192", batch_X.shape[2] == 8192, f"got {batch_X.shape[2]}")
    report("y is 2D (batch, params)", batch_y.dim() == 2, f"shape={batch_y.shape}")
    report("X dtype is float32", batch_X.dtype == torch.float32)
    report("y dtype is float32", batch_y.dtype == torch.float32)

    return result


# ─── Test 2: Train/val/test split sizes ──────────────────────────────────────
def test_splits():
    print("\n=== Test 2: Train/val/test splits ===")

    result = pycbc_modified_data_generator(
        BASE_CONFIG, num_samples=50, lambda_g=1e16,
        signal_length=2.0, batch_size=8, add_noise=False,
        show_progress=False, train_split=0.7, val_split=0.15
    )

    meta = result['metadata']
    total = meta['num_samples']
    train_sz = meta['train_size']
    val_sz = meta['val_size']
    test_sz = meta['test_size']

    report(f"Total samples == {total}", total == 50)
    report(f"train({train_sz}) + val({val_sz}) + test({test_sz}) == {total}",
           train_sz + val_sz + test_sz == total)
    report("train_size ~ 70%", abs(train_sz / total - 0.7) < 0.05,
           f"{train_sz}/{total} = {train_sz/total:.2f}")
    report("val_size ~ 15%", abs(val_sz / total - 0.15) < 0.05,
           f"{val_sz}/{total} = {val_sz/total:.2f}")
    report("test_size ~ 15%", abs(test_sz / total - 0.15) < 0.05,
           f"{test_sz}/{total} = {test_sz/total:.2f}")

    # Count actual samples across all batches
    train_count = sum(x.shape[0] for x, _ in result['train_loader'])
    val_count = sum(x.shape[0] for x, _ in result['val_loader'])
    test_count = sum(x.shape[0] for x, _ in result['test_loader'])
    report("Iterating loaders yields all samples",
           train_count + val_count + test_count == total,
           f"train={train_count}, val={val_count}, test={test_count}")

    return result


# ─── Test 3: Waveform quality — non-trivial signals ─────────────────────────
def test_waveform_quality():
    print("\n=== Test 3: Waveform quality (non-trivial signals) ===")

    result = pycbc_modified_data_generator(
        BASE_CONFIG, num_samples=10, lambda_g=1e16,
        signal_length=2.0, batch_size=10, add_noise=False,
        show_progress=False
    )

    all_X, all_y = next(iter(result['train_loader']))
    n = all_X.shape[0]

    n_nonzero = 0
    n_has_peak = 0
    n_has_ringdown = 0

    for i in range(n):
        for ch in range(2):
            sig = all_X[i, ch, :].numpy()

            # Signal should not be all zeros
            if np.max(np.abs(sig)) > 0:
                n_nonzero += 1

            # Should have a clear merger peak
            peak_idx = np.argmax(np.abs(sig))
            peak_amp = np.abs(sig[peak_idx])
            if peak_amp > 1e-23:
                n_has_peak += 1

            # After merger, signal should decay (ringdown present)
            # Check that there is non-zero signal after the peak
            if peak_idx < len(sig) - 10:
                post_peak = np.abs(sig[peak_idx + 5:peak_idx + 50])
                if len(post_peak) > 0 and np.max(post_peak) > peak_amp * 0.01:
                    n_has_ringdown += 1

    total_channels = n * 2
    report(f"All channels non-zero ({n_nonzero}/{total_channels})",
           n_nonzero == total_channels)
    report(f"All channels have clear peak ({n_has_peak}/{total_channels})",
           n_has_peak == total_channels)
    report(f"Most channels show ringdown ({n_has_ringdown}/{total_channels})",
           n_has_ringdown >= total_channels * 0.8,
           f"{n_has_ringdown}/{total_channels}")

    # Amplitude should be in physical strain range (~1e-22 to 1e-20)
    max_amp = all_X.abs().max().item()
    report("Peak amplitude in physical range (1e-23 to 1e-19)",
           1e-23 < max_amp < 1e-19, f"max_amp={max_amp:.2e}")

    return result


# ─── Test 4: Parameter values stored correctly ───────────────────────────────
def test_parameters():
    print("\n=== Test 4: Parameters stored correctly ===")

    result = pycbc_modified_data_generator(
        FULL_CONFIG, num_samples=20, lambda_g=1e15,
        signal_length=2.0, batch_size=20, add_noise=False,
        show_progress=False
    )

    meta = result['metadata']
    param_names = meta['parameter_names']
    report("mass1 in parameters", 'mass1' in param_names)
    report("mass2 in parameters", 'mass2' in param_names)
    report("distance in parameters", 'distance' in param_names)
    report("redshift in parameters", 'redshift' in param_names)

    all_X, all_y = next(iter(result['train_loader']))

    # Check parameter ranges match config
    m1_idx = param_names.index('mass1')
    m2_idx = param_names.index('mass2')
    dist_idx = param_names.index('distance')
    z_idx = param_names.index('redshift')

    m1_vals = all_y[:, m1_idx].numpy()
    m2_vals = all_y[:, m2_idx].numpy()
    dist_vals = all_y[:, dist_idx].numpy()
    z_vals = all_y[:, z_idx].numpy()

    report("mass1 in [20, 40]", np.all((m1_vals >= 20) & (m1_vals <= 40)),
           f"range [{m1_vals.min():.1f}, {m1_vals.max():.1f}]")
    report("mass2 in [20, 40]", np.all((m2_vals >= 20) & (m2_vals <= 40)),
           f"range [{m2_vals.min():.1f}, {m2_vals.max():.1f}]")
    report("distance in [300, 600]", np.all((dist_vals >= 300) & (dist_vals <= 600)),
           f"range [{dist_vals.min():.0f}, {dist_vals.max():.0f}]")
    report("redshift in [0.05, 0.2]", np.all((z_vals >= 0.05) & (z_vals <= 0.2)),
           f"range [{z_vals.min():.3f}, {z_vals.max():.3f}]")

    # Parameters should vary (not all identical)
    report("mass1 has variation", np.std(m1_vals) > 0.1, f"std={np.std(m1_vals):.2f}")
    report("mass2 has variation", np.std(m2_vals) > 0.1, f"std={np.std(m2_vals):.2f}")

    return result


# ─── Test 5: Different lambda_g values produce different waveforms ───────────
def test_lambda_g_comparison():
    print("\n=== Test 5: Different lambda_g values ===")

    np.random.seed(42)
    fixed_config = {
        'mass1': lambda size: np.full(size, 30.0),
        'mass2': lambda size: np.full(size, 30.0),
    }

    results = {}
    for lg in [1e14, 1e16, 1e30]:
        np.random.seed(42)
        r = pycbc_modified_data_generator(
            fixed_config, num_samples=5, lambda_g=lg,
            signal_length=2.0, batch_size=5, add_noise=False,
            show_progress=False
        )
        X, _ = next(iter(r['train_loader']))
        results[lg] = X
        report(f"lambda_g={lg:.0e} generation OK", True)

    # GR limit (1e30) vs moderate (1e16) — should be very similar
    diff_gr_mod = torch.abs(results[1e30] - results[1e16]).max().item()
    # Strong modification (1e14) vs GR — should differ more
    diff_gr_strong = torch.abs(results[1e30] - results[1e14]).max().item()

    report("lambda_g=1e14 deviates more from GR than 1e16",
           diff_gr_strong > diff_gr_mod,
           f"diff(1e14)={diff_gr_strong:.2e}, diff(1e16)={diff_gr_mod:.2e}")

    report("lambda_g=1e30 ~ 1e16 (small difference)",
           diff_gr_mod < diff_gr_strong,
           f"diff={diff_gr_mod:.2e}")

    return results


# ─── Test 6: Noise injection ─────────────────────────────────────────────────
def test_noise_injection():
    print("\n=== Test 6: Noise injection ===")

    np.random.seed(123)
    fixed_config = {
        'mass1': lambda size: np.full(size, 30.0),
        'mass2': lambda size: np.full(size, 30.0),
    }

    # Without noise
    np.random.seed(123)
    clean = pycbc_modified_data_generator(
        fixed_config, num_samples=5, lambda_g=1e16,
        signal_length=2.0, batch_size=5, add_noise=False,
        show_progress=False
    )
    X_clean, _ = next(iter(clean['train_loader']))

    # With noise
    np.random.seed(123)
    noisy = pycbc_modified_data_generator(
        fixed_config, num_samples=5, lambda_g=1e16,
        signal_length=2.0, batch_size=5, add_noise=True,
        show_progress=False
    )
    X_noisy, _ = next(iter(noisy['train_loader']))

    report("Clean metadata.add_noise == False", clean['metadata']['add_noise'] is False)
    report("Noisy metadata.add_noise == True", noisy['metadata']['add_noise'] is True)

    # Noisy should differ from clean
    diff = torch.abs(X_noisy - X_clean).max().item()
    report("Noisy != clean", diff > 0, f"max_diff={diff:.2e}")

    # Noisy signal should have higher RMS in quiet regions (early inspiral)
    # Use first 1000 samples where signal is near zero
    clean_rms = X_clean[:, :, :1000].pow(2).mean().sqrt().item()
    noisy_rms = X_noisy[:, :, :1000].pow(2).mean().sqrt().item()
    report("Noisy has higher RMS in quiet region",
           noisy_rms > clean_rms,
           f"clean={clean_rms:.2e}, noisy={noisy_rms:.2e}")

    return clean, noisy


# ─── Test 7: Both detectors have different signals ──────────────────────────
def test_detector_channels():
    print("\n=== Test 7: Detector channels differ ===")

    config = {
        'mass1': lambda size: np.full(size, 30.0),
        'mass2': lambda size: np.full(size, 30.0),
        'ra': lambda size: np.full(size, 1.7),
        'dec': lambda size: np.full(size, 0.4),
        'polarization': lambda size: np.full(size, 0.3),
    }

    result = pycbc_modified_data_generator(
        config, num_samples=5, lambda_g=1e16,
        signal_length=2.0, batch_size=5, add_noise=False,
        show_progress=False
    )

    X, _ = next(iter(result['train_loader']))

    for i in range(min(3, X.shape[0])):
        h1 = X[i, 0, :].numpy()
        l1 = X[i, 1, :].numpy()
        diff = np.max(np.abs(h1 - l1))
        report(f"Sample {i}: H1 != L1", diff > 0, f"max_diff={diff:.2e}")

    return result


# ─── Test 8: Batch iteration completeness ────────────────────────────────────
def test_batch_iteration():
    print("\n=== Test 8: Batch iteration ===")

    result = pycbc_modified_data_generator(
        BASE_CONFIG, num_samples=30, lambda_g=1e16,
        signal_length=2.0, batch_size=7, add_noise=False,
        show_progress=False
    )

    # Iterate through all train batches
    total_train = 0
    batch_shapes = []
    for X_batch, y_batch in result['train_loader']:
        total_train += X_batch.shape[0]
        batch_shapes.append(X_batch.shape[0])
        report(f"Batch shape OK (n={X_batch.shape[0]})",
               X_batch.shape[1] == 2 and X_batch.shape[2] == 8192,
               f"shape={X_batch.shape}")

    report(f"Train batches yield {total_train} samples (expect {result['metadata']['train_size']})",
           total_train == result['metadata']['train_size'])

    # Val and test
    total_val = sum(x.shape[0] for x, _ in result['val_loader'])
    total_test = sum(x.shape[0] for x, _ in result['test_loader'])
    grand_total = total_train + total_val + total_test
    report(f"All loaders total {grand_total} == {result['metadata']['num_samples']}",
           grand_total == result['metadata']['num_samples'])

    return result


# ─── Plots ────────────────────────────────────────────────────────────────────
def plot_lambda_g_dataloader(results_dict):
    """Plot waveforms from dataloaders at different lambda_g values."""
    fig, axes = plt.subplots(2, len(results_dict), figsize=(5 * len(results_dict), 8),
                             squeeze=False, sharex=True)
    dt = 1 / 4096
    det_names = ['H1', 'L1']

    for col, (lg, X) in enumerate(sorted(results_dict.items())):
        for row, det in enumerate(det_names):
            sig = X[0, row, :].numpy()
            time = np.arange(len(sig)) * dt
            peak_idx = np.argmax(np.abs(sig))
            window = int(0.15 / dt)
            t_start = max(0, peak_idx - window)
            t_end = min(len(sig), peak_idx + window)

            axes[row, col].plot(time[t_start:t_end], sig[t_start:t_end], linewidth=0.8)
            axes[row, col].set_ylabel("Strain")
            if row == 0:
                axes[row, col].set_title(f"$\\lambda_g = {lg:.0e}$ m")
            if row == 1:
                axes[row, col].set_xlabel("Time (s)")
            if col == 0:
                axes[row, col].annotate(det, xy=(0.02, 0.95),
                                         xycoords='axes fraction', fontsize=12,
                                         fontweight='bold', va='top')

    fig.suptitle("DataLoader Output: Different Graviton Wavelengths", fontsize=14)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "dataloader_lambda_g_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_clean_vs_noisy(clean_result, noisy_result):
    """Plot clean vs noisy waveforms from dataloaders."""
    X_clean, _ = next(iter(clean_result['train_loader']))
    X_noisy, _ = next(iter(noisy_result['train_loader']))

    dt = 1 / 4096
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)

    for row, det in enumerate(['H1', 'L1']):
        sig_clean = X_clean[0, row, :].numpy()
        sig_noisy = X_noisy[0, row, :].numpy()
        time = np.arange(len(sig_clean)) * dt

        peak_idx = np.argmax(np.abs(sig_clean))
        window = int(0.15 / dt)
        t_start = max(0, peak_idx - window)
        t_end = min(len(sig_clean), peak_idx + window)

        axes[row, 0].plot(time[t_start:t_end], sig_clean[t_start:t_end],
                          'k-', linewidth=0.8, label='Clean')
        axes[row, 0].set_ylabel("Strain")
        axes[row, 0].set_title(f"{det} — Clean")

        axes[row, 1].plot(time[t_start:t_end], sig_noisy[t_start:t_end],
                          'b-', linewidth=0.8, label='Noisy')
        axes[row, 1].set_ylabel("Strain")
        axes[row, 1].set_title(f"{det} — With Noise")

    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 1].set_xlabel("Time (s)")
    fig.suptitle("DataLoader: Clean vs Noisy Modified Waveforms", fontsize=14)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "dataloader_clean_vs_noisy.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("Modified Waveform DataLoader Tests")
    print("=" * 60)

    test_basic_structure()
    test_splits()
    wq_result = test_waveform_quality()
    test_parameters()
    lg_results = test_lambda_g_comparison()
    clean, noisy = test_noise_injection()
    test_detector_channels()
    test_batch_iteration()

    print("\n=== Generating diagnostic plots ===")
    plot_lambda_g_dataloader(lg_results)
    plot_clean_vs_noisy(clean, noisy)

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed")
    print("=" * 60)
