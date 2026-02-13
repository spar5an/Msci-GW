"""
Tests for modified (massive graviton) waveform generation pipeline.

Tests physics helpers, single waveform generation, GR limit, lambda_g scan,
DataLoader pipeline, and detector projection. Generates diagnostic plots.
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(__file__))
from JHPY import (
    _D_alpha, _additional_phase,
    _generate_single_modified_waveform,
    pycbc_modified_data_generator,
)

PLOT_DIR = os.path.join(os.path.dirname(__file__), 'test_plots')
os.makedirs(PLOT_DIR, exist_ok=True)

PASS = 0
FAIL = 0


def report(name, passed, detail=""):
    global PASS, FAIL
    status = "PASS" if passed else "FAIL"
    if not passed:
        FAIL += 1
    else:
        PASS += 1
    msg = f"  [{status}] {name}"
    if detail:
        msg += f" -- {detail}"
    print(msg)


# ─── Test 1: D_alpha ───────────────────────────────────────────────────────────
def test_D_alpha():
    print("\n=== Test 1: _D_alpha ===")

    d0 = _D_alpha(0, 0.0)
    report("D_alpha(0, z=0) == 0", abs(d0) < 1e-6, f"got {d0:.6e}")

    d1 = _D_alpha(0, 0.1)
    report("D_alpha(0, z=0.1) > 0", d1 > 0, f"got {d1:.6e} m")

    d2 = _D_alpha(0, 0.5)
    report("D_alpha monotonic: z=0.5 > z=0.1", d2 > d1,
           f"{d2:.6e} > {d1:.6e}")

    # Sanity: ~400 Mpc for z=0.1
    MPC = 3.086e22
    d_mpc = d1 / MPC
    report("D_alpha(0, 0.1) ~ hundreds of Mpc", 100 < d_mpc < 1000,
           f"{d_mpc:.1f} Mpc")


# ─── Test 2: additional_phase scaling ──────────────────────────────────────────
def test_additional_phase():
    print("\n=== Test 2: _additional_phase ===")

    freqs = np.array([100.0])
    chirp_mass = (30 * 30)**(3/5) / (30 + 30)**(1/5)

    ps1 = _additional_phase(freqs, chirp_mass, 0.1, 1e16)[0]
    ps2 = _additional_phase(freqs, chirp_mass, 0.1, 1e15)[0]
    ratio = ps2 / ps1
    report("Phase scales as 1/lambda_g^2 (ratio ~ 100)", abs(ratio - 100) < 1,
           f"ratio={ratio:.2f}")

    ps_gr = _additional_phase(freqs, chirp_mass, 0.1, 1e30)[0]
    report("GR limit (lambda_g=1e30): phase ~ 0", abs(ps_gr) < 1e-20,
           f"got {ps_gr:.6e}")

    ps_small = _additional_phase(freqs, chirp_mass, 0.1, 1e13)[0]
    report("Small lambda_g gives large phase shift", abs(ps_small) > abs(ps1),
           f"|{ps_small:.4e}| > |{ps1:.4e}|")


# ─── Test 3: single modified waveform generation ──────────────────────────────
def test_single_modified_waveform():
    print("\n=== Test 3: Single modified waveform generation ===")

    params = {
        'mass1': 30.0,
        'mass2': 30.0,
        'distance': 410.0,
        'redshift': 0.1,
    }
    target_length = 8192
    time_resolution = 1/4096

    result = _generate_single_modified_waveform(
        params, time_resolution=time_resolution,
        approximant='IMRPhenomD', f_lower=30.0,
        detectors=['H1', 'L1'], target_length=target_length,
        add_noise=False, lambda_g=1e16, f_final=2048.0
    )

    report("Generation succeeded", result['success'],
           result.get('error', ''))

    if result['success']:
        report("H1 signal present", 'H1' in result['detectors'])
        report("L1 signal present", 'L1' in result['detectors'])

        h1 = np.array(result['detectors']['H1'])
        l1 = np.array(result['detectors']['L1'])
        report(f"H1 length == {target_length}", len(h1) == target_length,
               f"got {len(h1)}")
        report(f"L1 length == {target_length}", len(l1) == target_length,
               f"got {len(l1)}")
        report("H1 signal non-zero", np.max(np.abs(h1)) > 0)
        report("L1 signal non-zero", np.max(np.abs(l1)) > 0)

    return result


# ─── Test 4: GR limit comparison ──────────────────────────────────────────────
def test_gr_limit():
    print("\n=== Test 4: GR limit (lambda_g -> large) ===")

    params = {
        'mass1': 30.0,
        'mass2': 30.0,
        'distance': 410.0,
        'redshift': 0.1,
    }
    target_length = 8192
    time_resolution = 1/4096

    # Modified with very large lambda_g (effectively GR)
    gr_result = _generate_single_modified_waveform(
        params, time_resolution=time_resolution,
        approximant='IMRPhenomD', f_lower=30.0,
        detectors=['H1', 'L1'], target_length=target_length,
        add_noise=False, lambda_g=1e30, f_final=2048.0
    )

    # Modified with large (but not extreme) lambda_g
    near_gr_result = _generate_single_modified_waveform(
        params, time_resolution=time_resolution,
        approximant='IMRPhenomD', f_lower=30.0,
        detectors=['H1', 'L1'], target_length=target_length,
        add_noise=False, lambda_g=1e20, f_final=2048.0
    )

    report("GR limit (1e30) succeeded", gr_result['success'],
           gr_result.get('error', ''))
    report("Near-GR (1e20) succeeded", near_gr_result['success'],
           near_gr_result.get('error', ''))

    if gr_result['success'] and near_gr_result['success']:
        for det in ['H1', 'L1']:
            gr_sig = np.array(gr_result['detectors'][det])
            near_sig = np.array(near_gr_result['detectors'][det])
            corr = np.corrcoef(gr_sig, near_sig)[0, 1]
            report(f"{det}: high correlation (1e30 vs 1e20)",
                   corr > 0.999, f"corr={corr:.6f}")

    return gr_result, near_gr_result


# ─── Test 5: lambda_g scan ────────────────────────────────────────────────────
def test_lambda_g_scan():
    print("\n=== Test 5: Lambda_g scan ===")

    params = {
        'mass1': 30.0,
        'mass2': 30.0,
        'distance': 410.0,
        'redshift': 0.1,
    }
    target_length = 8192
    time_resolution = 1/4096
    lambda_values = [1e13, 1e14, 1e15, 1e16, 1e17]

    # GR baseline
    gr_result = _generate_single_modified_waveform(
        params, time_resolution=time_resolution,
        approximant='IMRPhenomD', f_lower=30.0,
        detectors=['H1', 'L1'], target_length=target_length,
        add_noise=False, lambda_g=1e30, f_final=2048.0
    )

    results = {}
    for lg in lambda_values:
        r = _generate_single_modified_waveform(
            params, time_resolution=time_resolution,
            approximant='IMRPhenomD', f_lower=30.0,
            detectors=['H1', 'L1'], target_length=target_length,
            add_noise=False, lambda_g=lg, f_final=2048.0
        )
        results[lg] = r
        report(f"lambda_g={lg:.0e} succeeded", r['success'],
               r.get('error', ''))

    # Verify: smallest lambda_g produces most deviation, largest produces least
    if gr_result['success'] and all(r['success'] for r in results.values()):
        gr_h1 = np.array(gr_result['detectors']['H1'])
        diffs = {}
        for lg in lambda_values:
            h1 = np.array(results[lg]['detectors']['H1'])
            diffs[lg] = np.sqrt(np.mean((h1 - gr_h1)**2))

        lg_max = max(lambda_values)
        lg_min = min(lambda_values)
        report(f"Smallest lambda_g ({lg_min:.0e}) deviates more than largest ({lg_max:.0e})",
               diffs[lg_min] > diffs[lg_max],
               f"rms {lg_min:.0e}={diffs[lg_min]:.4e}, {lg_max:.0e}={diffs[lg_max]:.4e}")

        for lg in lambda_values:
            report(f"lambda_g={lg:.0e}: non-zero deviation from GR",
                   diffs[lg] > 0, f"rms_diff={diffs[lg]:.6e}")

    return gr_result, results


# ─── Test 6: DataLoader pipeline ──────────────────────────────────────────────
def test_dataloaders():
    print("\n=== Test 6: DataLoader pipeline ===")

    config = {
        'mass1': lambda size: np.random.uniform(25, 35, size),
        'mass2': lambda size: np.random.uniform(25, 35, size),
    }

    result = pycbc_modified_data_generator(
        config, num_samples=20, lambda_g=1e16,
        signal_length=2.0, batch_size=8, add_noise=False,
        show_progress=False
    )

    report("Result has train_loader", 'train_loader' in result)
    report("Result has val_loader", 'val_loader' in result)
    report("Result has test_loader", 'test_loader' in result)
    report("Result has metadata", 'metadata' in result)

    meta = result['metadata']
    report("metadata.lambda_g == 1e16", meta.get('lambda_g') == 1e16)
    report("metadata.modified == True", meta.get('modified') is True)
    report("metadata.f_final present", 'f_final' in meta)

    # Iterate and check shapes
    batch_X, batch_y = next(iter(result['train_loader']))
    report(f"Batch X shape: (N, 2, T)", batch_X.dim() == 3 and batch_X.shape[1] == 2,
           f"got {batch_X.shape}")
    report(f"Batch y shape: (N, params)", batch_y.dim() == 2,
           f"got {batch_y.shape}")

    target_length = int(2.0 / (1/4096))
    report(f"Waveform length == {target_length}", batch_X.shape[2] == target_length,
           f"got {batch_X.shape[2]}")

    return result


# ─── Test 7: detectors differ ─────────────────────────────────────────────────
def test_detectors_differ():
    print("\n=== Test 7: H1 vs L1 detector signals differ ===")

    params = {
        'mass1': 30.0,
        'mass2': 30.0,
        'distance': 410.0,
        'redshift': 0.1,
        'ra': 1.7,
        'dec': 0.4,
        'polarization': 0.3,
    }
    target_length = 8192
    time_resolution = 1/4096

    result = _generate_single_modified_waveform(
        params, time_resolution=time_resolution,
        approximant='IMRPhenomD', f_lower=30.0,
        detectors=['H1', 'L1'], target_length=target_length,
        add_noise=False, lambda_g=1e16, f_final=2048.0
    )

    if result['success']:
        h1 = np.array(result['detectors']['H1'])
        l1 = np.array(result['detectors']['L1'])
        diff = np.max(np.abs(h1 - l1))
        report("H1 != L1 (max abs diff > 0)", diff > 0, f"max_diff={diff:.6e}")
    else:
        report("Generation succeeded", False, result.get('error', ''))

    return result


# ─── Plots ─────────────────────────────────────────────────────────────────────
def plot_modified_vs_gr(gr_result, mod_result, lambda_g_val):
    """Plot 1: GR vs modified time-domain waveform for both detectors."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    time_resolution = 1/4096

    for idx, det in enumerate(['H1', 'L1']):
        gr_sig = np.array(gr_result['detectors'][det])
        mod_sig = np.array(mod_result['detectors'][det])
        time = np.arange(len(gr_sig)) * time_resolution

        # Zoom around merger (last 0.15s)
        peak_idx = np.argmax(np.abs(gr_sig))
        window = int(0.15 / time_resolution)
        t_start = max(0, peak_idx - window)
        t_end = min(len(gr_sig), peak_idx + window)

        axes[idx].plot(time[t_start:t_end], gr_sig[t_start:t_end],
                       label="GR", color='black', linewidth=1.5)
        axes[idx].plot(time[t_start:t_end], mod_sig[t_start:t_end],
                       label=f"Modified ($\\lambda_g = {lambda_g_val:.0e}$ m)",
                       alpha=0.7, color='tab:red')
        axes[idx].set_ylabel("Strain")
        axes[idx].set_title(f"{det} Detector")
        axes[idx].legend()

    axes[1].set_xlabel("Time (s)")
    fig.suptitle("GR vs Modified Waveform (Time Domain)", fontsize=14)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "modified_vs_gr_timedomain.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_lambda_g_scan(gr_result, scan_results):
    """Plot 2: Lambda_g scan for both detectors."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    time_resolution = 1/4096

    for idx, det in enumerate(['H1', 'L1']):
        gr_sig = np.array(gr_result['detectors'][det])
        time = np.arange(len(gr_sig)) * time_resolution

        peak_idx = np.argmax(np.abs(gr_sig))
        window = int(0.15 / time_resolution)
        t_start = max(0, peak_idx - window)
        t_end = min(len(gr_sig), peak_idx + window)

        axes[idx].plot(time[t_start:t_end], gr_sig[t_start:t_end],
                       label="GR", color='black', linewidth=1.5)

        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(scan_results)))
        for (lg, r), c in zip(sorted(scan_results.items()), colors):
            sig = np.array(r['detectors'][det])
            axes[idx].plot(time[t_start:t_end], sig[t_start:t_end],
                          label=f"$\\lambda_g = 10^{{{int(np.log10(lg))}}}$ m",
                          alpha=0.7, color=c)

        axes[idx].set_ylabel("Strain")
        axes[idx].set_title(f"{det} Detector")
        axes[idx].legend(loc='upper left', fontsize='small')

    axes[1].set_xlabel("Time (s)")
    fig.suptitle("Modified Waveforms for Different Graviton Wavelengths", fontsize=14)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "lambda_g_scan_pipeline.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_dataloader_samples(dl_result):
    """Plot 3: Sample waveforms from the DataLoader."""
    batch_X, batch_y = next(iter(dl_result['train_loader']))
    n_show = min(4, batch_X.shape[0])
    time_resolution = 1/4096

    fig, axes = plt.subplots(n_show, 2, figsize=(14, 3 * n_show), sharex=True)
    if n_show == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_show):
        time = np.arange(batch_X.shape[2]) * time_resolution
        for j, det in enumerate(['H1', 'L1']):
            sig = batch_X[i, j, :].numpy()

            # Zoom to last 0.2s where the signal lives
            peak_idx = np.argmax(np.abs(sig))
            window = int(0.2 / time_resolution)
            t_start = max(0, peak_idx - window)
            t_end = min(len(sig), peak_idx + window)

            axes[i, j].plot(time[t_start:t_end], sig[t_start:t_end], linewidth=0.8)
            axes[i, j].set_ylabel("Strain")
            if i == 0:
                axes[i, j].set_title(f"{det} Detector")

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    fig.suptitle(f"DataLoader Samples (lambda_g={dl_result['metadata']['lambda_g']:.0e} m)",
                 fontsize=14)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "dataloader_samples_modified.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("Modified Waveform Generation Tests")
    print("=" * 60)

    test_D_alpha()
    test_additional_phase()
    single_result = test_single_modified_waveform()

    gr_mod_result, near_gr_result = test_gr_limit()

    gr_result, scan_results = test_lambda_g_scan()

    dl_result = test_dataloaders()

    det_result = test_detectors_differ()

    # Generate plots
    print("\n=== Generating diagnostic plots ===")

    if single_result and single_result['success'] and gr_result and gr_result['success']:
        plot_modified_vs_gr(gr_result, single_result, 1e16)

    if gr_result and scan_results:
        plot_lambda_g_scan(gr_result, scan_results)

    if dl_result:
        plot_dataloader_samples(dl_result)

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed")
    print("=" * 60)

    if FAIL > 0:
        sys.exit(1)
