"""
Multi-event GW comparison: Real data vs Simulated waveforms, H1 vs L1.

Processes multiple high-SNR BBH events from GWOSC, compares each against
a simulated waveform using IMRPhenomXP, and compares H1 vs L1 signals.

Pipeline replicates the exact processing from Real Data/test_real_data.py
create_real_vs_simulated_comparison() (lines 457-714), extended to
both detectors and multiple events.

Events processed:
    GW150914 - First detection, highest SNR (~25.1)
    GW151226 - Second detection, lower mass BBH (~13.1)
    GW170104 - O2 run, intermediate mass (~13.0)
    GW170608 - Low mass BBH, high SNR (~14.9)
    GW170814 - First 3-detector event (~18.3)

Usage:
    python test_multi_event_comparison.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import json
import time as time_mod

# Add paths for imports (same convention as other test files)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Real Data'))

# PyCBC imports (same set used by create_real_vs_simulated_comparison)
from pycbc.waveform import get_td_waveform
from pycbc.types import TimeSeries, FrequencySeries
from pycbc.detector import Detector
from pycbc.psd import welch, interpolate
from pycbc.filter import highpass_fir, lowpass_fir
from scipy.signal import correlate

# Real Data module imports
from catalog_utils import get_event_parameters, get_event_summary
from real_data_loader import load_real_event


# ============================================================
# Configuration
# ============================================================

EVENTS = [
    'GW150914',   # M1~35.6, M2~30.6, D~410 Mpc, SNR~25.1
    'GW151226',   # M1~14.2, M2~7.5,  D~440 Mpc, SNR~13.1
    'GW170104',   # M1~31.2, M2~19.4, D~880 Mpc, SNR~13.0
    'GW170608',   # M1~12.0, M2~7.6,  D~340 Mpc, SNR~14.9
    'GW170814',   # M1~30.5, M2~25.3, D~540 Mpc, SNR~18.3
]

DETECTORS = ['H1', 'L1']
SAMPLE_RATE = 4096
DELTA_T = 1.0 / SAMPLE_RATE
DATA_DURATION = 32.0
BANDPASS_LOW = 35.0
BANDPASS_HIGH = 300.0
F_LOWER = 20.0
APPROXIMANT = 'IMRPhenomXP'

# Grid search parameters
INC_VALUES_DEFAULT = [np.pi/8, np.pi/6, np.pi/4, np.pi/3]
N_PHASE_STEPS = 16

# Plot zoom windows (seconds relative to merger)
ZOOM_WIDE = (-0.5, 0.1)
ZOOM_MERGER = (-0.15, 0.05)
ALIGN_WINDOW = (-0.5, 0.1)
CORRELATION_WINDOW = (-0.2, 0.05)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# Core Processing Functions
# (exact replicas of test_real_data.py create_real_vs_simulated_comparison)
# ============================================================

def whiten_strain(strain_array, psd_in, sample_rate):
    """
    Whiten and bandpass a strain array using a pre-computed PSD.

    Replicates test_real_data.py lines 503-520:
    FFT -> divide by sqrt(PSD) -> bandpass 35-300 Hz.
    """
    ts = TimeSeries(strain_array.astype(np.float64), delta_t=1.0/sample_rate)
    psd_interp = interpolate(psd_in, 1.0 / ts.duration)
    freq_series = ts.to_frequencyseries()
    psd_interp.resize(len(freq_series))

    psd_arr = np.array(psd_interp)
    psd_arr[psd_arr <= 0] = 1e-40
    psd_safe = FrequencySeries(psd_arr, delta_f=psd_interp.delta_f)

    white = (freq_series / (psd_safe ** 0.5)).to_timeseries()
    white = highpass_fir(white, BANDPASS_LOW, 8)
    white = lowpass_fir(white, BANDPASS_HIGH, 8)
    return np.array(white)


def generate_and_whiten_simulated(
    detector_name, mass1, mass2, spin1z, spin2z,
    coa_phase_val, inc_val, distance, ra, dec, polarization,
    coalescence_time, real_strain_length, merger_idx, psd, sample_rate
):
    """
    Generate simulated waveform, project to detector, embed at merger
    location in zero-padded array, and whiten with same PSD as real data.

    Replicates test_real_data.py lines 531-555, parameterized by detector.
    """
    detector = Detector(detector_name)
    hp_t, hc_t = get_td_waveform(
        approximant=APPROXIMANT,
        mass1=mass1, mass2=mass2,
        spin1z=spin1z, spin2z=spin2z,
        inclination=inc_val, coa_phase=coa_phase_val,
        distance=distance, delta_t=DELTA_T, f_lower=F_LOWER
    )
    fp_t, fc_t = detector.antenna_pattern(ra, dec, polarization, coalescence_time)
    strain_t = fp_t * hp_t + fc_t * hc_t
    arr_t = np.array(strain_t)

    padded = np.zeros(real_strain_length)
    start_idx = merger_idx - len(arr_t)
    if start_idx >= 0:
        padded[start_idx:merger_idx] = arr_t
    else:
        padded[:merger_idx] = arr_t[-merger_idx:]

    return whiten_strain(padded, psd, sample_rate)


def optimize_phase_inclination(
    real_whitened, time_relative,
    detector_name, mass1, mass2, spin1z, spin2z,
    distance, ra, dec, polarization, coalescence_time,
    real_strain_length, merger_idx, psd, sample_rate,
    catalog_inclination=None
):
    """
    Grid search over inclination and coalescence phase for best
    Pearson correlation with real data.

    Replicates test_real_data.py lines 557-583.
    """
    inc_values = (
        [catalog_inclination] if catalog_inclination
        else INC_VALUES_DEFAULT
    )
    phase_values = np.linspace(0, 2*np.pi, N_PHASE_STEPS, endpoint=False)

    best_corr = -1
    best_phase = 0
    best_inc = inc_values[0]
    test_mask = (
        (time_relative > CORRELATION_WINDOW[0]) &
        (time_relative < CORRELATION_WINDOW[1])
    )

    n_total = len(inc_values) * len(phase_values)
    print(f"    Grid search: {len(inc_values)} inclinations x {len(phase_values)} phases = {n_total} trials...")

    for inc_test in inc_values:
        for phase_test in phase_values:
            try:
                sim_test = generate_and_whiten_simulated(
                    detector_name, mass1, mass2, spin1z, spin2z,
                    phase_test, inc_test, distance, ra, dec, polarization,
                    coalescence_time, real_strain_length, merger_idx,
                    psd, sample_rate
                )
                corr_val = np.abs(np.corrcoef(
                    real_whitened[test_mask], sim_test[test_mask]
                )[0, 1])
                if corr_val > best_corr:
                    best_corr = corr_val
                    best_phase = phase_test
                    best_inc = inc_test
            except Exception:
                continue

    return {
        'best_phase': best_phase,
        'best_inclination': best_inc,
        'best_correlation': best_corr
    }


def align_and_scale(real_whitened, sim_whitened, time_relative, sample_rate):
    """
    Align simulated waveform to real via cross-correlation,
    detect phase flip, and compute amplitude scaling.

    Replicates test_real_data.py lines 604-648.
    """
    align_mask = (
        (time_relative > ALIGN_WINDOW[0]) &
        (time_relative < ALIGN_WINDOW[1])
    )
    real_for_align = real_whitened[align_mask]
    sim_for_align = sim_whitened[align_mask]

    corr = correlate(real_for_align, sim_for_align, mode='full')
    max_pos = np.max(corr)
    max_neg = np.max(-corr)

    if max_pos >= max_neg:
        lag_idx = np.argmax(corr) - len(sim_for_align) + 1
        phase_flip = 1.0
    else:
        lag_idx = np.argmax(-corr) - len(sim_for_align) + 1
        phase_flip = -1.0

    sim_aligned = np.roll(sim_whitened * phase_flip, lag_idx)
    if lag_idx > 0:
        sim_aligned[:lag_idx] = 0
    elif lag_idx < 0:
        sim_aligned[lag_idx:] = 0

    # Amplitude scaling
    zoom_mask = (
        (time_relative > ZOOM_WIDE[0]) &
        (time_relative < ZOOM_WIDE[1])
    )
    real_zoom = real_whitened[zoom_mask]
    sim_zoom = sim_aligned[zoom_mask]
    scale = np.std(real_zoom) / (np.std(sim_zoom) + 1e-10)
    sim_scaled = sim_aligned * scale

    correlation = np.corrcoef(real_zoom, sim_zoom)[0, 1]

    return {
        'sim_aligned': sim_aligned,
        'sim_scaled': sim_scaled,
        'time_shift_samples': int(lag_idx),
        'time_shift_ms': lag_idx / sample_rate * 1000,
        'phase_flip': phase_flip,
        'scale_factor': scale,
        'correlation': correlation
    }


# ============================================================
# Per-Event Orchestration
# ============================================================

def process_single_event(event_name):
    """
    Process a single GW event: load real data, whiten, generate simulated
    waveform, optimize, align, and compute statistics for both H1 and L1.
    """
    print(f"\n  Loading {event_name} real data ({DATA_DURATION}s)...")
    real_data = load_real_event(event_name, detectors=DETECTORS, duration=DATA_DURATION)

    sample_rate = real_data['sample_rate']
    coalescence_time = real_data['coalescence_time']

    # Get event parameters for simulation
    params = get_event_parameters(event_name)
    mass1 = params.get('mass1') or params.get('mass1_source', 35.0)
    mass2 = params.get('mass2') or params.get('mass2_source', 30.0)
    spin1z = params.get('spin1z', 0.0) or 0.0
    spin2z = params.get('spin2z', 0.0) or 0.0
    distance = params.get('luminosity_distance', 410.0)
    inclination = params.get('inclination', 0.0) or 0.0
    ra = params.get('ra', 0.0) or 0.0
    dec = params.get('dec', 0.0) or 0.0
    polarization = params.get('polarization', 0.0) or 0.0

    print(f"  Parameters: M1={mass1:.1f}, M2={mass2:.1f}, D={distance:.0f} Mpc")
    print(f"  Spins: s1z={spin1z:.2f}, s2z={spin2z:.2f}")

    detector_results = {}
    shared_opt = None  # Grid search result shared across detectors

    for det in DETECTORS:
        if det not in real_data['strains']:
            print(f"  WARNING: {det} data not available for {event_name}, skipping")
            continue

        print(f"\n  Processing {det}...")

        # Extract raw strain and compute merger index
        real_strain = np.array(real_data['strains'][det])
        det_start_time = float(real_data['strains'][det].start_time)
        merger_idx = int((coalescence_time - det_start_time) * sample_rate)

        # Estimate PSD from full 32s strain (per-detector)
        print(f"    Estimating {det} PSD...")
        ts_real = TimeSeries(real_strain.astype(np.float64), delta_t=DELTA_T)
        psd = welch(ts_real, seg_len=4096, seg_stride=2048)

        # Whiten real data
        print(f"    Whitening {det} real data...")
        real_whitened = whiten_strain(real_strain, psd, sample_rate)

        # Time axis relative to merger
        time_full = np.arange(len(real_whitened)) / sample_rate
        time_relative = time_full - time_full[merger_idx]

        # Grid search: run on H1 only, share result for L1
        if shared_opt is None:
            print(f"    Optimizing phase/inclination on {det}...")
            shared_opt = optimize_phase_inclination(
                real_whitened, time_relative,
                det, mass1, mass2, spin1z, spin2z,
                distance, ra, dec, polarization, coalescence_time,
                len(real_strain), merger_idx, psd, sample_rate,
                catalog_inclination=inclination
            )
            print(f"    Best: phase={shared_opt['best_phase']:.3f} rad, "
                  f"inc={shared_opt['best_inclination']:.3f} rad, "
                  f"corr={shared_opt['best_correlation']:.3f}")

        # Generate final simulated waveform with optimized parameters
        print(f"    Generating {det} simulated waveform...")
        sim_whitened = generate_and_whiten_simulated(
            det, mass1, mass2, spin1z, spin2z,
            shared_opt['best_phase'], shared_opt['best_inclination'],
            distance, ra, dec, polarization,
            coalescence_time, len(real_strain), merger_idx,
            psd, sample_rate
        )

        # Check amplitude ratio and adjust distance if needed
        test_mask = (time_relative > CORRELATION_WINDOW[0]) & (time_relative < CORRELATION_WINDOW[1])
        amp_ratio = np.std(real_whitened[test_mask]) / (np.std(sim_whitened[test_mask]) + 1e-10)
        effective_distance = distance

        if amp_ratio > 1.5:
            effective_distance = distance / amp_ratio
            print(f"    Amplitude ratio {amp_ratio:.2f}x, trying effective distance {effective_distance:.0f} Mpc")
            sim_whitened = generate_and_whiten_simulated(
                det, mass1, mass2, spin1z, spin2z,
                shared_opt['best_phase'], shared_opt['best_inclination'],
                effective_distance, ra, dec, polarization,
                coalescence_time, len(real_strain), merger_idx,
                psd, sample_rate
            )

        # Align and scale
        print(f"    Aligning {det}...")
        alignment = align_and_scale(real_whitened, sim_whitened, time_relative, sample_rate)

        print(f"    {det} result: corr={alignment['correlation']:.4f}, "
              f"shift={alignment['time_shift_ms']:.2f}ms, "
              f"flip={alignment['phase_flip']:.0f}")

        detector_results[det] = {
            'real_whitened': real_whitened,
            'sim_scaled': alignment['sim_scaled'],
            'time_relative': time_relative,
            'merger_idx': merger_idx,
            'alignment': alignment,
            'optimization': shared_opt,
            'psd': psd,
            'effective_distance': effective_distance,
        }

    # H1 vs L1 comparison
    h1_l1_delay_theoretical = None
    h1_l1_delay_empirical = None
    h1_l1_correlation = None
    h1_l1_phase_flip = None

    if 'H1' in detector_results and 'L1' in detector_results:
        print(f"\n  Computing H1 vs L1 comparison...")

        # Theoretical delay
        try:
            h1_det = Detector('H1')
            l1_det = Detector('L1')
            if ra is not None and dec is not None:
                h1_l1_delay_theoretical = l1_det.time_delay_from_detector(
                    h1_det, ra, dec, coalescence_time
                )
                print(f"    Theoretical H1-L1 delay: {h1_l1_delay_theoretical*1000:.2f} ms")
        except Exception as e:
            print(f"    Could not compute theoretical delay: {e}")

        # Empirical delay via cross-correlation of whitened real signals
        h1_w = detector_results['H1']['real_whitened']
        l1_w = detector_results['L1']['real_whitened']
        h1_tr = detector_results['H1']['time_relative']
        l1_tr = detector_results['L1']['time_relative']

        # Use chirp region for cross-correlation
        h1_mask = (h1_tr > ALIGN_WINDOW[0]) & (h1_tr < ALIGN_WINDOW[1])
        l1_mask = (l1_tr > ALIGN_WINDOW[0]) & (l1_tr < ALIGN_WINDOW[1])

        h1_seg = h1_w[h1_mask]
        l1_seg = l1_w[l1_mask]

        # Match lengths
        min_len = min(len(h1_seg), len(l1_seg))
        h1_seg = h1_seg[:min_len]
        l1_seg = l1_seg[:min_len]

        cc = correlate(h1_seg, l1_seg, mode='full')
        max_pos = np.max(cc)
        max_neg = np.max(-cc)

        if max_pos >= max_neg:
            cc_lag = np.argmax(cc) - len(l1_seg) + 1
            h1_l1_phase_flip = 1.0
        else:
            cc_lag = np.argmax(-cc) - len(l1_seg) + 1
            h1_l1_phase_flip = -1.0

        h1_l1_delay_empirical = cc_lag / sample_rate
        print(f"    Empirical H1-L1 delay: {h1_l1_delay_empirical*1000:.2f} ms")
        if h1_l1_phase_flip < 0:
            print(f"    Phase inversion detected between H1 and L1")

        # Correlation after alignment
        l1_shifted = np.roll(l1_seg * h1_l1_phase_flip, cc_lag)
        if cc_lag > 0:
            l1_shifted[:cc_lag] = 0
        elif cc_lag < 0:
            l1_shifted[cc_lag:] = 0
        h1_l1_correlation = np.corrcoef(h1_seg, l1_shifted)[0, 1]
        print(f"    H1-L1 correlation (aligned): {h1_l1_correlation:.4f}")

    return {
        'event_name': event_name,
        'params': {
            'mass1': mass1, 'mass2': mass2,
            'spin1z': spin1z, 'spin2z': spin2z,
            'distance': distance,
            'inclination': inclination,
            'ra': ra, 'dec': dec,
            'polarization': polarization,
        },
        'detectors': detector_results,
        'optimization': shared_opt,
        'h1_l1_delay_theoretical': h1_l1_delay_theoretical,
        'h1_l1_delay_empirical': h1_l1_delay_empirical,
        'h1_l1_correlation': h1_l1_correlation,
        'h1_l1_phase_flip': h1_l1_phase_flip,
    }


# ============================================================
# Plotting Functions
# ============================================================

def plot_event_detector_comparison(event_result, output_path):
    """
    3x2 figure per event: H1 (left) vs L1 (right).
    Row 1: Real vs Simulated overlay (-0.5s to +0.1s)
    Row 2: Merger detail (-0.15s to +0.05s)
    Row 3: Residual
    """
    event_name = event_result['event_name']
    det_results = event_result['detectors']
    available_dets = [d for d in DETECTORS if d in det_results]

    if not available_dets:
        print(f"  No detector data to plot for {event_name}")
        return

    n_cols = len(available_dets)
    fig, axes = plt.subplots(3, n_cols, figsize=(7 * n_cols, 12))
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    p = event_result['params']
    fig.suptitle(
        f"{event_name}: Real vs Simulated (both whitened, {BANDPASS_LOW:.0f}-{BANDPASS_HIGH:.0f} Hz)\n"
        f"M1={p['mass1']:.1f}, M2={p['mass2']:.1f} Msun, D={p['distance']:.0f} Mpc",
        fontsize=13, y=0.98
    )

    for col, det in enumerate(available_dets):
        dr = det_results[det]
        tr = dr['time_relative']
        real_w = dr['real_whitened']
        sim_s = dr['sim_scaled']
        alignment = dr['alignment']
        residual = real_w - sim_s

        # Row 1: Wide zoom overlay
        mask1 = (tr > ZOOM_WIDE[0]) & (tr < ZOOM_WIDE[1])
        ax = axes[0, col]
        ax.plot(tr[mask1], real_w[mask1], 'b-', linewidth=0.8, alpha=0.8, label='Real')
        ax.plot(tr[mask1], sim_s[mask1], 'r--', linewidth=0.8, alpha=0.8, label='Simulated')
        ax.axvline(x=0, color='k', linestyle=':', alpha=0.5)
        ax.set_title(f"{det}  (corr={alignment['correlation']:.3f}, "
                     f"shift={alignment['time_shift_ms']:.1f}ms)")
        ax.set_ylabel('Whitened Strain')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

        # Row 2: Merger detail
        mask2 = (tr > ZOOM_MERGER[0]) & (tr < ZOOM_MERGER[1])
        ax = axes[1, col]
        ax.plot(tr[mask2], real_w[mask2], 'b-', linewidth=1.0, alpha=0.9, label='Real')
        ax.plot(tr[mask2], sim_s[mask2], 'r--', linewidth=1.0, alpha=0.9, label='Simulated')
        ax.axvline(x=0, color='k', linestyle=':', alpha=0.5)
        ax.set_title(f'{det} Merger Detail')
        ax.set_ylabel('Whitened Strain')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

        # Row 3: Residual
        ax = axes[2, col]
        ax.plot(tr[mask1], residual[mask1], 'g-', linewidth=0.6, alpha=0.8)
        ax.axvline(x=0, color='k', linestyle=':', alpha=0.5)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.set_title(f'{det} Residual (Real - Simulated)')
        ax.set_xlabel('Time relative to merger (s)')
        ax.set_ylabel('Residual')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(output_path)}")


def plot_h1_vs_l1(event_result, output_path):
    """
    2-panel figure: H1 & L1 whitened real signals overlaid (with time shift),
    and cross-correlation function.
    """
    event_name = event_result['event_name']
    det_results = event_result['detectors']

    if 'H1' not in det_results or 'L1' not in det_results:
        print(f"  Cannot plot H1 vs L1 for {event_name}: missing detector data")
        return

    h1_dr = det_results['H1']
    l1_dr = det_results['L1']

    h1_w = h1_dr['real_whitened']
    l1_w = l1_dr['real_whitened']
    h1_tr = h1_dr['time_relative']
    l1_tr = l1_dr['time_relative']

    # Cross-correlation in chirp region
    h1_mask = (h1_tr > ALIGN_WINDOW[0]) & (h1_tr < ALIGN_WINDOW[1])
    l1_mask = (l1_tr > ALIGN_WINDOW[0]) & (l1_tr < ALIGN_WINDOW[1])

    h1_seg = h1_w[h1_mask]
    l1_seg = l1_w[l1_mask]
    min_len = min(len(h1_seg), len(l1_seg))
    h1_seg = h1_seg[:min_len]
    l1_seg = l1_seg[:min_len]

    cc = correlate(h1_seg, l1_seg, mode='full')
    cc_lags = np.arange(len(cc)) - (len(l1_seg) - 1)
    cc_time = cc_lags / SAMPLE_RATE * 1000  # milliseconds

    # Find best alignment
    max_pos = np.max(cc)
    max_neg = np.max(-cc)
    if max_pos >= max_neg:
        best_lag = np.argmax(cc) - len(l1_seg) + 1
        flip = 1.0
    else:
        best_lag = np.argmax(-cc) - len(l1_seg) + 1
        flip = -1.0

    delay_ms = best_lag / SAMPLE_RATE * 1000

    # Shift L1 to align with H1
    l1_shifted = np.roll(l1_seg * flip, best_lag)
    if best_lag > 0:
        l1_shifted[:best_lag] = 0
    elif best_lag < 0:
        l1_shifted[best_lag:] = 0

    t_seg = h1_tr[h1_mask][:min_len]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    theo_delay = event_result.get('h1_l1_delay_theoretical')
    flip_label = " (inverted)" if flip < 0 else ""

    # Panel 1: Overlaid signals
    ax = axes[0]
    ax.plot(t_seg, h1_seg, 'b-', linewidth=0.8, alpha=0.8, label='H1')
    ax.plot(t_seg, l1_shifted, 'g-', linewidth=0.8, alpha=0.8,
            label=f'L1 (shifted {delay_ms:.1f}ms{flip_label})')
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Merger')
    ax.set_title(f'{event_name}: H1 vs L1 Whitened Real Signals')
    ax.set_xlabel('Time relative to merger (s)')
    ax.set_ylabel('Whitened Strain')

    info_text = f"Empirical delay: {delay_ms:.2f} ms"
    if theo_delay is not None:
        info_text += f"\nTheoretical delay: {theo_delay*1000:.2f} ms"
    corr_val = event_result.get('h1_l1_correlation')
    if corr_val is not None:
        info_text += f"\nCorrelation: {corr_val:.3f}"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Cross-correlation function
    ax = axes[1]
    # Limit to +/-50ms window
    cc_mask = (cc_time > -50) & (cc_time < 50)
    ax.plot(cc_time[cc_mask], cc[cc_mask], 'k-', linewidth=0.8)
    ax.axvline(x=delay_ms, color='r', linestyle='--', alpha=0.7,
               label=f'Peak: {delay_ms:.2f} ms')
    if theo_delay is not None:
        ax.axvline(x=theo_delay*1000, color='orange', linestyle=':', alpha=0.7,
                   label=f'Theoretical: {theo_delay*1000:.2f} ms')
    ax.set_title(f'{event_name}: H1-L1 Cross-Correlation')
    ax.set_xlabel('Lag (ms)')
    ax.set_ylabel('Cross-Correlation')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(output_path)}")


def plot_summary_bar_chart(all_results, output_path):
    """
    Summary figure: grouped bar chart of correlations across all events.
    Top: H1/L1 real-vs-sim correlation per event.
    Bottom: H1-L1 inter-detector correlation per event.
    """
    event_names = []
    h1_corrs = []
    l1_corrs = []
    h1l1_corrs = []

    for r in all_results:
        event_names.append(r['event_name'])
        dets = r['detectors']
        h1_corrs.append(dets['H1']['alignment']['correlation'] if 'H1' in dets else 0)
        l1_corrs.append(dets['L1']['alignment']['correlation'] if 'L1' in dets else 0)
        h1l1_corrs.append(r.get('h1_l1_correlation') or 0)

    x = np.arange(len(event_names))
    width = 0.35

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Top: Real vs Simulated correlation
    ax = axes[0]
    bars1 = ax.bar(x - width/2, h1_corrs, width, label='H1', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, l1_corrs, width, label='L1', color='darkorange', alpha=0.8)

    # Add value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                f'{h:.2f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                f'{h:.2f}', ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Pearson Correlation')
    ax.set_title('Real vs Simulated Waveform Correlation (per detector)')
    ax.set_xticks(x)
    ax.set_xticklabels(event_names, fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.15)

    # Bottom: H1-L1 correlation
    ax = axes[1]
    colors = ['mediumseagreen' if c > 0.5 else 'salmon' for c in h1l1_corrs]
    bars = ax.bar(x, h1l1_corrs, width*1.5, color=colors, alpha=0.8)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                f'{h:.2f}', ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Pearson Correlation')
    ax.set_title('H1 vs L1 Inter-Detector Correlation (after alignment)')
    ax.set_xticks(x)
    ax.set_xticklabels(event_names, fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(output_path)}")


# ============================================================
# Results Summary
# ============================================================

def print_results_table(all_results):
    """Print formatted table of all event results."""
    print("\n" + "=" * 110)
    print("MULTI-EVENT COMPARISON RESULTS")
    print("=" * 110)

    header = (
        f"{'Event':<12} {'M1':>6} {'M2':>6} {'Dist':>6} "
        f"{'H1 Corr':>8} {'L1 Corr':>8} "
        f"{'H1 Shift':>9} {'L1 Shift':>9} "
        f"{'H1L1 Delay':>11} {'H1L1 Corr':>10}"
    )
    print(header)
    print("-" * 110)

    for r in all_results:
        p = r['params']
        dets = r['detectors']

        h1_corr = dets['H1']['alignment']['correlation'] if 'H1' in dets else float('nan')
        l1_corr = dets['L1']['alignment']['correlation'] if 'L1' in dets else float('nan')
        h1_shift = dets['H1']['alignment']['time_shift_ms'] if 'H1' in dets else float('nan')
        l1_shift = dets['L1']['alignment']['time_shift_ms'] if 'L1' in dets else float('nan')

        delay_emp = r.get('h1_l1_delay_empirical')
        delay_str = f"{delay_emp*1000:.2f}ms" if delay_emp is not None else "N/A"

        h1l1_corr = r.get('h1_l1_correlation')
        h1l1_str = f"{h1l1_corr:.4f}" if h1l1_corr is not None else "N/A"

        print(
            f"{r['event_name']:<12} "
            f"{p['mass1']:>6.1f} {p['mass2']:>6.1f} {p['distance']:>6.0f} "
            f"{h1_corr:>8.4f} {l1_corr:>8.4f} "
            f"{h1_shift:>8.2f}ms {l1_shift:>8.2f}ms "
            f"{delay_str:>11} {h1l1_str:>10}"
        )

    print("=" * 110)

    # Interpretation guide
    print("\nInterpretation:")
    print("  Corr > 0.8: Excellent match between real and simulated")
    print("  Corr 0.5-0.8: Good match (differences from parameter uncertainties + noise)")
    print("  H1L1 Delay: Should be 0-10 ms (light travel time between sites)")
    print("  H1L1 Corr > 0.8: Strong agreement between detectors")


def save_results_json(all_results, output_path):
    """Save numerical results to JSON for later analysis."""
    serializable = []
    for r in all_results:
        entry = {
            'event_name': r['event_name'],
            'params': r['params'],
            'detectors': {},
        }
        for det, dr in r['detectors'].items():
            entry['detectors'][det] = {
                'correlation': float(dr['alignment']['correlation']),
                'time_shift_ms': float(dr['alignment']['time_shift_ms']),
                'phase_flip': float(dr['alignment']['phase_flip']),
                'scale_factor': float(dr['alignment']['scale_factor']),
                'effective_distance': float(dr['effective_distance']),
            }

        if r.get('optimization'):
            entry['optimization'] = {
                'best_phase': float(r['optimization']['best_phase']),
                'best_inclination': float(r['optimization']['best_inclination']),
                'best_correlation': float(r['optimization']['best_correlation']),
            }

        entry['h1_l1_delay_theoretical'] = (
            float(r['h1_l1_delay_theoretical'])
            if r.get('h1_l1_delay_theoretical') is not None else None
        )
        entry['h1_l1_delay_empirical'] = (
            float(r['h1_l1_delay_empirical'])
            if r.get('h1_l1_delay_empirical') is not None else None
        )
        entry['h1_l1_correlation'] = (
            float(r['h1_l1_correlation'])
            if r.get('h1_l1_correlation') is not None else None
        )
        entry['h1_l1_phase_flip'] = (
            float(r['h1_l1_phase_flip'])
            if r.get('h1_l1_phase_flip') is not None else None
        )

        serializable.append(entry)

    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"  Saved: {os.path.basename(output_path)}")


# ============================================================
# Main Entry Point
# ============================================================

def run_multi_event_comparison():
    """Main: process all events and generate all outputs."""
    print("=" * 70)
    print("Multi-Event GW Comparison: Real vs Simulated, H1 vs L1")
    print("=" * 70)
    print(f"Events: {EVENTS}")
    print(f"Detectors: {DETECTORS}")
    print(f"Processing: whiten (PSD, {BANDPASS_LOW}-{BANDPASS_HIGH} Hz), "
          f"IMRPhenomXP simulation, grid search optimization")

    all_results = []

    for i, event_name in enumerate(EVENTS):
        print(f"\n{'='*70}")
        print(f"EVENT {i+1}/{len(EVENTS)}: {event_name}")
        print(f"{'='*70}")
        print(get_event_summary(event_name))

        t0 = time_mod.time()
        try:
            result = process_single_event(event_name)
            all_results.append(result)

            # Per-event plots
            plot_event_detector_comparison(
                result,
                os.path.join(OUTPUT_DIR, f'{event_name}_dual_detector_comparison.png')
            )
            plot_h1_vs_l1(
                result,
                os.path.join(OUTPUT_DIR, f'{event_name}_H1_vs_L1.png')
            )

            elapsed = time_mod.time() - t0
            print(f"\n  Completed {event_name} in {elapsed:.1f}s")

        except Exception as e:
            print(f"\n  FAILED to process {event_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_results:
        print("\nNo events processed successfully!")
        return False

    # Summary outputs
    print_results_table(all_results)

    plot_summary_bar_chart(
        all_results,
        os.path.join(OUTPUT_DIR, 'multi_event_comparison_summary.png')
    )

    save_results_json(
        all_results,
        os.path.join(OUTPUT_DIR, 'multi_event_comparison_results.json')
    )

    print(f"\nProcessed {len(all_results)}/{len(EVENTS)} events successfully")
    print(f"Output directory: {OUTPUT_DIR}")
    return True


if __name__ == "__main__":
    success = run_multi_event_comparison()
    sys.exit(0 if success else 1)
