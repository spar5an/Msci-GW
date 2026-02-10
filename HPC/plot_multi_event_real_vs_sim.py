"""
Consolidated multi-event plot: real GW data vs simulated waveforms.

Produces a single figure with one row per event, two columns (H1, L1),
showing whitened real signals overlaid with IMRPhenomXP simulations.

Uses the exact processing pipeline from Real Data/test_real_data.py.

Usage:
    python plot_multi_event_real_vs_sim.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Real Data'))

from pycbc.waveform import get_td_waveform
from pycbc.types import TimeSeries, FrequencySeries
from pycbc.detector import Detector
from pycbc.psd import welch, interpolate
from pycbc.filter import highpass_fir, lowpass_fir
from scipy.signal import correlate

from catalog_utils import get_event_parameters
from real_data_loader import load_real_event


# ============================================================
# Configuration
# ============================================================

EVENTS = [
    'GW150914',
    'GW151226',
    'GW170104',
    'GW170608',
    'GW170814',
]

DETECTORS = ['H1', 'L1']
SAMPLE_RATE = 4096
DELTA_T = 1.0 / SAMPLE_RATE
BANDPASS_LOW = 35.0
BANDPASS_HIGH = 300.0
F_LOWER = 20.0
APPROXIMANT = 'IMRPhenomXP'

INC_VALUES = [np.pi/8, np.pi/6, np.pi/4, np.pi/3]
N_PHASE_STEPS = 16

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# Processing (exact pipeline from test_real_data.py lines 503-648)
# ============================================================

def whiten_strain(strain_array, psd_in, sample_rate):
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


def generate_and_whiten_sim(det_name, mass1, mass2, spin1z, spin2z,
                            phase, inc, distance, ra, dec, pol,
                            t_coal, strain_len, merger_idx, psd):
    detector = Detector(det_name)
    hp, hc = get_td_waveform(
        approximant=APPROXIMANT, mass1=mass1, mass2=mass2,
        spin1z=spin1z, spin2z=spin2z, inclination=inc,
        coa_phase=phase, distance=distance,
        delta_t=DELTA_T, f_lower=F_LOWER
    )
    fp, fc = detector.antenna_pattern(ra, dec, pol, t_coal)
    strain = fp * hp + fc * hc
    arr = np.array(strain)

    padded = np.zeros(strain_len)
    si = merger_idx - len(arr)
    if si >= 0:
        padded[si:merger_idx] = arr
    else:
        padded[:merger_idx] = arr[-merger_idx:]
    return whiten_strain(padded, psd, SAMPLE_RATE)


def process_event_detector(event_name, det_name, real_data, params, shared_opt=None):
    """Process one detector for one event. Returns dict with all results."""
    coal_time = real_data['coalescence_time']
    real_strain = np.array(real_data['strains'][det_name])
    det_start = float(real_data['strains'][det_name].start_time)
    merger_idx = int((coal_time - det_start) * SAMPLE_RATE)

    mass1 = params.get('mass1') or params.get('mass1_source', 35.0)
    mass2 = params.get('mass2') or params.get('mass2_source', 30.0)
    spin1z = params.get('spin1z', 0.0) or 0.0
    spin2z = params.get('spin2z', 0.0) or 0.0
    distance = params.get('luminosity_distance', 410.0)
    inc = params.get('inclination', 0.0) or 0.0
    ra = params.get('ra', 0.0) or 0.0
    dec = params.get('dec', 0.0) or 0.0
    pol = params.get('polarization', 0.0) or 0.0

    # PSD estimation
    ts_real = TimeSeries(real_strain.astype(np.float64), delta_t=DELTA_T)
    psd = welch(ts_real, seg_len=4096, seg_stride=2048)

    # Whiten real data
    real_w = whiten_strain(real_strain, psd, SAMPLE_RATE)
    time_full = np.arange(len(real_w)) / SAMPLE_RATE
    time_rel = time_full - time_full[merger_idx]

    # Grid search (only if no shared result yet)
    if shared_opt is None:
        inc_vals = [inc] if inc else INC_VALUES
        phase_vals = np.linspace(0, 2*np.pi, N_PHASE_STEPS, endpoint=False)
        test_mask = (time_rel > -0.2) & (time_rel < 0.05)
        best_corr, best_phase, best_inc = -1, 0, inc_vals[0]

        for iv in inc_vals:
            for pv in phase_vals:
                try:
                    st = generate_and_whiten_sim(
                        det_name, mass1, mass2, spin1z, spin2z,
                        pv, iv, distance, ra, dec, pol,
                        coal_time, len(real_strain), merger_idx, psd
                    )
                    cv = np.abs(np.corrcoef(real_w[test_mask], st[test_mask])[0, 1])
                    if cv > best_corr:
                        best_corr, best_phase, best_inc = cv, pv, iv
                except Exception:
                    continue

        shared_opt = {'phase': best_phase, 'inc': best_inc, 'corr': best_corr}

    # Final simulation with optimized params
    sim_w = generate_and_whiten_sim(
        det_name, mass1, mass2, spin1z, spin2z,
        shared_opt['phase'], shared_opt['inc'], distance, ra, dec, pol,
        coal_time, len(real_strain), merger_idx, psd
    )

    # Distance adjustment if amplitude is way off
    test_mask = (time_rel > -0.2) & (time_rel < 0.05)
    amp_ratio = np.std(real_w[test_mask]) / (np.std(sim_w[test_mask]) + 1e-10)
    eff_dist = distance
    if amp_ratio > 1.5:
        eff_dist = distance / amp_ratio
        sim_w = generate_and_whiten_sim(
            det_name, mass1, mass2, spin1z, spin2z,
            shared_opt['phase'], shared_opt['inc'], eff_dist, ra, dec, pol,
            coal_time, len(real_strain), merger_idx, psd
        )

    # Cross-correlation alignment
    align_mask = (time_rel > -0.5) & (time_rel < 0.1)
    r_seg = real_w[align_mask]
    s_seg = sim_w[align_mask]
    cc = correlate(r_seg, s_seg, mode='full')
    if np.max(cc) >= np.max(-cc):
        lag = np.argmax(cc) - len(s_seg) + 1
        flip = 1.0
    else:
        lag = np.argmax(-cc) - len(s_seg) + 1
        flip = -1.0

    sim_aligned = np.roll(sim_w * flip, lag)
    if lag > 0:
        sim_aligned[:lag] = 0
    elif lag < 0:
        sim_aligned[lag:] = 0

    # Amplitude scaling
    zoom_mask = (time_rel > -0.5) & (time_rel < 0.1)
    scale = np.std(real_w[zoom_mask]) / (np.std(sim_aligned[zoom_mask]) + 1e-10)
    sim_scaled = sim_aligned * scale

    corr = np.corrcoef(real_w[zoom_mask], sim_aligned[zoom_mask])[0, 1]

    return {
        'real_w': real_w,
        'sim_scaled': sim_scaled,
        'time_rel': time_rel,
        'corr': corr,
        'shift_ms': lag / SAMPLE_RATE * 1000,
        'flip': flip,
        'mass1': mass1, 'mass2': mass2, 'distance': distance,
        'shared_opt': shared_opt,
    }


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 65)
    print("Multi-Event Real vs Simulated Comparison Plot")
    print("=" * 65)

    n_events = len(EVENTS)
    n_dets = len(DETECTORS)

    # Collect all results first
    all_results = {}  # event_name -> {det_name -> result}

    for event_name in EVENTS:
        print(f"\nProcessing {event_name}...")
        try:
            real_data = load_real_event(event_name, detectors=DETECTORS, duration=32.0)
            params = get_event_parameters(event_name)
        except Exception as e:
            print(f"  FAILED to load {event_name}: {e}")
            continue

        event_results = {}
        shared_opt = None

        for det in DETECTORS:
            if det not in real_data['strains']:
                print(f"  {det} not available, skipping")
                continue
            print(f"  {det}...", end="", flush=True)
            res = process_event_detector(event_name, det, real_data, params, shared_opt)
            shared_opt = res['shared_opt']
            event_results[det] = res
            print(f" corr={res['corr']:.3f}")

        if event_results:
            all_results[event_name] = event_results

    if not all_results:
        print("No events processed!")
        return

    # ========================================================
    # Figure 1: Multi-event real vs simulated (merger zoom)
    # One row per event, one column per detector
    # ========================================================
    n_rows = len(all_results)
    fig, axes = plt.subplots(n_rows, n_dets, figsize=(14, 3.2 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    det_colors = {'H1': ('steelblue', 'Real H1'), 'L1': ('seagreen', 'Real L1')}

    for row, event_name in enumerate(all_results):
        er = all_results[event_name]
        for col, det in enumerate(DETECTORS):
            ax = axes[row, col]
            if det not in er:
                ax.text(0.5, 0.5, f'{det} not available',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_facecolor('#f8f8f8')
                continue

            r = er[det]
            tr = r['time_rel']
            mask = (tr > -0.2) & (tr < 0.05)

            real_color, real_label = det_colors[det]
            ax.plot(tr[mask], r['real_w'][mask], color=real_color,
                    linewidth=0.9, alpha=0.85, label='Real')
            ax.plot(tr[mask], r['sim_scaled'][mask], color='crimson',
                    linewidth=0.9, alpha=0.75, linestyle='--', label='Simulated')
            ax.axvline(x=0, color='k', linestyle=':', alpha=0.4, linewidth=0.7)

            # Annotations
            ax.text(0.97, 0.95,
                    f"r = {r['corr']:.3f}\n"
                    f"shift = {r['shift_ms']:.1f}ms"
                    + (f"\ninverted" if r['flip'] < 0 else ""),
                    transform=ax.transAxes, fontsize=7.5,
                    ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

            if col == 0:
                ax.set_ylabel(
                    f"{event_name}\n"
                    f"({r['mass1']:.0f}+{r['mass2']:.0f} M$_\\odot$, "
                    f"{r['distance']:.0f} Mpc)",
                    fontsize=8.5)
            if row == 0:
                ax.set_title(f'{det}', fontsize=11, fontweight='bold')
            if row == n_rows - 1:
                ax.set_xlabel('Time relative to merger (s)', fontsize=9)
            else:
                ax.set_xticklabels([])

            ax.legend(loc='upper left', fontsize=7, framealpha=0.7)
            ax.grid(True, alpha=0.2)
            ax.tick_params(labelsize=8)

    fig.suptitle(
        f'Real GW Events vs IMRPhenomXP Simulations '
        f'(whitened, {BANDPASS_LOW:.0f}-{BANDPASS_HIGH:.0f} Hz)',
        fontsize=13, fontweight='bold', y=1.0)
    plt.tight_layout()

    path1 = os.path.join(OUTPUT_DIR, 'multi_event_real_vs_simulated.png')
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {path1}")

    # ========================================================
    # Figure 2: Wider view with residuals
    # Two rows per event: top = overlay, bottom = residual
    # ========================================================
    fig2, axes2 = plt.subplots(n_rows * 2, n_dets, figsize=(14, 2.8 * n_rows * 2),
                               gridspec_kw={'height_ratios': [2, 1] * n_rows})

    for ev_idx, event_name in enumerate(all_results):
        er = all_results[event_name]
        row_sig = ev_idx * 2
        row_res = ev_idx * 2 + 1

        for col, det in enumerate(DETECTORS):
            ax_sig = axes2[row_sig, col]
            ax_res = axes2[row_res, col]

            if det not in er:
                for ax in [ax_sig, ax_res]:
                    ax.text(0.5, 0.5, f'{det} N/A',
                            ha='center', va='center', transform=ax.transAxes)
                    ax.set_facecolor('#f8f8f8')
                continue

            r = er[det]
            tr = r['time_rel']
            mask = (tr > -0.5) & (tr < 0.1)

            real_color = det_colors[det][0]

            # Signal panel
            ax_sig.plot(tr[mask], r['real_w'][mask], color=real_color,
                        linewidth=0.7, alpha=0.8, label='Real')
            ax_sig.plot(tr[mask], r['sim_scaled'][mask], color='crimson',
                        linewidth=0.7, alpha=0.7, linestyle='--', label='Simulated')
            ax_sig.axvline(x=0, color='k', linestyle=':', alpha=0.4, linewidth=0.7)
            ax_sig.legend(loc='upper left', fontsize=7)
            ax_sig.grid(True, alpha=0.2)
            ax_sig.tick_params(labelsize=7)
            ax_sig.set_xticklabels([])

            if col == 0:
                ax_sig.set_ylabel(
                    f"{event_name}\n({r['mass1']:.0f}+{r['mass2']:.0f} M$_\\odot$)",
                    fontsize=8)
            if ev_idx == 0:
                ax_sig.set_title(f'{det}', fontsize=11, fontweight='bold')

            ax_sig.text(0.97, 0.95, f"r = {r['corr']:.3f}",
                        transform=ax_sig.transAxes, fontsize=7.5,
                        ha='right', va='top',
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

            # Residual panel
            residual = r['real_w'] - r['sim_scaled']
            ax_res.plot(tr[mask], residual[mask], color='gray',
                        linewidth=0.5, alpha=0.7)
            ax_res.axvline(x=0, color='k', linestyle=':', alpha=0.4, linewidth=0.7)
            ax_res.axhline(y=0, color='k', linestyle='-', alpha=0.2)
            ax_res.grid(True, alpha=0.2)
            ax_res.tick_params(labelsize=7)

            if col == 0:
                ax_res.set_ylabel('Residual', fontsize=7.5)
            if ev_idx == n_rows - 1:
                ax_res.set_xlabel('Time relative to merger (s)', fontsize=8)
            else:
                ax_res.set_xticklabels([])

    fig2.suptitle(
        f'Real vs Simulated with Residuals '
        f'(whitened, {BANDPASS_LOW:.0f}-{BANDPASS_HIGH:.0f} Hz)',
        fontsize=13, fontweight='bold', y=1.0)
    plt.tight_layout()

    path2 = os.path.join(OUTPUT_DIR, 'multi_event_real_vs_sim_with_residuals.png')
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path2}")

    # ========================================================
    # Figure 3: Summary statistics
    # ========================================================
    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 5))

    event_labels = list(all_results.keys())
    x = np.arange(len(event_labels))
    width = 0.3

    h1_corrs = [all_results[e].get('H1', {}).get('corr', 0) for e in event_labels]
    l1_corrs = [all_results[e].get('L1', {}).get('corr', 0) for e in event_labels]

    bars1 = ax3.bar(x - width/2, h1_corrs, width, label='H1',
                    color='steelblue', alpha=0.8, edgecolor='white')
    bars2 = ax3.bar(x + width/2, l1_corrs, width, label='L1',
                    color='darkorange', alpha=0.8, edgecolor='white')

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                     f'{h:.2f}', ha='center', va='bottom', fontsize=9)

    # Add mass annotations below
    for i, e in enumerate(event_labels):
        r = list(all_results[e].values())[0]
        ax3.text(i, -0.07,
                 f"{r['mass1']:.0f}+{r['mass2']:.0f} M$_\\odot$\n{r['distance']:.0f} Mpc",
                 ha='center', va='top', fontsize=7.5, color='gray',
                 transform=ax3.get_xaxis_transform())

    ax3.set_ylabel('Pearson Correlation', fontsize=11)
    ax3.set_title('Real vs Simulated Match Quality Across Events', fontsize=13)
    ax3.set_xticks(x)
    ax3.set_xticklabels(event_labels, fontsize=10)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.2, axis='y')
    ax3.set_ylim(0, max(max(h1_corrs), max(l1_corrs)) * 1.25)

    plt.tight_layout()
    path3 = os.path.join(OUTPUT_DIR, 'multi_event_correlation_summary.png')
    plt.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path3}")

    # Print summary table
    print(f"\n{'='*75}")
    print(f"{'Event':<12} {'Masses':>14} {'Dist':>7} {'H1 corr':>9} {'L1 corr':>9}")
    print(f"{'-'*75}")
    for e in event_labels:
        r0 = list(all_results[e].values())[0]
        h1c = all_results[e].get('H1', {}).get('corr', float('nan'))
        l1c = all_results[e].get('L1', {}).get('corr', float('nan'))
        print(f"{e:<12} {r0['mass1']:>6.1f}+{r0['mass2']:<6.1f} {r0['distance']:>6.0f}  "
              f"{h1c:>8.3f}  {l1c:>8.3f}")
    print(f"{'='*75}")
    print(f"\nDone! 3 plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
