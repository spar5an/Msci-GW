"""
Comparison utilities for real vs simulated GW data.

This module provides functions to compare real gravitational wave events
with simulated waveforms generated using the same physical parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from pycbc.waveform import get_td_waveform
from pycbc.detector import Detector
from pycbc.types import TimeSeries, FrequencySeries
from pycbc.filter import match, matched_filter, sigmasq
from pycbc.psd import welch, interpolate

try:
    from .catalog_utils import get_event_parameters
except ImportError:
    from catalog_utils import get_event_parameters


def generate_comparison_waveform(
    event_name: str,
    approximant: str = 'IMRPhenomXP',
    time_resolution: float = 1/4096,
    signal_length: float = 2.0,
    detectors: List[str] = None,
    add_noise: bool = False,
    f_lower: float = 20.0
) -> Dict:
    """
    Generate a simulated waveform using parameters from a real event.

    Extracts parameters from GWTC catalog and generates a waveform
    with the same physical parameters for direct comparison.

    Parameters
    ----------
    event_name : str
        Real event name (e.g., 'GW150914')
    approximant : str
        Waveform approximant. Default: 'IMRPhenomXP'
    time_resolution : float
        Time step. Default: 1/4096
    signal_length : float
        Signal duration (seconds). Default: 2.0
    detectors : List[str], optional
        Detector names. Default: ['H1', 'L1']
    add_noise : bool
        Add simulated noise. Default: False
    f_lower : float
        Lower frequency cutoff. Default: 20.0 Hz

    Returns
    -------
    Dict containing:
        - waveforms: Dict[str, np.ndarray] per detector
        - parameters: Dict of source parameters used
        - metadata: Dict with generation details
    """
    if detectors is None:
        detectors = ['H1', 'L1']

    # Get event parameters
    params = get_event_parameters(event_name)

    # Extract waveform parameters
    mass1 = params.get('mass1') or params.get('mass1_source', 35.0)
    mass2 = params.get('mass2') or params.get('mass2_source', 30.0)
    spin1z = params.get('spin1z', 0.0) or 0.0
    spin2z = params.get('spin2z', 0.0) or 0.0
    distance = params.get('luminosity_distance', 410.0)
    inclination = params.get('inclination', 0.0) or 0.0
    coa_phase = params.get('coa_phase', 0.0) or 0.0
    ra = params.get('ra', 0.0) or 0.0
    dec = params.get('dec', 0.0) or 0.0
    polarization = params.get('polarization', 0.0) or 0.0

    # Generate waveform
    hp, hc = get_td_waveform(
        approximant=approximant,
        mass1=mass1,
        mass2=mass2,
        spin1z=spin1z,
        spin2z=spin2z,
        inclination=inclination,
        coa_phase=coa_phase,
        distance=distance,
        delta_t=time_resolution,
        f_lower=f_lower
    )

    target_length = int(signal_length / time_resolution)

    # Project to detectors
    waveforms = {}
    for det_name in detectors:
        try:
            detector = Detector(det_name)

            # Project waveform to detector
            # Use a fixed GPS time for projection (doesn't affect comparison)
            gps_time = 1126259462.0  # GW150914 time
            fp, fc = detector.antenna_pattern(ra, dec, polarization, gps_time)
            strain = fp * hp + fc * hc

            # Crop/pad to target length
            strain_array = np.array(strain)

            if len(strain_array) > target_length:
                # Keep the end (merger portion)
                strain_array = strain_array[-target_length:]
            elif len(strain_array) < target_length:
                # Pad at the beginning
                pad_length = target_length - len(strain_array)
                strain_array = np.pad(strain_array, (pad_length, 0), mode='constant')

            waveforms[det_name] = strain_array

        except Exception as e:
            print(f"Warning: Could not project to {det_name}: {e}")
            waveforms[det_name] = np.zeros(target_length)

    return {
        'waveforms': waveforms,
        'parameters': {
            'mass1': mass1,
            'mass2': mass2,
            'spin1z': spin1z,
            'spin2z': spin2z,
            'distance': distance,
            'inclination': inclination,
            'coa_phase': coa_phase,
            'ra': ra,
            'dec': dec,
            'polarization': polarization
        },
        'metadata': {
            'event_name': event_name,
            'approximant': approximant,
            'time_resolution': time_resolution,
            'signal_length': signal_length,
            'f_lower': f_lower,
            'detectors': detectors
        }
    }


def compute_match(
    real_strain: np.ndarray,
    simulated_strain: np.ndarray,
    delta_t: float,
    f_lower: float = 20.0,
    psd: np.ndarray = None
) -> Tuple[float, float, float]:
    """
    Compute match (overlap) between real and simulated strains.

    The match is the inner product maximized over time and phase shifts,
    normalized to give a value between 0 and 1.

    Parameters
    ----------
    real_strain : np.ndarray
        Real detector strain
    simulated_strain : np.ndarray
        Simulated waveform
    delta_t : float
        Time resolution
    f_lower : float
        Lower frequency cutoff. Default: 20.0 Hz
    psd : np.ndarray, optional
        PSD for matched filter weighting

    Returns
    -------
    Tuple[float, float, float]
        (match, time_shift, phase_shift)
        - match: Overlap value (0-1)
        - time_shift: Optimal time shift in seconds
        - phase_shift: Optimal phase shift in radians
    """
    # Convert to TimeSeries
    ts_real = TimeSeries(real_strain.astype(np.float64), delta_t=delta_t)
    ts_sim = TimeSeries(simulated_strain.astype(np.float64), delta_t=delta_t)

    # Estimate PSD if not provided
    if psd is None:
        n_samples = len(ts_real)
        if n_samples >= 4096:
            seg_len = 4096
            seg_stride = 2048
        else:
            seg_len = max(64, n_samples // 4)
            seg_stride = seg_len // 2

        psd_fs = welch(ts_real, seg_len=seg_len, seg_stride=seg_stride)
    else:
        psd_fs = FrequencySeries(psd.astype(np.float64), delta_f=1.0 / (len(psd) * delta_t))

    try:
        # Compute match using PyCBC's match function
        match_val, idx = match(ts_real, ts_sim, psd=psd_fs, low_frequency_cutoff=f_lower)

        # Convert index to time shift
        if idx > len(ts_real) // 2:
            idx = idx - len(ts_real)
        time_shift = idx * delta_t

        # Phase shift is not directly returned by match, approximate as 0
        phase_shift = 0.0

        return float(match_val), float(time_shift), phase_shift

    except Exception as e:
        print(f"Warning: Match computation failed: {e}")
        return 0.0, 0.0, 0.0


def compute_snr(
    strain: np.ndarray,
    delta_t: float,
    psd: np.ndarray = None,
    f_lower: float = 20.0
) -> float:
    """
    Compute the signal-to-noise ratio of a strain.

    Parameters
    ----------
    strain : np.ndarray
        Detector strain
    delta_t : float
        Time resolution
    psd : np.ndarray, optional
        Power spectral density
    f_lower : float
        Lower frequency cutoff. Default: 20.0 Hz

    Returns
    -------
    float
        SNR value
    """
    ts = TimeSeries(strain.astype(np.float64), delta_t=delta_t)

    if psd is None:
        n_samples = len(ts)
        if n_samples >= 4096:
            seg_len = 4096
            seg_stride = 2048
        else:
            seg_len = max(64, n_samples // 4)
            seg_stride = seg_len // 2
        psd_fs = welch(ts, seg_len=seg_len, seg_stride=seg_stride)
    else:
        psd_fs = FrequencySeries(psd.astype(np.float64), delta_f=1.0 / ts.duration)

    try:
        sigma = sigmasq(ts, psd=psd_fs, low_frequency_cutoff=f_lower)
        return float(np.sqrt(sigma))
    except Exception as e:
        print(f"Warning: SNR computation failed: {e}")
        return 0.0


def compute_residual(
    real_strain: np.ndarray,
    simulated_strain: np.ndarray,
    time_shift: float = 0.0,
    delta_t: float = 1/4096
) -> np.ndarray:
    """
    Compute residual after subtracting best-fit simulated waveform.

    Parameters
    ----------
    real_strain : np.ndarray
        Real detector strain
    simulated_strain : np.ndarray
        Simulated waveform
    time_shift : float
        Time shift to apply to simulated waveform. Default: 0.0
    delta_t : float
        Time resolution

    Returns
    -------
    np.ndarray
        Residual (real - aligned_simulated)
    """
    # Apply time shift to simulated strain
    shift_samples = int(time_shift / delta_t)

    if shift_samples != 0:
        aligned_sim = np.roll(simulated_strain, shift_samples)
        if shift_samples > 0:
            aligned_sim[:shift_samples] = 0
        else:
            aligned_sim[shift_samples:] = 0
    else:
        aligned_sim = simulated_strain

    # Ensure same length
    min_len = min(len(real_strain), len(aligned_sim))
    residual = real_strain[:min_len] - aligned_sim[:min_len]

    return residual


def compare_real_vs_simulated(
    real_data: Dict,
    simulated_data: Dict = None,
    event_name: str = None,
    detectors: List[str] = None,
    delta_t: float = 1/4096
) -> Dict:
    """
    Compare real GW event with simulated waveform.

    Parameters
    ----------
    real_data : Dict
        Output from load_real_event() or processed real data dict
        Expected keys: 'strains' (Dict[str, array]) or direct array data
    simulated_data : Dict, optional
        Pre-generated simulated data. If None, generates automatically
        using parameters from real_data.
    event_name : str, optional
        Event name (required if simulated_data is None)
    detectors : List[str], optional
        Detectors to compare. Default: all available
    delta_t : float
        Time resolution. Default: 1/4096

    Returns
    -------
    Dict containing:
        - match_values: Dict[str, float] per detector (0-1 match score)
        - snr_real: Dict[str, float] per detector
        - snr_simulated: Dict[str, float] per detector
        - time_shift: Dict[str, float] optimal time shift per detector
        - residuals: Dict[str, np.ndarray] per detector
    """
    # Handle different input formats
    if 'strains' in real_data:
        real_strains = {k: np.array(v) for k, v in real_data['strains'].items()}
        if event_name is None:
            event_name = real_data.get('event_name')
    else:
        # Assume direct waveform array format
        real_strains = real_data

    if detectors is None:
        detectors = list(real_strains.keys())

    # Generate simulated data if not provided
    if simulated_data is None:
        if event_name is None:
            raise ValueError("Either simulated_data or event_name must be provided")
        simulated_data = generate_comparison_waveform(
            event_name,
            time_resolution=delta_t,
            detectors=detectors
        )

    sim_waveforms = simulated_data['waveforms']

    # Compare each detector
    match_values = {}
    snr_real = {}
    snr_simulated = {}
    time_shifts = {}
    residuals = {}

    for det in detectors:
        if det not in real_strains or det not in sim_waveforms:
            continue

        real = real_strains[det]
        sim = sim_waveforms[det]

        # Ensure same length
        min_len = min(len(real), len(sim))
        real = real[-min_len:]  # Keep end (merger)
        sim = sim[-min_len:]

        # Compute match
        match_val, time_shift, _ = compute_match(real, sim, delta_t)
        match_values[det] = match_val
        time_shifts[det] = time_shift

        # Compute SNRs
        snr_real[det] = compute_snr(real, delta_t)
        snr_simulated[det] = compute_snr(sim, delta_t)

        # Compute residual
        residuals[det] = compute_residual(real, sim, time_shift, delta_t)

    return {
        'match_values': match_values,
        'snr_real': snr_real,
        'snr_simulated': snr_simulated,
        'time_shift': time_shifts,
        'residuals': residuals,
        'event_name': event_name,
        'simulated_params': simulated_data.get('parameters', {})
    }


def create_comparison_plot(
    real_data: Dict,
    simulated_data: Dict,
    event_name: str,
    output_path: str = None,
    show_residual: bool = True,
    show_spectrogram: bool = False,
    delta_t: float = 1/4096
) -> plt.Figure:
    """
    Create detailed comparison plot of real vs simulated data.

    Parameters
    ----------
    real_data : Dict
        Real event data
    simulated_data : Dict
        Simulated waveform data
    event_name : str
        Event name for title
    output_path : str, optional
        Path to save figure. If None, returns figure object.
    show_residual : bool
        Include residual subplot. Default: True
    show_spectrogram : bool
        Include Q-transform spectrograms. Default: False
    delta_t : float
        Time resolution. Default: 1/4096

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Get strains
    if 'strains' in real_data:
        real_strains = {k: np.array(v) for k, v in real_data['strains'].items()}
    else:
        real_strains = real_data

    sim_waveforms = simulated_data['waveforms']
    detectors = list(set(real_strains.keys()) & set(sim_waveforms.keys()))

    # Compute comparison metrics
    comparison = compare_real_vs_simulated(
        {'strains': real_strains},
        simulated_data,
        event_name,
        detectors,
        delta_t
    )

    # Create figure
    n_rows = len(detectors)
    if show_residual:
        n_rows *= 2

    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 4 * n_rows))
    if n_rows == 1:
        axes = [axes]

    # Create time axis
    for i, det in enumerate(detectors):
        real = real_strains[det]
        sim = sim_waveforms[det]

        # Ensure same length
        min_len = min(len(real), len(sim))
        real = real[-min_len:]
        sim = sim[-min_len:]

        time = np.arange(min_len) * delta_t - min_len * delta_t

        # Plot index
        plot_idx = i * 2 if show_residual else i

        ax = axes[plot_idx]
        ax.plot(time, real, 'b-', alpha=0.7, label='Real', linewidth=0.8)
        ax.plot(time, sim, 'r--', alpha=0.7, label='Simulated', linewidth=0.8)

        match_val = comparison['match_values'].get(det, 0)
        ax.set_title(f'{event_name} - {det} (Match: {match_val:.3f})')
        ax.set_xlabel('Time before merger (s)')
        ax.set_ylabel('Strain')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        # Plot residual
        if show_residual:
            ax_res = axes[plot_idx + 1]
            residual = comparison['residuals'].get(det, np.zeros_like(real))
            ax_res.plot(time, residual, 'g-', alpha=0.7, linewidth=0.8)
            ax_res.set_title(f'{det} Residual (Real - Simulated)')
            ax_res.set_xlabel('Time before merger (s)')
            ax_res.set_ylabel('Residual Strain')
            ax_res.grid(True, alpha=0.3)

    plt.tight_layout()

    # Add overall title with parameters
    params = simulated_data.get('parameters', {})
    param_str = f"M1={params.get('mass1', '?'):.1f}, M2={params.get('mass2', '?'):.1f} Msun, D={params.get('distance', '?'):.0f} Mpc"
    fig.suptitle(f'{event_name}: Real vs Simulated\n{param_str}', fontsize=14, y=1.02)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {output_path}")

    return fig


def batch_compare_events(
    event_names: List[str],
    approximant: str = 'IMRPhenomXP',
    output_dir: str = None,
    show_progress: bool = True
) -> Dict:
    """
    Compare multiple real events with simulated counterparts.

    Parameters
    ----------
    event_names : List[str]
        List of event names
    approximant : str
        Waveform approximant. Default: 'IMRPhenomXP'
    output_dir : str, optional
        Directory to save comparison plots
    show_progress : bool
        Show progress bar. Default: True

    Returns
    -------
    Dict mapping event_name -> comparison_results
    """
    try:
        from .real_data_loader import load_real_event
    except ImportError:
        from real_data_loader import load_real_event

    results = {}
    iterator = tqdm(event_names, desc="Comparing events") if show_progress else event_names

    for event_name in iterator:
        try:
            # Load real data
            real_data = load_real_event(event_name, duration=4.0)

            # Generate simulated
            simulated = generate_comparison_waveform(
                event_name,
                approximant=approximant
            )

            # Compare
            comparison = compare_real_vs_simulated(real_data, simulated, event_name)
            results[event_name] = comparison

            # Save plot if output_dir specified
            if output_dir:
                import os
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f'{event_name}_comparison.png')
                create_comparison_plot(real_data, simulated, event_name, output_path)

        except Exception as e:
            print(f"Warning: Could not compare {event_name}: {e}")
            continue

    return results


def compute_parameter_recovery_accuracy(
    event_name: str,
    inferred_params: Dict,
    param_names: List[str] = None
) -> Dict:
    """
    Compare inferred parameters with GWTC published values.

    Parameters
    ----------
    event_name : str
        Event name
    inferred_params : Dict
        Parameters inferred by neural network
    param_names : List[str], optional
        Parameters to compare. Default: all available

    Returns
    -------
    Dict containing:
        - true_values: Dict
        - inferred_values: Dict
        - absolute_errors: Dict
        - relative_errors: Dict (percentage)
        - within_90_credible: Dict[str, bool]
    """
    # Get true parameters from catalog
    true_params = get_event_parameters(event_name)

    if param_names is None:
        param_names = list(inferred_params.keys())

    # Mapping for parameter names
    name_mapping = {
        'mass1': ['mass1', 'mass_1', 'mass1_source'],
        'mass2': ['mass2', 'mass_2', 'mass2_source'],
        'distance': ['luminosity_distance', 'distance'],
        'spin1z': ['spin1z', 'spin_1z'],
        'spin2z': ['spin2z', 'spin_2z'],
        'chi_eff': ['chi_eff'],
    }

    true_values = {}
    inferred_values = {}
    absolute_errors = {}
    relative_errors = {}
    within_90_credible = {}

    for param in param_names:
        # Get true value
        true_val = None
        for mapped_name in name_mapping.get(param, [param]):
            if mapped_name in true_params and true_params[mapped_name] is not None:
                true_val = true_params[mapped_name]
                break

        if true_val is None:
            continue

        # Get inferred value
        inferred_val = inferred_params.get(param)
        if inferred_val is None:
            continue

        true_values[param] = true_val
        inferred_values[param] = inferred_val

        # Compute errors
        abs_err = abs(inferred_val - true_val)
        absolute_errors[param] = abs_err

        if true_val != 0:
            relative_errors[param] = 100 * abs_err / abs(true_val)
        else:
            relative_errors[param] = float('inf') if abs_err > 0 else 0

        # Check if within 90% credible interval
        uncertainties = true_params.get('parameter_uncertainties', {})
        param_key = name_mapping.get(param, [param])[0]
        if param_key in uncertainties:
            low, high = uncertainties[param_key]
            within_90_credible[param] = (true_val - low) <= inferred_val <= (true_val + high)
        else:
            within_90_credible[param] = None

    return {
        'true_values': true_values,
        'inferred_values': inferred_values,
        'absolute_errors': absolute_errors,
        'relative_errors': relative_errors,
        'within_90_credible': within_90_credible
    }


def print_comparison_summary(comparison: Dict) -> None:
    """
    Print a formatted summary of comparison results.

    Parameters
    ----------
    comparison : Dict
        Output from compare_real_vs_simulated()
    """
    print(f"\n{'='*60}")
    print(f"Comparison Results: {comparison.get('event_name', 'Unknown Event')}")
    print(f"{'='*60}")

    print("\nMatch Values (0-1, higher is better):")
    for det, match_val in comparison.get('match_values', {}).items():
        print(f"  {det}: {match_val:.4f}")

    print("\nSNR Values:")
    for det in comparison.get('snr_real', {}).keys():
        snr_r = comparison['snr_real'].get(det, 0)
        snr_s = comparison['snr_simulated'].get(det, 0)
        print(f"  {det}: Real={snr_r:.2f}, Simulated={snr_s:.2f}")

    print("\nOptimal Time Shifts:")
    for det, shift in comparison.get('time_shift', {}).items():
        print(f"  {det}: {shift*1000:.2f} ms")

    params = comparison.get('simulated_params', {})
    if params:
        print(f"\nSimulated Parameters:")
        print(f"  Masses: {params.get('mass1', '?'):.1f} + {params.get('mass2', '?'):.1f} Msun")
        print(f"  Distance: {params.get('distance', '?'):.0f} Mpc")
        print(f"  Chi_eff: {params.get('spin1z', 0):.2f}, {params.get('spin2z', 0):.2f}")

    print(f"{'='*60}\n")
