"""
PyCBC Data Generator for Gravitational Wave Waveforms

This module generates gravitational wave waveforms using PyCBC, projects them to
detectors (H1, L1, V1, etc.), and returns PyTorch DataLoaders ready for training.

Usage:
    from data_generator import pycbc_data_generator
    
    config = {
        'mass1': lambda size: np.random.uniform(10, 50, size=size),
        'mass2': lambda size: np.random.uniform(10, 50, size=size),
    }
    
    result = pycbc_data_generator(config, num_samples=1000)
    train_loader = result['train_loader']
"""

import numpy as np
import torch
from pycbc.waveform import get_td_waveform
from pycbc.detector import Detector
from pycbc.types import TimeSeries
from multiprocessing import Pool, cpu_count
from functools import partial
from typing import Dict, Callable, List, Tuple, Union
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt


def _validate_config(config: Dict[str, Callable]) -> None:
    """Validate configuration dictionary."""
    if not isinstance(config, dict):
        raise TypeError("config must be a dictionary")
    
    if not config:
        raise ValueError("config cannot be empty")
    
    for key, value in config.items():
        if not callable(value):
            raise TypeError(f"config['{key}'] must be callable (e.g., a distribution function)")
        
        try:
            test = value(size=2)
            if not isinstance(test, np.ndarray):
                raise TypeError(f"config['{key}']() must return numpy array")
        except Exception as e:
            raise ValueError(f"config['{key}'] failed test call: {e}")


def _generate_parameter_sets(config: Dict[str, Callable], num_samples: int) -> List[Dict]:
    """Generate all parameter combinations upfront."""
    param_arrays = {}
    for param_name, dist_func in config.items():
        param_arrays[param_name] = dist_func(size=num_samples)
    
    param_dicts = []
    for i in range(num_samples):
        param_dict = {name: float(values[i]) for name, values in param_arrays.items()}
        param_dicts.append(param_dict)
    
    return param_dicts


def _generate_single_waveform(params: Dict, time_resolution: float, approximant: str,
                              f_lower: float, detectors: List[str]) -> Dict:
    """Worker function to generate a single waveform and project to detectors."""
    try:
        hp, hc = get_td_waveform(
            approximant=approximant,
            mass1=params['mass1'],
            mass2=params['mass2'],
            spin1z=params.get('spin1z', 0.0),
            spin2z=params.get('spin2z', 0.0),
            inclination=params.get('inclination', 0.0),
            coa_phase=params.get('coa_phase', 0.0),
            delta_t=time_resolution,
            f_lower=f_lower
        )
        
        # Get sky location parameters (defaults to north pole and zero polarization)
        ra = params.get('ra', 0.0)
        dec = params.get('dec', np.pi/2)  # North pole
        polarization = params.get('polarization', 0.0)
        tc = params.get('tc', 0.0)
        
        # Project to detectors
        detector_signals = {}
        for det_name in detectors:
            detector = Detector(det_name)
            signal = detector.project_wave(hp, hc, ra, dec, polarization, method='lal')
            detector_signals[det_name] = signal
        
        result = {
            'success': True,
            'detectors': detector_signals,
            'params': params
        }
        
        return result
        
    except Exception as e:
        return {'success': False, 'error': str(e), 'params': params}


def _generate_waveforms_parallel(param_dicts: List[Dict],
                                time_resolution: float,
                                approximant: str,
                                f_lower: float,
                                num_workers: int,
                                show_progress: bool,
                                detectors: List[str]) -> List[Dict]:
    """Generate waveforms in parallel using multiprocessing."""
    worker_func = partial(_generate_single_waveform,
                          time_resolution=time_resolution,
                          approximant=approximant,
                          f_lower=f_lower,
                          detectors=detectors)
    
    with Pool(processes=num_workers) as pool:
        if show_progress:
            results = list(tqdm(
                pool.imap_unordered(worker_func, param_dicts, chunksize=100),
                total=len(param_dicts),
                desc="Generating waveforms"
            ))
        else:
            results = list(pool.imap_unordered(worker_func, param_dicts, chunksize=100))
    
    return results


def normalize_waveforms(signal_array: np.ndarray, method: str = 'global_standardize') -> Tuple[np.ndarray, List[float]]:
    """
    Normalize waveform data using various strategies.
    
    Parameters
    ----------
    signal_array : np.ndarray
        Shape: (num_samples, num_detectors, time_steps)
    method : str
        Normalization method:
        - 'per_sample_minmax': Per-sample min-max to [0, 100] (PROBLEMATIC - loses amplitude info)
        - 'global_standardize': Global z-score (mean=0, std=1) across all data (RECOMMENDED)
        - 'per_sample_standardize': Per-sample z-score (preserves relative structure)
        - 'global_minmax': Global min-max to [0, 100] (preserves relative amplitudes)
        - 'scale_constant': Divide by scale (preserves all relationships, simple)
        - 'none': No normalization
    
    Returns
    -------
    normalized_array : np.ndarray
        Normalized waveforms
    amplitude_stats : list
        Average amplitude per sample (for diagnostics)
    """
    num_samples = signal_array.shape[0]
    amplitude_stats = []
    
    if method == 'per_sample_minmax':
        print(f"  Normalizing: per-sample min-max to [0, 100]")
        print(f"  ⚠ WARNING: This removes global amplitude information!")
        for i in range(num_samples):
            sample_min = signal_array[i].min()
            sample_max = signal_array[i].max()
            
            if sample_max > sample_min:
                signal_array[i] = (signal_array[i] - sample_min) / (sample_max - sample_min) * 100
            else:
                signal_array[i] = 50.0
            
            amplitude_stats.append(signal_array[i].mean())
    
    elif method == 'global_standardize':
        print(f"  Normalizing: global z-score (mean=0, std=1)")
        print(f"  ✓ RECOMMENDED: Preserves amplitude variation between samples")
        global_mean = signal_array.mean()
        global_std = signal_array.std()
        
        if global_std > 0:
            signal_array = (signal_array - global_mean) / global_std
        
        for i in range(num_samples):
            amplitude_stats.append(signal_array[i].std())  # Track variation
        
        print(f"  Global stats: mean={global_mean:.2e}, std={global_std:.2e}")
    
    elif method == 'per_sample_standardize':
        print(f"  Normalizing: per-sample z-score (mean=0, std=1)")
        print(f"  Note: Preserves relative structure within each sample")
        for i in range(num_samples):
            sample_mean = signal_array[i].mean()
            sample_std = signal_array[i].std()
            
            if sample_std > 0:
                signal_array[i] = (signal_array[i] - sample_mean) / sample_std
            else:
                signal_array[i] = 0.0
            
            amplitude_stats.append(sample_std)  # Track original std
    
    elif method == 'global_minmax':
        print(f"  Normalizing: global min-max to [0, 100]")
        print(f"  ✓ Preserves relative amplitude differences between samples")
        global_min = signal_array.min()
        global_max = signal_array.max()
        
        if global_max > global_min:
            signal_array = (signal_array - global_min) / (global_max - global_min) * 100
        else:
            signal_array = 50.0
        
        for i in range(num_samples):
            amplitude_stats.append(signal_array[i].mean())
        
        print(f"  Global range: [{global_min:.2e}, {global_max:.2e}]")
    
    elif method == 'scale_constant':
        print(f"  Normalizing: constant scaling by 1e-21")
        print(f"  ✓ Preserves ALL relationships, simple and physically meaningful")
        
        scale_factor = 1e-21
        signal_array = signal_array / scale_factor * 10
        
        for i in range(num_samples):
            amplitude_stats.append(np.abs(signal_array[i]).max())  # Track peak values
        
        print(f"  Scale factor: {scale_factor:.2e}")
    
    elif method == 'none':
        print(f"  No waveform normalization applied")
        for i in range(num_samples):
            amplitude_stats.append(signal_array[i].mean())
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    if amplitude_stats:
        print(f"  Amplitude stats: mean={np.mean(amplitude_stats):.2f} ± {np.std(amplitude_stats):.2f}")
    
    return signal_array, amplitude_stats


def normalize_parameters(param_array: np.ndarray, param_names: List[str], method: str = 'zscore') -> Tuple[np.ndarray, Dict]:
    """
    Normalize parameter data.
    
    Parameters
    ----------
    param_array : np.ndarray
        Shape: (num_samples, num_params)
    param_names : list of str
        Parameter names
    method : str
        Normalization method:
        - 'zscore': Z-score normalization (mean=0, std=1) - RECOMMENDED for flows
        - 'minmax': Min-max to [-1, 1]
        - 'none': No normalization
    
    Returns
    -------
    normalized_array : np.ndarray
        Normalized parameters
    norm_info : dict
        Normalization statistics for each parameter
    """
    num_params = param_array.shape[1]
    param_means = param_array.mean(axis=0)
    param_stds = param_array.std(axis=0)
    param_mins = param_array.min(axis=0)
    param_maxs = param_array.max(axis=0)
    
    # Store normalization info
    param_norm_info = {}
    for j, name in enumerate(param_names):
        param_norm_info[name] = {
            'mean': float(param_means[j]),
            'std': float(param_stds[j]),
            'min': float(param_mins[j]),
            'max': float(param_maxs[j]),
            'method': method
        }
    
    if method == 'zscore':
        print(f"  Normalizing parameters: z-score (mean=0, std=1)")
        for j in range(num_params):
            if param_stds[j] > 0:
                param_array[:, j] = (param_array[:, j] - param_means[j]) / param_stds[j]
            else:
                param_array[:, j] = 0
        
        print(f"  Parameters normalized:")
        for name, info in param_norm_info.items():
            print(f"    {name}: mean={info['mean']:.4f}, std={info['std']:.4f}")
    
    elif method == 'minmax':
        print(f"  Normalizing parameters: min-max to [-1, 1]")
        for j in range(num_params):
            if param_maxs[j] > param_mins[j]:
                # Scale to [-1, 1]
                param_array[:, j] = 2 * (param_array[:, j] - param_mins[j]) / (param_maxs[j] - param_mins[j]) - 1
            else:
                param_array[:, j] = 0
        
        print(f"  Parameters normalized:")
        for name, info in param_norm_info.items():
            print(f"    {name}: [{info['min']:.4f}, {info['max']:.4f}]")
    
    elif method == 'none':
        print(f"  No parameter normalization applied")
    
    else:
        raise ValueError(f"Unknown parameter normalization method: {method}")
    
    return param_array, param_norm_info


def pycbc_data_generator(config: Dict[str, Callable],
                        num_samples: int,
                        time_resolution: float = 1/4096,
                        approximant: str = 'IMRPhenomXP',
                        f_lower: float = 40.0,
                        num_workers: int = None,
                        batch_size: int = 256,
                        chunk_size: int = 10000,
                        target_length: int = None,
                        allow_padding: bool = True,
                        train_split: float = 0.8,
                        val_split: float = 0.1,
                        show_progress: bool = True,
                        detectors: List[str] = None,
                        waveform_normalization: str = 'global_standardize',
                        parameter_normalization: str = 'zscore') -> Dict:
    """
    Generate PyCBC waveforms projected to detectors.
    Returns PyTorch DataLoaders for training, validation, and testing.
    
    Parameters
    ----------
    config : dict
        Dictionary mapping parameter names to numpy distribution functions.
        
        Required parameters:
        - 'mass1': Primary mass (solar masses)
        - 'mass2': Secondary mass (solar masses)
        
        Optional parameters:
        - 'spin1z', 'spin2z': Spin components
        - 'inclination', 'coa_phase': Orientation angles
        - 'ra': Right ascension (radians) - default: 0.0
        - 'dec': Declination (radians) - default: π/2 (north pole)
        - 'polarization': Polarization angle (radians) - default: 0.0
        - 'tc': Coalescence time - default: 0.0
        scale_constant', '
    num_samples : int
        Total number of waveforms to generate
    time_resolution : float
        Time step (delta_t). Default: 1/4096
    approximant : str
        Waveform approximant. Default: 'IMRPhenomXP'
    f_lower : float
        Lower frequency cutoff (Hz). Default: 40.0
    num_workers : int
        Parallel processes. Default: min(cpu_count(), 8)
    batch_size : int
        DataLoader batch size. Default: 256
    chunk_size : int
        Process in chunks for memory. Default: 10000
    target_length : int
        Target length for padding/truncation. If None, uses max length.
        Only used if allow_padding=True.
    allow_padding : bool
        Allow zero-padding of waveforms to common length. Default: False.
        If False and waveforms have different lengths, raises ValueError.
        If True, pads shorter waveforms with zeros at the beginning.
    waveform_normalization : str
        Waveform normalization method. Default: 'global_standardize'
        Options: 'global_standardize', 'per_sample_minmax', 'per_sample_standardize', 
                 'global_minmax', 'none'
    parameter_normalization : str
        Parameter normalization method. Default: 'zscore'
        Options: 'zscore', 'minmax', 'none'
    train_split : float
        Training fraction. Default: 0.8
    val_split : float
        Validation fraction. Default: 0.1
    detectors : list of str
        Detector names. Default: ['H1', 'L1']
    
    Returns
    -------
    dict with 'train_loader', 'val_loader', 'test_loader', 'metadata'
        
    Notes
    -----
    If ra, dec, or polarization are not provided in config, they default to:
    - ra = 0.0
    - dec = π/2 (north pole)
    - polarization = 0.0
    """
    # Validate inputs
    _validate_config(config)
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if not 0 < train_split < 1 or not 0 < val_split < 1:
        raise ValueError("train_split and val_split must be between 0 and 1")
    if train_split + val_split >= 1:
        raise ValueError("train_split + val_split must be < 1")
    
    if num_workers is None:
        num_workers = min(cpu_count(), 8)
    
    # Set default detectors
    if detectors is None:
        detectors = ['H1', 'L1']
    
    # Check which sky parameters are provided
    sky_params_provided = {
        'ra': 'ra' in config,
        'dec': 'dec' in config,
        'polarization': 'polarization' in config
    }
    
    if any(sky_params_provided.values()):
        print(f"Generating {num_samples} waveforms with projection to {detectors}")
        print(f"  Sky parameters: ra={'provided' if sky_params_provided['ra'] else 'default (0.0)'}, "
              f"dec={'provided' if sky_params_provided['dec'] else 'default (π/2)'}, "
              f"psi={'provided' if sky_params_provided['polarization'] else 'default (0.0)'}")
    else:
        print(f"Generating {num_samples} waveforms with projection to {detectors}")
        print(f"  Using default sky location: ra=0.0, dec=π/2 (north pole), psi=0.0")
    
    # Generate all parameters upfront
    param_dicts = _generate_parameter_sets(config, num_samples)
    
    # Process in chunks for memory efficiency
    all_successful = []
    all_failed = []
    num_chunks = (num_samples + chunk_size - 1) // chunk_size
    
    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, num_samples)
        chunk_params = param_dicts[chunk_start:chunk_end]
        
        if num_chunks > 1:
            print(f"\nChunk {chunk_idx + 1}/{num_chunks} ({len(chunk_params)} waveforms)...")
        
        # Generate waveforms with detector projection
        chunk_results = _generate_waveforms_parallel(
            chunk_params, time_resolution, approximant, f_lower, num_workers, show_progress=True,  detectors=detectors
        )
        
        # Single pass: separate and accumulate
        for r in chunk_results:
            if r['success']:
                all_successful.append(r)
            else:
                all_failed.append(r)
        
        if num_chunks > 1:
            print(f"  Chunk: {len([r for r in chunk_results if r['success']])} successful")
    
    if not all_successful:
        raise RuntimeError("No waveforms were successfully generated!")
    
    num_success = len(all_successful)
    num_failed = len(all_failed)
    print(f"\nGeneration complete: {num_success} successful, {num_failed} failed")
    
    # Extract everything in one pass with pre-allocated arrays
    print(f"\nProcessing {num_success} waveforms...")
    
    # Get parameter names and detector names
    param_names = list(all_successful[0]['params'].keys())
    num_params = len(param_names)
    detector_names = list(all_successful[0]['detectors'].keys())
    num_detectors = len(detector_names)
    
    print(f"  Detector channels: {detector_names}")
    
    # Pre-scan for lengths
    lengths = np.array([len(w['detectors'][detector_names[0]]) for w in all_successful])
    min_len, max_len = lengths.min(), lengths.max()
    
    print(f"  Waveform lengths: min={min_len}, max={max_len}")
    
    # Check if padding is needed
    if min_len != max_len:
        if not allow_padding:
            raise ValueError(
                f"Waveforms have variable lengths (min={min_len}, max={max_len}). "
                f"This happens because PyCBC generates waveforms of different durations "
                f"depending on the binary parameters (especially masses). "
                f"\n\nTo handle this, set allow_padding=True to zero-pad shorter waveforms. "
                f"Padding adds zeros at the beginning (before the signal), which is "
                f"physically meaningful and standard practice in GW analysis."
            )
        else:
            print(f"  WARNING: Variable lengths detected. Applying zero-padding.")
            print(f"  Padding will be added at the beginning (before signal starts).")
    
    # Determine target length
    if target_length is None:
        target_length = max_len
    elif target_length < max_len:
        print(f"  WARNING: target_length ({target_length}) < max length ({max_len}). Will truncate!")
    
    print(f"  Target length: {target_length}")
    
    # Pre-allocate arrays
    signal_array = np.zeros((num_success, num_detectors, target_length), dtype=np.float32)
    param_array = np.zeros((num_success, num_params), dtype=np.float32)
    
    # Single pass: extract everything
    for i, w in enumerate(all_successful):
        # Extract detector signals
        for j, det_name in enumerate(detector_names):
            signal_data = w['detectors'][det_name].numpy()
            current_len = len(signal_data)
            
            # Pad or truncate
            if current_len <= target_length:
                # Pad at beginning (zero-padding before signal starts)
                start_idx = target_length - current_len
                signal_array[i, j, start_idx:] = signal_data
            else:
                # Truncate (keep end where merger happens)
                signal_array[i, j] = signal_data[-target_length:]
        
        # Extract parameters
        for j, name in enumerate(param_names):
            param_array[i, j] = w['params'][name]
    
    # Normalize waveforms using specified method
    print(f"  Converting to PyTorch tensors...")
    signal_array, amplitude_stats = normalize_waveforms(signal_array, method=waveform_normalization)
    
    # Normalize parameters using specified method
    param_array, param_norm_info = normalize_parameters(param_array, param_names, method=parameter_normalization)
    
    # Convert to PyTorch
    X = torch.from_numpy(signal_array)  # (N, num_detectors, T)
    y = torch.from_numpy(param_array)    # (N, num_params) - NOW NORMALIZED
    
    print(f"  Tensors: X={X.shape}, y={y.shape}")
    
    # Create dataset and split
    dataset = TensorDataset(X, y)
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    train_data, val_data, test_data = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    print(f"  Splits: train={train_size}, val={val_size}, test={test_size}")
    
    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    print(f"\nReady! DataLoaders with batch_size={batch_size}")
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'amplitude_stats': amplitude_stats,  # NEW: Amplitude histogram data
        'metadata': {
            'parameter_names': param_names,
            'num_samples': num_success,
            'num_failed': num_failed,
            'waveform_shape': tuple(X.shape[1:]),
            'channels': detector_names,
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_size,
            'batch_size': batch_size,
            'time_resolution': time_resolution,
            'approximant': approximant,
            'f_lower': f_lower,
            'detectors': detector_names,
            'target_length': target_length,
            'original_length_range': (int(min_len), int(max_len)),
            'chunk_size': chunk_size,
            'sky_params_provided': sky_params_provided,
            'allow_padding': allow_padding,
            'waveform_normalization': waveform_normalization,
            'parameter_normalization_method': parameter_normalization,
            'parameter_normalization': param_norm_info,  # Parameter scaling info
        }
    }


def plot_amplitude_histogram(amplitude_stats, output_dir='./Plots'):
    """Plot histogram of average waveform amplitudes after normalization."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram of amplitudes
    ax = axes[0]
    ax.hist(amplitude_stats, bins=40, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    mean_amp = np.mean(amplitude_stats)
    std_amp = np.std(amplitude_stats)
    ax.axvline(mean_amp, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_amp:.2f}')
    ax.axvline(mean_amp - std_amp, color='orange', linestyle=':', linewidth=1.5, label=f'±1σ: [{mean_amp-std_amp:.2f}, {mean_amp+std_amp:.2f}]')
    ax.axvline(mean_amp + std_amp, color='orange', linestyle=':', linewidth=1.5)
    ax.set(xlabel='Average Waveform Amplitude (normalized to [0, 100])', ylabel='Density', 
           title='Distribution of Waveform Amplitudes')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    
    # Statistics box
    ax = axes[1]
    ax.axis('off')
    stats_text = f"""
    Waveform Amplitude Statistics
    
    Total samples: {len(amplitude_stats)}
    
    Min amplitude: {np.min(amplitude_stats):.2f}
    Max amplitude: {np.max(amplitude_stats):.2f}
    Mean amplitude: {mean_amp:.2f}
    Std deviation: {std_amp:.2f}
    
    25th percentile: {np.percentile(amplitude_stats, 25):.2f}
    Median: {np.median(amplitude_stats):.2f}
    75th percentile: {np.percentile(amplitude_stats, 75):.2f}
    
    Normalized to [0, 100] range
    (bijective min-max scaling per sample)
    """
    ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', 
            facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'waveform_amplitude_histogram.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Amplitude histogram saved to {output_path}")
    return output_path


if __name__ == "__main__":
    # Example 1: Basic usage with default sky location (north pole, zero polarization)
    print("=" * 80)
    print("Example 1: Basic usage with default sky location")
    print("=" * 80)
    
    config = {
        'mass1': lambda size: np.random.uniform(10, 50, size=size),
        'mass2': lambda size: np.random.uniform(10, 50, size=size),
        'spin1z': lambda size: np.random.uniform(-0.5, 0.5, size=size),
    }

    # Generate with H1 and L1 projection (default detectors)
    result = pycbc_data_generator(
        config, 
        num_samples=1000, 
        batch_size=16, 
        num_workers=2,
        allow_padding=True
    )

    # Access the loaders
    train_loader = result['train_loader']
    val_loader = result['val_loader']
    test_loader = result['test_loader']

    # Print metadata
    print(f"\nMetadata:")
    for key, value in result['metadata'].items():
        print(f"  {key}: {value}")

    # Show first batch
    print(f"\nCollecting all training data into one array:")
    all_combined = []
    for waveforms, params in train_loader:
        print(f"  Waveforms shape: {waveforms.shape}")
        print(f"  Parameters shape: {params.shape}")
        print(f"  H1 channel (first 5): {waveforms[0, 0, :5]}")
        print(f"  L1 channel (first 5): {waveforms[0, 1, :5]}")
        
        # Convert to torch tensors and concatenate
        train_params = torch.FloatTensor(params)
        train_data = torch.FloatTensor(waveforms)
        print(train_data.shape)
        
        combined = torch.cat([train_params, train_data[:, 0, :]], dim=1)    
        all_combined.append(combined)
    

    data_test = []
    params_test = []
    for waveforms, params in test_loader:
        test_params = torch.FloatTensor(params)
        test_data = torch.FloatTensor(waveforms)
        params_test.append(test_params)
        data_test.append(test_data)
    all_test_data = torch.cat(data_test, dim=0)
    all_test_params = torch.cat(params_test, dim=0)
    print(f"\nTest data shape: {len(all_test_data[:, 0])}")

    # Concatenate all batches into one big array
    all_data = torch.cat(all_combined, dim=0)
    print(f"  Total combined shape: {all_data.shape[1]}")
    #print(all_data)

'''
    # Example 2: Including sky location parameters
    print("\n" + "=" * 80)
    print("Example 2: Including sky location parameters (random positions)")
    print("=" * 80)
    
    config_with_sky = {
        'mass1': lambda size: np.random.uniform(10, 50, size=size),
        'mass2': lambda size: np.random.uniform(10, 50, size=size),
        'spin1z': lambda size: np.random.uniform(-0.5, 0.5, size=size),
        'ra': lambda size: np.random.uniform(0, 2*np.pi, size=size),
        'dec': lambda size: np.random.uniform(-np.pi/2, np.pi/2, size=size),
        'polarization': lambda size: np.random.uniform(0, 2*np.pi, size=size),
    }

    result_sky = pycbc_data_generator(
        config_with_sky, 
        num_samples=100, 
        batch_size=16,
        num_workers=2,
        allow_padding=True
    )

    print(f"\nWith random sky locations:")
    print(f"  Sky params provided: {result_sky['metadata']['sky_params_provided']}")
    for waveforms, params in result_sky['train_loader']:
        print(f"  Waveforms shape: {waveforms.shape}")
        print(f"  Parameters include: {result_sky['metadata']['parameter_names']}")
        break

    # Example 3: Using three detectors
    print("\n" + "=" * 80)
    print("Example 3: Using three detectors (H1, L1, V1)")
    print("=" * 80)
    
    config_three = {
        'mass1': lambda size: np.random.uniform(10, 50, size=size),
        'mass2': lambda size: np.random.uniform(10, 50, size=size),
        'spin1z': lambda size: np.random.uniform(-0.5, 0.5, size=size),
        'ra': lambda size: np.random.uniform(0, 2*np.pi, size=size),
        'dec': lambda size: np.random.uniform(-np.pi/2, np.pi/2, size=size),
        'polarization': lambda size: np.random.uniform(0, 2*np.pi, size=size),
    }

    result_three = pycbc_data_generator(
        config_three, 
        num_samples=50, 
        batch_size=16,
        detectors=['H1', 'L1', 'V1'],
        num_workers=2,
        allow_padding=True
    )

    print(f"\nWith three detectors:")
    for waveforms, params in result_three['train_loader']:
        print(f"  Waveforms shape: {waveforms.shape}")
        print(f"  Channels: {result_three['metadata']['channels']}")
        print(f"  H1 (first 5): {waveforms[0, 0, :5]}")
        print(f"  L1 (first 5): {waveforms[0, 1, :5]}")
        print(f"  V1 (first 5): {waveforms[0, 2, :5]}")
        break
    
    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)
'''