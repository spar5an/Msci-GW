"""
Welcome to John-Hamza py.
This is a library containing the neural networks we are using for our project.
A lot of this is written by Claude.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist
import itertools
import random
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm
from typing import Dict, List, Callable
from functools import partial
from multiprocessing import Pool
from pycbc.waveform import get_td_waveform
from pycbc.detector import Detector
from pycbc.psd import aLIGOZeroDetHighPower
from pycbc.noise import noise_from_psd
from pycbc.types import TimeSeries


################### Miscellaneous functions ###################

def simulate_sine_wave(frequency, num_points=1000, noise_std=0.1, amplitude=1.0, phase=0):
    """
    Generate noisy sine wave with given frequency, amplitude, and phase.

    Args:
        frequency: sine wave frequency
        num_points: number of time points. Defaults to 1000.
        noise_std: Gaussian noise std dev. Defaults to 0.1.
        amplitude: sine amplitude. Defaults to 1.0.
        phase: phase shift. Defaults to 0.

    Returns:
        ndarray: Noisy sine wave observations.
    """
    t = np.linspace(0, 6*np.pi, num_points)
    signal = amplitude * np.sin(2*np.pi*frequency * t + phase)
    noise = np.random.normal(0, noise_std, num_points)
    observed_data = signal + noise
    return observed_data

def generate_sine_data(num_simulations=10000, freq_low=0.5, freq_high=5.0, phase_low=-3, phase_high=3, amplitude_low=0.5, amplitude_high=3.0, num_points=1000, noise_std=0.1, batch_size=256):
    """
    Generate vectorized training dataset with sine signals and DataLoaders.

    Args:
        num_simulations: number of samples. Defaults to 10000.
        freq_low, freq_high: frequency range. Defaults to 0.5, 5.0.
        phase_low, phase_high: phase range. Defaults to -3, 3.
        amplitude_low, amplitude_high: amplitude range. Defaults to 0.5, 3.0.
        num_points: time points per signal. Defaults to 1000.
        noise_std: noise std dev. Defaults to 0.1.
        batch_size: DataLoader batch size. Defaults to 256.

    Returns:
        dict: Amplitudes, Phases, Frequencies tensors and Train/Test/Val_Loader DataLoaders.
    """
    print(f"generating {num_simulations} samples for training")
    
    # Sample parameter ranges
    frequencies = np.random.uniform(freq_low, freq_high, num_simulations)
    phases = np.random.uniform(phase_low, phase_high, num_simulations)
    amplitudes = np.random.uniform(amplitude_low, amplitude_high, num_simulations)

    t = np.linspace(0, 6*np.pi, num_points)

    # Generate signals with broadcasting
    signal = amplitudes[:, np.newaxis] * np.sin(
        2*np.pi*frequencies[:, np.newaxis] * t + phases[:, np.newaxis]
    )

    noise = np.random.normal(0, noise_std, (num_simulations, num_points))
    X = torch.FloatTensor(signal + noise)
    y = torch.FloatTensor(np.column_stack([amplitudes, frequencies, phases]))

    frequencies = torch.FloatTensor(frequencies).unsqueeze(1)
    phases = torch.FloatTensor(phases).unsqueeze(1)
    amplitudes = torch.FloatTensor(amplitudes).unsqueeze(1)

    data = TensorDataset(X, y)  
    
    train_data, val_data, test_data = random_split(data, lengths=[0.8,0.1,0.1])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    print("data generated")
    
    return {"Amplitudes": amplitudes, "Phases": phases, "Frequencies":frequencies,
            "Train_Loader": train_loader, "Test_Loader": test_loader, "Val_Loader": val_loader}

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
            distance=params.get('distance', 410.0),  # Luminosity distance in Mpc, default 410 Mpc (GW150914)
            delta_t=time_resolution,
            f_lower=f_lower
        )
        
        # Get sky location parameters (defaults to north pole and zero polarization)
        ra = params.get('ra', 0.0)
        dec = params.get('dec', np.pi/2)  # North pole
        polarization = params.get('polarization', 0.0)
        
        # Project to detectors
        detector_signals = {}
        for det_name in detectors:
            detector = Detector(det_name)
            signal = detector.project_wave(hp, hc, ra, dec, polarization, method='lal')
            detector_signals[det_name] = signal
            
        # Add noise
        for det_name in detectors:
            signal = detector_signals[det_name]

            # Calculate PSD parameters correctly
            delta_t = signal.delta_t
            duration = len(signal) * delta_t
            delta_f = 1.0 / duration
            flen = int(duration / delta_t / 2) + 1

            # Create PSD with correct delta_f (frequency resolution, NOT delta_t)
            psd = aLIGOZeroDetHighPower(flen, delta_f, f_lower)

            # Verify compatibility: delta_t and psd.delta_f must satisfy length = int(1 / delta_t / psd.delta_f)
            expected_length = int(1.0 / delta_t / psd.delta_f)

            # Generate noise at exactly the expected_length
            noise = noise_from_psd(expected_length, delta_t, psd)

            # Adjust signal and noise to match lengths WITHOUT padding
            # Use the minimum length to avoid zero-padding artifacts
            min_length = min(len(signal), len(noise))

            # Truncate both to minimum length (keep end where merger is)
            if len(signal) > min_length:
                signal = signal[len(signal) - min_length:]
            if len(noise) > min_length:
                noise = noise[len(noise) - min_length:]

            # Set noise epoch to match signal epoch (align time series)
            noise._epoch = signal._epoch

            # Inject noise into signal (both now have same length, no padding)
            detector_signals[det_name] = signal.inject(noise)
            

        
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


def pycbc_data_generator(config: Dict[str, Callable],
                        num_samples: int,
                        time_resolution: float = 1/4096,
                        approximant: str = 'IMRPhenomXP',
                        f_lower: float = 40.0,
                        num_workers: int = None,
                        batch_size: int = 256,
                        chunk_size: int = 10000,
                        target_length: int = None,
                        allow_padding: bool = False,
                        train_split: float = 0.8,
                        val_split: float = 0.1,
                        show_progress: bool = True,
                        detectors: List[str] = None) -> Dict:
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
        - 'distance': Luminosity distance in Mpc - default: 410.0 (GW150914)
        - 'spin1z', 'spin2z': Spin components - default: 0.0
        - 'inclination', 'coa_phase': Orientation angles - default: 0.0
        - 'ra': Right ascension (radians) - default: 0.0
        - 'dec': Declination (radians) - default: π/2 (north pole)
        - 'polarization': Polarization angle (radians) - default: 0.0
        - 'tc': Coalescence time - default: 0.0
        
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
        num_workers = 1
    
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
            chunk_params, time_resolution, approximant, f_lower, num_workers, show_progress, detectors
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
    
    # Check if we have variable lengths
    if min_len != max_len:
        if not allow_padding:
            print(f"  Variable lengths detected: min={min_len}, max={max_len}")
            print(f"  allow_padding=False: Truncating all to minimum length (no padding)")
            target_length = min_len
        else:
            print(f"  WARNING: Variable lengths detected. Applying zero-padding.")
            print(f"  Padding will be added at the beginning (before signal starts).")
            if target_length is None:
                target_length = max_len
    else:
        # All same length
        target_length = min_len

    # Override target_length if provided
    if target_length is not None and target_length < max_len and allow_padding:
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

            # Truncate or pad based on allow_padding
            if current_len <= target_length:
                if allow_padding:
                    # Pad at beginning (zero-padding before signal starts)
                    start_idx = target_length - current_len
                    signal_array[i, j, start_idx:] = signal_data
                else:
                    # No padding - just use the signal as-is (already truncated to target_length)
                    signal_array[i, j, :current_len] = signal_data
            else:
                # Truncate (keep end where merger happens)
                signal_array[i, j] = signal_data[-target_length:]
        
        # Extract parameters
        for j, name in enumerate(param_names):
            param_array[i, j] = w['params'][name]
    
    print(f"  Converting to PyTorch tensors...")
    
    # Convert to PyTorch
    X = torch.from_numpy(signal_array)  # (N, num_detectors, T)
    y = torch.from_numpy(param_array)    # (N, num_params)
    
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
            'allow_padding': allow_padding
        }
    }

def save_dataloaders(result: Dict, save_path: str) -> None:
    """
    Save the datasets from a pycbc_data_generator result.
    
    Parameters
    ----------
    result : dict
        The result dictionary from pycbc_data_generator containing 
        train_loader, val_loader, test_loader, and metadata
    save_path : str
        Path where to save the data (e.g., 'my_data.pt')
    
    Examples
    --------
    >>> result = pycbc_data_generator(config, num_samples=1000)
    >>> save_dataloaders(result, 'my_waveforms.pt')
    """
    print(f"Saving datasets to {save_path}...")
    
    # Extract the underlying datasets and indices from the DataLoaders
    train_dataset = result['train_loader'].dataset
    val_dataset = result['val_loader'].dataset
    test_dataset = result['test_loader'].dataset
    
    # Get the full tensors from the base dataset
    # The subsets have .dataset (base) and .indices attributes
    base_dataset = train_dataset.dataset
    X = base_dataset.tensors[0]
    y = base_dataset.tensors[1]
    
    save_data = {
        'train_indices': train_dataset.indices,
        'val_indices': val_dataset.indices,
        'test_indices': test_dataset.indices,
        'X': X,
        'y': y,
        'metadata': result['metadata']
    }
    
    torch.save(save_data, save_path)
    print(f"  Saved successfully!")


def load_dataloaders(load_path: str, batch_size: int = None, shuffle_train: bool = True) -> Dict:
    """
    Load previously saved datasets and create DataLoaders.
    
    Parameters
    ----------
    load_path : str
        Path to the saved data file (created with save_dataloaders)
    batch_size : int, optional
        Batch size for DataLoaders. If None, uses the batch_size from metadata.
    shuffle_train : bool
        Whether to shuffle training data. Default: True
    
    Returns
    -------
    dict with 'train_loader', 'val_loader', 'test_loader', 'metadata'
    
    Examples
    --------
    >>> # Save after generation
    >>> result = pycbc_data_generator(config, num_samples=1000)
    >>> save_dataloaders(result, 'my_data.pt')
    >>> 
    >>> # Later, load the data
    >>> loaded = load_dataloaders('my_data.pt')
    >>> train_loader = loaded['train_loader']
    >>> val_loader = loaded['val_loader']
    >>> test_loader = loaded['test_loader']
    """
    print(f"Loading datasets from {load_path}...")
    
    # Load the saved data
    save_data = torch.load(load_path, weights_only=False)
    
    X = save_data['X']
    y = save_data['y']
    train_indices = save_data['train_indices']
    val_indices = save_data['val_indices']
    test_indices = save_data['test_indices']
    metadata = save_data['metadata']
    
    # Use saved batch_size if not provided
    if batch_size is None:
        batch_size = metadata['batch_size']
    else:
        # Update metadata with new batch_size
        metadata = metadata.copy()
        metadata['batch_size'] = batch_size
    
    print(f"  Tensors: X={X.shape}, y={y.shape}")
    print(f"  Splits: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")
    
    # Recreate the dataset
    full_dataset = TensorDataset(X, y)
    
    # Create subsets using the saved indices
    from torch.utils.data import Subset
    train_data = Subset(full_dataset, train_indices)
    val_data = Subset(full_dataset, val_indices)
    test_data = Subset(full_dataset, test_indices)
    
    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle_train)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    print(f"\nReady! DataLoaders with batch_size={batch_size}")
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'metadata': metadata
    }
################### Neural Network Layers ###################

class AffineCouplingLayer(nn.Module):
    """
    Affine coupling layer for normalizing flows

    Splits input, transforms one half conditioned on the other:
    x2_new = x2 * exp(s(x1, context)) + t(x1, context)
    """
    def __init__(self, dim, context_dim, hidden_dim=128, mask_type='half'):
        super().__init__()
        self.dim = dim

        # Create alternating mask for coupling layer
        self.register_buffer('mask', torch.zeros(dim))
        if mask_type in ['half', 'even']:
            self.mask[::2] = 1
        elif mask_type == 'odd':
            self.mask[1::2] = 1

        # Scale and translation networks
        self.scale_net = nn.Sequential(
            nn.Linear(dim + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(hidden_dim, dim),
            nn.Tanh()  # Stabilize training
        )

        self.translation_net = nn.Sequential(
            nn.Linear(dim + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x, context, reverse=False):
        """
        Forward (data -> latent) or reverse (latent -> data) transformation

        Args:
            x: input tensor [batch_size, dim]
            context: conditioning context (embedded data) [batch_size, context_dim]
            reverse: if True, compute inverse transformation

        Returns:
            output: transformed tensor
            log_det: log determinant of Jacobian
        """
        masked_x = x * self.mask

        scale_input = torch.cat([masked_x, context], dim=1)
        translation_input = torch.cat([masked_x, context], dim=1)

        # Compute scale and translation
        s = self.scale_net(scale_input)
        t = self.translation_net(translation_input)

        # Only apply to unmasked dimensions
        s = s * (1 - self.mask)
        t = t * (1 - self.mask)

        if not reverse:
            # Forward: x -> z
            y = x * torch.exp(s) + t
            log_det = s.sum(dim=1)
        else:
            # Reverse: z -> x
            y = (x - t) * torch.exp(-s)
            log_det = -s.sum(dim=1)

        return y, log_det

class EmbeddingNetwork(nn.Module):
    """
    Neural network to embed observed data into context vector.
    Similar to DINGO's data compression network.
    Supports multi-detector data with multiple processing strategies.
    """
    def __init__(self, data_dim=100, context_dim=64, hidden_dim=128, num_detectors=1, multi_detector_mode='concatenate'):
        super().__init__()

        self.num_detectors = num_detectors
        self.multi_detector_mode = multi_detector_mode
        self.data_dim = data_dim
        self.context_dim = context_dim

        if multi_detector_mode == 'concatenate':
            # Concatenate all detector data and process together
            self.network = nn.Sequential(
                nn.Linear(data_dim * num_detectors, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.0),  # Dropout layer (rate set by training function)
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.0),  # Dropout layer (rate set by training function)
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.0),  # Dropout layer (rate set by training function)
                nn.Linear(hidden_dim, context_dim)
            )

        elif multi_detector_mode == 'separate':
            # Process each detector separately, then combine
            self.detector_networks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(data_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.0),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.0),
                    nn.Linear(hidden_dim, context_dim // num_detectors)
                )
                for _ in range(num_detectors)
            ])

            # Combine detector embeddings
            self.combine_network = nn.Sequential(
                nn.Linear(context_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.0),
                nn.Linear(hidden_dim, context_dim)
            )

        elif multi_detector_mode == 'shared':
            # Shared network for all detectors, then combine
            self.shared_network = nn.Sequential(
                nn.Linear(data_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.0),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.0),
                nn.Linear(hidden_dim, context_dim // num_detectors)
            )

            self.combine_network = nn.Sequential(
                nn.Linear(context_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.0),
                nn.Linear(hidden_dim, context_dim)
            )
        else:
            raise ValueError(f"Unknown multi_detector_mode: {multi_detector_mode}")

    def forward(self, data):
        """
        Args:
            data: observed data
                - If num_detectors == 1: [batch_size, data_dim]
                - If num_detectors > 1: [batch_size, num_detectors, data_dim]

        Returns:
            context: embedded representation [batch_size, context_dim]
        """
        if self.num_detectors == 1:
            # Single detector case
            if self.multi_detector_mode == 'concatenate':
                return self.network(data)
            else:
                # For consistency with multi-detector modes
                return self.network(data) if hasattr(self, 'network') else self.shared_network(data)

        # Multi-detector case
        batch_size = data.shape[0]

        if self.multi_detector_mode == 'concatenate':
            # Flatten detectors: [batch, num_detectors, data_dim] -> [batch, num_detectors * data_dim]
            data_flat = data.reshape(batch_size, -1)
            return self.network(data_flat)

        elif self.multi_detector_mode == 'separate':
            # Process each detector separately
            detector_embeddings = []
            for i in range(self.num_detectors):
                embedding = self.detector_networks[i](data[:, i, :])
                detector_embeddings.append(embedding)

            # Concatenate and combine
            combined = torch.cat(detector_embeddings, dim=1)
            return self.combine_network(combined)

        elif self.multi_detector_mode == 'shared':
            # Process each detector with shared weights
            detector_embeddings = []
            for i in range(self.num_detectors):
                embedding = self.shared_network(data[:, i, :])
                detector_embeddings.append(embedding)

            # Concatenate and combine
            combined = torch.cat(detector_embeddings, dim=1)
            return self.combine_network(combined)

################### Model Classes ###################

class ParameterPredictor(nn.Module):
    """
    LSTM-based neural network for predicting scalar parameters from time series.

    Configurable model with LSTM layers followed by fully connected layers.
    Config options: lstm_hidden_size (256), lstm_num_layers (1), fc_layer_sizes ([128, 64]),
    activation ('silu'/'relu'/'tanh'), dropout (0.0).
    """
    def __init__(self, config=None):
        """
        Initialize model with optional config overrides.

        Args:
            config (dict, optional): Config dict with lstm_hidden_size, lstm_num_layers, fc_layer_sizes, activation, dropout.
        """
        super().__init__()

        # Default configuration
        default_config = {
            'lstm_hidden_size': 256,
            'lstm_num_layers': 1,
            'fc_layer_sizes': [128, 64],  # Sizes of fully connected layers before output
            'activation': 'silu',  # 'silu', 'relu', 'tanh'
            'dropout': 0.0,  # Dropout probability
        }

        if config is None:
            config = {}
        self.config = {**default_config, **config}
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=self.config['lstm_hidden_size'],
            num_layers=self.config['lstm_num_layers'],
            batch_first=True,
            dropout=self.config['dropout'] if self.config['lstm_num_layers'] > 1 else 0.0
        )

        fc_layers = []
        input_size = self.config['lstm_hidden_size']

        for hidden_size in self.config['fc_layer_sizes']:
            fc_layers.append(nn.Linear(input_size, hidden_size))

            if self.config['activation'] == 'silu':
                fc_layers.append(nn.SiLU())
            elif self.config['activation'] == 'relu':
                fc_layers.append(nn.ReLU())
            elif self.config['activation'] == 'tanh':
                fc_layers.append(nn.Tanh())

            if self.config['dropout'] > 0:
                fc_layers.append(nn.Dropout(self.config['dropout']))

            input_size = hidden_size

        fc_layers.append(nn.Linear(input_size, 3))

        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        """
        Forward pass: process time series through LSTM and FC layers.

        Args:
            x (torch.Tensor): Input shape [batch, sequence_length]

        Returns:
            torch.Tensor: Output shape [batch, 3] with predictions for amplitude, frequency, phase
        """
        x = x.unsqueeze(-1)
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        return self.fc(last_out)

class NormalizingFlow(nn.Module):
    """
    Normalizing flow: stack of coupling layers
    Transforms base distribution into complex posterior
    """
    def __init__(self, param_dim=1, context_dim=64, num_layers=6, hidden_dim=128, config=None):
        super().__init__()

        # Support both positional args and config dict
        if config is not None:
            param_dim = config.get('param_dim', param_dim)
            context_dim = config.get('context_dim', context_dim)
            num_layers = config.get('num_flow_layers', config.get('num_layers', num_layers))
            hidden_dim = config.get('hidden_dim', hidden_dim)

        # Store config for checkpointing
        self.config = {
            'param_dim': param_dim,
            'context_dim': context_dim,
            'num_flow_layers': num_layers,
            'hidden_dim': hidden_dim
        }

        self.param_dim = param_dim
        self.context_dim = context_dim

        # Stack of coupling layers with alternating masks
        self.layers = nn.ModuleList([
            AffineCouplingLayer(
                dim=param_dim,
                context_dim=context_dim,
                hidden_dim=hidden_dim,
                mask_type='even' if i % 2 == 0 else 'odd'
            )
            for i in range(num_layers)
        ])

        # Base distribution: standard Gaussian using PyTorch distributions
        self.base_dist = dist.Normal(loc=0.0, scale=1.0)

    def forward(self, params, context):
        """
        Forward pass: compute log probability of parameters given context

        Args:
            params: parameter values [batch_size, param_dim]
            context: embedded observed data [batch_size, context_dim]

        Returns:
            log_prob: log p(params | context)
        """
        z = params
        log_det_sum = 0

        # Apply flow transformations
        for layer in self.layers:
            z, log_det = layer(z, context, reverse=False)
            log_det_sum += log_det

        # Compute log probability under base distribution using PyTorch distributions
        log_prob_base = self.base_dist.log_prob(z).sum(dim=1)

        # Apply change of variables
        log_prob = log_prob_base + log_det_sum

        return log_prob

    def sample(self, context, num_samples=1):
        """
        Sample from posterior p(params | context)

        Args:
            context: embedded observed data [batch_size, context_dim]
            num_samples: number of samples per context

        Returns:
            samples: parameter samples [batch_size * num_samples, param_dim]
        """
        batch_size = context.shape[0]

        context_repeated = context.repeat_interleave(num_samples, dim=0)

        # Sample from base distribution using PyTorch distributions
        z = self.base_dist.sample((batch_size * num_samples, self.param_dim)).to(context.device)

        # Apply inverse flow transformations
        for layer in reversed(self.layers):
            z, _ = layer(z, context_repeated, reverse=True)

        return z

class DINGOModel(nn.Module):
    """
    Complete DINGO-style neural posterior estimation model

    Architecture:
    observed_data -> EmbeddingNet -> context -> NormalizingFlow -> log p(params | data)

    Supports multi-detector data with configurable processing strategies.
    """
    def __init__(self, data_dim=100, param_dim=1, context_dim=64,
                 num_flow_layers=6, hidden_dim=128, num_detectors=1,
                 multi_detector_mode='concatenate', config=None):
        super().__init__()

        # Support both positional args and config dict
        if config is not None:
            data_dim = config.get('data_dim', data_dim)
            param_dim = config.get('param_dim', param_dim)
            context_dim = config.get('context_dim', context_dim)
            num_flow_layers = config.get('num_flow_layers', num_flow_layers)
            hidden_dim = config.get('hidden_dim', hidden_dim)
            num_detectors = config.get('num_detectors', num_detectors)
            multi_detector_mode = config.get('multi_detector_mode', multi_detector_mode)

        # Store config for checkpointing
        self.config = {
            'data_dim': data_dim,
            'param_dim': param_dim,
            'context_dim': context_dim,
            'num_flow_layers': num_flow_layers,
            'hidden_dim': hidden_dim,
            'num_detectors': num_detectors,
            'multi_detector_mode': multi_detector_mode
        }

        self.embedding_net = EmbeddingNetwork(
            data_dim=data_dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            num_detectors=num_detectors,
            multi_detector_mode=multi_detector_mode
        )

        self.flow = NormalizingFlow(
            param_dim=param_dim,
            context_dim=context_dim,
            num_layers=num_flow_layers,
            hidden_dim=hidden_dim
        )

    def forward(self, params, data):
        """
        Compute log probability of parameters given data

        Args:
            params: parameter values [batch_size, param_dim]
            data: observed data
                - Single detector: [batch_size, data_dim]
                - Multi-detector: [batch_size, num_detectors, data_dim]

        Returns:
            log_prob: log p(params | data)
        """
        context = self.embedding_net(data)
        log_prob = self.flow(params, context)
        return log_prob

    def sample_posterior(self, data, num_samples=1000):
        """
        Sample from posterior p(params | data)

        Args:
            data: observed data
                - Single detector: [batch_size, data_dim]
                - Multi-detector: [batch_size, num_detectors, data_dim]
            num_samples: number of samples to draw

        Returns:
            samples: posterior samples [batch_size * num_samples, param_dim]
        """
        self.eval()
        with torch.no_grad():
            context = self.embedding_net(data)
            samples = self.flow.sample(context, num_samples=num_samples)
        return samples

################### Training Functions ###################
    
def train_predictor_model(model, optimizer, loss_fcn, n_epochs, train_dloader, val_dloader, start_epoch=0, patience=8, scheduler=None, save_best_model=True, model_path='best_predictor_model.pt', grad_clip_norm=5.0, dropout_rate=None):
    """
    Train model with early stopping, validation monitoring, gradient clipping, and optional checkpointing.

    Args:
        model (nn.Module): PyTorch model to train
        optimizer (torch.optim.Optimizer): Optimizer for training
        loss_fcn (callable): Loss function
        n_epochs (int): Number of epochs to train
        train_dloader (DataLoader): Training data loader
        val_dloader (DataLoader): Validation data loader
        start_epoch (int, optional): Starting epoch for resume. Defaults to 0.
        patience (int, optional): Epochs to wait before early stopping. Defaults to 8.
        scheduler (torch.optim.lr_scheduler, optional): LR scheduler. Defaults to None.
        save_best_model (bool, optional): Save best model checkpoint. Defaults to True.
        model_path (str, optional): Path for checkpoint. Defaults to 'best_predictor_model.pt'.
        grad_clip_norm (float, optional): Max gradient norm for clipping. Set to None to disable. Defaults to 5.0.
        dropout_rate (float, optional): Dropout probability (0.0-1.0). None uses model's default. Defaults to None.

    Returns:
        dict: Contains 'train_losses', 'val_losses', 'train_metrics', 'val_metrics', 'best_val_loss', 'best_val_epoch'.
    """
    train_losses, val_losses = [], []
    train_metrics, val_metrics = [], []
    best_val_loss = float('inf')
    best_val_epoch = 0

    # Set dropout rate if specified
    if dropout_rate is not None:
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = dropout_rate

    for epoch in range(start_epoch, start_epoch + n_epochs):
        model.train()
        tloss, vloss = 0, 0
        train_predictions = []
        train_targets = []

        for X_train, y_train in tqdm(train_dloader, desc='Epoch {}, training'.format(epoch+1)):
            optimizer.zero_grad()
            pred = model(X_train)
            loss = loss_fcn(pred, y_train)
            tloss += loss.item()
            loss.backward()

            # Gradient clipping
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

            optimizer.step()

            train_predictions.extend(pred.detach().numpy())
            train_targets.extend(y_train.numpy())

            
        model.eval()
        vloss = 0
        val_predictions = []
        val_targets = []

        with torch.no_grad():
            for X_valid, y_valid in tqdm(val_dloader, desc='Epoch {}, validation'.format(epoch+1)):
                pred = model(X_valid)
                loss = loss_fcn(pred, y_valid)
                vloss += loss.item()
                val_predictions.extend(pred.numpy())
                val_targets.extend(y_valid.numpy())

        # Calculate metrics
        train_metrics_dict = calculate_metrics(
            np.array(train_predictions), 
            np.array(train_targets)
        )
        val_metrics_dict = calculate_metrics(
            np.array(val_predictions), 
            np.array(val_targets)
        )

        # Store losses
        avg_train_loss = tloss / len(train_dloader)
        avg_val_loss = vloss / len(val_dloader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Store metrics
        train_metrics.append(train_metrics_dict)
        val_metrics.append(val_metrics_dict)

        # Print epoch results
        print(f"\n[Epoch {epoch+1:2d}]")
        print(f"Training - Loss: {avg_train_loss:.4f}, MAE: {train_metrics_dict['mae']:.4f}, "
              f"RMSE: {train_metrics_dict['rmse']:.4f}, R²: {train_metrics_dict['r2']:.4f}")
        print(f"Validation - Loss: {avg_val_loss:.4f}, MAE: {val_metrics_dict['mae']:.4f}, "
              f"RMSE: {val_metrics_dict['rmse']:.4f}, R²: {val_metrics_dict['r2']:.4f} \n")

        # Learning rate scheduling
        if scheduler is not None:
            old_lr = [param_group['lr'] for param_group in optimizer.param_groups]
            scheduler.step(avg_val_loss)
            new_lr = [param_group['lr'] for param_group in optimizer.param_groups]

            # If learning rate changed, load best model and continue training
            if old_lr != new_lr and save_best_model:
                print(f"Learning rate reduced. Loading best model from {model_path}")
                checkpoint = torch.load(model_path, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # Keep the reduced learning rate by reapplying it to all param groups
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr[optimizer.param_groups.index(param_group)]
                print("Best model loaded, resuming training\n")

            print(f"Current learning rates: {new_lr}")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            print("New best validation performance \n")
            best_val_loss = avg_val_loss
            best_val_epoch = epoch
            
            # Save the best model
            if save_best_model:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'model_config': model.config if hasattr(model, 'config') else None,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics
                }
                torch.save(checkpoint, model_path)
                print(f"Model checkpoint saved to {model_path}\n")
                
        elif best_val_epoch <= epoch - patience:
            print(f'No improvement in validation loss in last {patience} epochs \n')
            break

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'best_val_loss': best_val_loss,
        'best_val_epoch': best_val_epoch
    }
    
def train_npe_model(model, optimizer, n_epochs, train_dloader, val_dloader, start_epoch=0, patience=15, scheduler=None, save_best_model=True, model_path='best_npe_model.pt', grad_clip_norm=5.0, dropout_rate=None):
    """
    Train NPE model with log probability, early stopping, validation monitoring, and optional checkpointing.

    Args:
        model (nn.Module): PyTorch model to train
        optimizer (torch.optim.Optimizer): Optimizer for training
        n_epochs (int): Number of epochs to train
        train_dloader (DataLoader): Training data loader
        val_dloader (DataLoader): Validation data loader
        start_epoch (int, optional): Starting epoch for resume. Defaults to 0.
        patience (int, optional): Epochs to wait for improvement before early stopping. Defaults to 15.
        scheduler (torch.optim.lr_scheduler, optional): LR scheduler. Defaults to None.
        save_best_model (bool, optional): Save best model checkpoint. Defaults to True.
        model_path (str, optional): Path for checkpoint. Defaults to 'best_npe_model.pt'.
        grad_clip_norm (float, optional): Max gradient norm for clipping. Set to None to disable. Defaults to 5.0.
        dropout_rate (float, optional): Dropout probability (0.0-1.0). None uses model's default. Defaults to None.

    Returns:
        dict: Contains 'train_log_probs', 'val_log_probs', 'best_val_log_prob', 'best_val_epoch'.
    """
    train_log_probs = []
    val_log_probs = []
    best_val_log_prob = float('-inf')  # Higher is better for log probability
    best_val_epoch = 0

    # Set dropout rate if specified
    if dropout_rate is not None:
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = dropout_rate

    for epoch in range(start_epoch, start_epoch + n_epochs):
        model.train()
        train_log_prob_sum = 0

        for X_train, y_train in tqdm(train_dloader, desc='Epoch {}, training'.format(epoch+1)):
            optimizer.zero_grad()
            log_prob = model(y_train, X_train)
            loss = -log_prob.mean()  # Negative log prob for gradient descent
            
            loss.backward()
            
            # Gradient clipping
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            
            optimizer.step()
            
            train_log_prob_sum += log_prob.mean().item()

        model.eval()
        val_log_prob_sum = 0

        with torch.no_grad():
            for X_valid, y_valid in tqdm(val_dloader, desc='Epoch {}, validation'.format(epoch+1)):
                log_prob = model(y_valid, X_valid)
                val_log_prob_sum += log_prob.mean().item()

        # Compute averages
        avg_train_log_prob = train_log_prob_sum / len(train_dloader)
        avg_val_log_prob = val_log_prob_sum / len(val_dloader)
        train_log_probs.append(avg_train_log_prob)
        val_log_probs.append(avg_val_log_prob)

        # Print epoch results
        print(f"\n[Epoch {epoch+1:2d}]")
        print(f"Training - Log Prob: {avg_train_log_prob:.4f}")
        print(f"Validation - Log Prob: {avg_val_log_prob:.4f}")

        # Learning rate scheduling (higher log prob is better)
        if scheduler is not None:
            old_lr = [param_group['lr'] for param_group in optimizer.param_groups]
            scheduler.step(avg_val_log_prob)
            new_lr = [param_group['lr'] for param_group in optimizer.param_groups]

            # If learning rate changed, load best model and continue training
            if old_lr != new_lr and save_best_model:
                print(f"Learning rate reduced. Loading best model from {model_path}")
                checkpoint = torch.load(model_path, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # Keep the reduced learning rate by reapplying it to all param groups
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr[optimizer.param_groups.index(param_group)]
                print("Best model loaded, resuming training\n")

            print(f"Current learning rates: {new_lr}")

        # Early stopping check (higher log prob is better)
        if avg_val_log_prob > best_val_log_prob:
            print("New best validation performance \n")
            best_val_log_prob = avg_val_log_prob
            best_val_epoch = epoch

            # Save the best model
            if save_best_model:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_log_prob': best_val_log_prob,
                    'model_config': model.config if hasattr(model, 'config') else None,
                    'train_log_probs': train_log_probs,
                    'val_log_probs': val_log_probs
                }
                torch.save(checkpoint, model_path)
                print(f"Model checkpoint saved to {model_path}\n")

        elif best_val_epoch <= epoch - patience:
            print(f'No improvement in validation log prob in last {patience} epochs \n')
            break

    return {
        'train_log_probs': train_log_probs,
        'val_log_probs': val_log_probs,
        'best_val_log_prob': best_val_log_prob,
        'best_val_epoch': best_val_epoch
    }
    
################### Other Neural Network Functions ###################

def calculate_metrics(predictions, targets):
    """
    Calculate MAE, RMSE, and R² metrics.

    Args:
        predictions (np.ndarray): Model predictions
        targets (np.ndarray): Target values

    Returns:
        dict: Contains 'mae', 'rmse', 'r2'.
    """
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }

def load_predictor(model_path='best_predictor_model.pt'):
    """
    Load saved ParameterPredictor checkpoint.

    Args:
        model_path (str, optional): Path to checkpoint. Defaults to 'best_predictor_model.pt'.

    Returns:
        tuple: (model, checkpoint dict)
    """
    checkpoint = torch.load(model_path, weights_only=False)

    model = ParameterPredictor(checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded ParameterPredictor from {model_path}")
    print(f"  Best epoch: {checkpoint['epoch'] + 1}")
    print(f"  Best validation loss: {checkpoint['best_val_loss']:.4f}")

    return model, checkpoint

def load_npe(model_path='best_npe_model.pt', model_class=DINGOModel):
    """
    Load saved NPE model checkpoint (NormalizingFlow or DINGOModel).

    Args:
        model_path (str, optional): Path to checkpoint. Defaults to 'best_npe_model.pt'.
        model_class: NPE model class (DINGOModel or NormalizingFlow). Defaults to DINGOModel.

    Returns:
        tuple: (model, checkpoint dict)
    """
    checkpoint = torch.load(model_path, weights_only=False)

    model = model_class(config=checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded {model_class.__name__} from {model_path}")
    print(f"  Best epoch: {checkpoint['epoch'] + 1}")
    print(f"  Best validation log prob: {checkpoint['best_val_log_prob']:.4f}")

    return model, checkpoint

def create_dingo_from_data(dataloader_result, param_dim=None, context_dim=64,
                           num_flow_layers=6, hidden_dim=128, multi_detector_mode='concatenate'):
    """
    Create a DINGOModel with dimensions automatically inferred from dataloader metadata.

    Args:
        dataloader_result: Result dict from pycbc_data_generator or load_dataloaders
        param_dim: Number of parameters to infer. If None, inferred from metadata
        context_dim: Context dimension. Defaults to 64
        num_flow_layers: Number of flow layers. Defaults to 6
        hidden_dim: Hidden dimension. Defaults to 128
        multi_detector_mode: 'concatenate', 'separate', or 'shared'. Defaults to 'concatenate'

    Returns:
        DINGOModel: Model configured with correct dimensions

    Examples:
        >>> data = load_dataloaders('my_data.pt')
        >>> model = create_dingo_from_data(data, context_dim=128, num_flow_layers=8)
    """
    metadata = dataloader_result['metadata']

    # Infer dimensions from metadata
    if 'waveform_shape' in metadata:
        # Format: (num_detectors, time_length)
        num_detectors, data_dim = metadata['waveform_shape']
    elif 'channels' in metadata and 'target_length' in metadata:
        num_detectors = len(metadata['channels'])
        data_dim = metadata['target_length']
    else:
        raise ValueError("Cannot infer data dimensions from metadata. Missing 'waveform_shape' or 'channels'/'target_length'")

    if param_dim is None:
        if 'parameter_names' in metadata:
            param_dim = len(metadata['parameter_names'])
        else:
            raise ValueError("Cannot infer param_dim from metadata. Please specify explicitly.")

    print(f"Creating DINGOModel with inferred dimensions:")
    print(f"  data_dim (time length per detector): {data_dim}")
    print(f"  num_detectors: {num_detectors}")
    print(f"  param_dim: {param_dim}")
    print(f"  context_dim: {context_dim}")
    print(f"  num_flow_layers: {num_flow_layers}")
    print(f"  multi_detector_mode: {multi_detector_mode}")

    model = DINGOModel(
        data_dim=data_dim,
        param_dim=param_dim,
        context_dim=context_dim,
        num_flow_layers=num_flow_layers,
        hidden_dim=hidden_dim,
        num_detectors=num_detectors,
        multi_detector_mode=multi_detector_mode
    )

    return model

def predictor_hyperparameter_search(param_grid, train_loader, val_loader, n_epochs=20, n_trials=None, model_path='best_predictor_model.pt'):
    """
    Search for best hyperparameter configuration.

    Args:
        param_grid: Dict of parameter names to value lists
        train_loader: Training data loader
        val_loader: Validation data loader
        n_epochs: Epochs per configuration. Defaults to 20.
        n_trials: Random trials to try; None = all combinations. Defaults to None.
        model_path: Path to save best model. Defaults to 'best_predictor_model.pt'.

    Returns:
        tuple: (best_config dict, results list)
    """

    results = []
    best_val_loss = float('inf')
    best_config = None

    # Generate all combinations or sample randomly
    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]

    if n_trials is None:
        # Try all combinations (grid search)
        all_combinations = list(itertools.product(*param_values))
    else:
        # Random search: sample n_trials random combinations
        all_combinations = []
        for _ in range(n_trials):
            combo = tuple(random.choice(values) for values in param_values)
            all_combinations.append(combo)
    
    print(f"Testing {len(all_combinations)} configurations...\n")
    
    for i, combo in enumerate(all_combinations):
        config = dict(zip(param_names, combo))

        print(f"{'='*60}")
        print(f"Trial {i+1}/{len(all_combinations)}")
        print(f"Config: {config}")
        print(f"{'='*60}")

        model = ParameterPredictor(config)
        lr = config.get('learning_rate', 0.01)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        )

        loss_fcn = nn.MSELoss()

        try:
            outputs = train_predictor_model(
                model,
                optimizer,
                loss_fcn,
                n_epochs,
                train_loader,
                val_loader,
                patience=8,
                scheduler=scheduler,
                save_best_model=False
            )

            final_val_loss = min(outputs['val_losses'])
            final_val_metrics = outputs['val_metrics'][outputs['val_losses'].index(final_val_loss)]

            result = {
                'config': config.copy(),
                'best_val_loss': final_val_loss,
                'best_val_mae': final_val_metrics['mae'],
                'best_val_rmse': final_val_metrics['rmse'],
                'best_val_r2': final_val_metrics['r2'],
                'n_epochs_trained': len(outputs['val_losses'])
            }
            results.append(result)

            print(f"\nFinal validation loss: {final_val_loss:.4f}")
            print(f"Best validation R²: {final_val_metrics['r2']:.4f}\n")

            if final_val_loss < best_val_loss:
                best_val_loss = final_val_loss
                best_config = config.copy()
                checkpoint = {
                    'epoch': len(outputs['val_losses']) - 1,
                    'model_state_dict': model.state_dict(),
                    'best_val_loss': best_val_loss,
                    'model_config': model.config if hasattr(model, 'config') else None,
                    'train_losses': outputs['train_losses'],
                    'val_losses': outputs['val_losses'],
                    'train_metrics': outputs['train_metrics'],
                    'val_metrics': outputs['val_metrics']
                }
                torch.save(checkpoint, model_path)
                print(f"*** New best configuration found! ***\n")
        
        except Exception as e:
            print(f"Error training with config {config}: {e}\n")
            continue

    print(f"\n{'='*60}")
    print("HYPERPARAMETER SEARCH COMPLETE")
    print(f"{'='*60}")
    print(f"\nBest configuration:")
    for key, value in best_config.items():
        print(f"  {key}: {value}")
    print(f"\nBest validation loss: {best_val_loss:.4f}")

    results.sort(key=lambda x: x['best_val_loss'])
    
    return best_config, results

def npe_hyperparameter_search(param_grid, train_loader, val_loader, model_class=DINGOModel, n_epochs=20, n_trials=None, model_path='best_npe_model.pt'):
    """
    Hyperparameter search optimized for NPE models (NormalizingFlow, DINGOModel).

    Args:
        param_grid: Dict of parameter names to value lists
        train_loader: Training data loader
        val_loader: Validation data loader
        model_class: NPE model class (DINGOModel or NormalizingFlow). Defaults to DINGOModel.
        n_epochs: Epochs per configuration. Defaults to 20.
        n_trials: Random trials to try; None = all combinations. Defaults to None.
        model_path: Path to save best model. Defaults to 'best_npe_model.pt'.

    Returns:
        tuple: (best_config dict, results list sorted by validation log prob)
    """
    results = []
    best_val_log_prob = float('-inf')
    best_config = None

    # Generate all combinations or sample randomly
    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]

    if n_trials is None:
        # Try all combinations (grid search)
        all_combinations = list(itertools.product(*param_values))
    else:
        # Random search: sample n_trials random combinations
        all_combinations = []
        for _ in range(n_trials):
            combo = tuple(random.choice(values) for values in param_values)
            all_combinations.append(combo)

    print(f"Testing {len(all_combinations)} configurations with {model_class.__name__}...\n")

    for i, combo in enumerate(all_combinations):
        config = dict(zip(param_names, combo))

        print(f"{'='*60}")
        print(f"Trial {i+1}/{len(all_combinations)}")
        print(f"Config: {config}")
        print(f"{'='*60}")

        model = model_class(config=config)
        lr = config.get('learning_rate', 0.001)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=2,
            min_lr=1e-7,
        )

        grad_clip_norm = config.get('grad_clip_norm', 5.0)
        dropout_rate = config.get('dropout_rate', None)

        try:
            outputs = train_npe_model(
                model,
                optimizer,
                n_epochs,
                train_loader,
                val_loader,
                patience=8,
                scheduler=scheduler,
                grad_clip_norm=grad_clip_norm,
                dropout_rate=dropout_rate,
                save_best_model=False
            )

            final_val_log_prob = max(outputs['val_log_probs'])

            result = {
                'config': config.copy(),
                'best_val_log_prob': final_val_log_prob,
                'n_epochs_trained': len(outputs['val_log_probs'])
            }
            results.append(result)

            print(f"\nFinal validation log prob: {final_val_log_prob:.4f}\n")

            if final_val_log_prob > best_val_log_prob:
                best_val_log_prob = final_val_log_prob
                best_config = config.copy()
                checkpoint = {
                    'epoch': len(outputs['val_log_probs']) - 1,
                    'model_state_dict': model.state_dict(),
                    'best_val_log_prob': best_val_log_prob,
                    'model_config': model.config if hasattr(model, 'config') else None,
                    'train_log_probs': outputs['train_log_probs'],
                    'val_log_probs': outputs['val_log_probs']
                }
                torch.save(checkpoint, model_path)
                print(f"*** New best configuration found! ***\n")

        except Exception as e:
            print(f"Error training with config {config}: {e}\n")
            continue

    print(f"\n{'='*60}")
    print("NPE HYPERPARAMETER SEARCH COMPLETE")
    print(f"{'='*60}")
    print(f"\nBest configuration:")
    for key, value in best_config.items():
        print(f"  {key}: {value}")
    print(f"\nBest validation log prob: {best_val_log_prob:.4f}")

    results.sort(key=lambda x: x['best_val_log_prob'], reverse=True)

    return best_config, results

def infer_NPE(model, observed_data, num_samples=5000):
    """
    Perform inference of NPE
    
    Args:
        model: trained model
        observed_data: observed sine wave [data_dim]
        num_samples: number of posterior samples
    
    Returns:
        samples: posterior samples [num_samples, param_dim]
        statistics: dict with mean, median, std, quantiles
    """
    model.eval()

    data_tensor = torch.FloatTensor(observed_data).unsqueeze(0)

    with torch.no_grad():
        samples = model.sample_posterior(data_tensor, num_samples=num_samples)
        samples = samples.numpy()

    if samples.shape[1] == 1:
        samples = samples.flatten()
        statistics = {
            'mean': np.mean(samples),
            'median': np.median(samples),
            'std': np.std(samples),
            'q05': np.percentile(samples, 5),
            'q95': np.percentile(samples, 95),
        }
    else:
        statistics = None
    
    return samples, statistics
