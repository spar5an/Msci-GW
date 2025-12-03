import numpy as np
import torch
from pycbc.waveform import get_td_waveform
from pycbc.detector import Detector
from multiprocessing import Pool, cpu_count
from functools import partial
from typing import Dict, Callable, List
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, random_split


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
        - 'spin1z', 'spin2z': Spin components
        - 'inclination', 'coa_phase': Orientation angles
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
        If True, pads waveforms to common length. If False, returns variable-length data. Default: False.
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
    - If allow_padding=False and waveforms have variable lengths, returns data as lists of tensors
    - If allow_padding=True, pads shorter waveforms with zeros at the beginning
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

    # Check if padding is needed
    if min_len != max_len and not allow_padding:
        print(f"  Variable lengths detected. Returning data in variable-length format (list of tensors).")

        # Extract variable-length data as lists
        waveforms_list = []
        params_list = []

        for w in all_successful:
            # Extract detector signals for this sample
            detector_data = []
            for det_name in detector_names:
                signal_data = w['detectors'][det_name].numpy()
                detector_data.append(torch.from_numpy(signal_data.astype(np.float32)))

            # Stack detectors: shape (num_detectors, time)
            waveforms_list.append(torch.stack(detector_data))

            # Extract parameters
            params = torch.tensor([w['params'][name] for name in param_names], dtype=torch.float32)
            params_list.append(params)

        # Create custom dataset for variable-length data
        class VariableLengthDataset(torch.utils.data.Dataset):
            def __init__(self, waveforms, params):
                self.waveforms = waveforms
                self.params = params

            def __len__(self):
                return len(self.waveforms)

            def __getitem__(self, idx):
                return self.waveforms[idx], self.params[idx]

        dataset = VariableLengthDataset(waveforms_list, params_list)
        total_size = len(dataset)
        train_size = int(train_split * total_size)
        val_size = int(val_split * total_size)
        test_size = total_size - train_size - val_size

        train_data, val_data, test_data = random_split(
            dataset, [train_size, val_size, test_size]
        )

        print(f"  Splits: train={train_size}, val={val_size}, test={test_size}")

        # Custom collate function to handle variable lengths
        def collate_fn(batch):
            """Return batch as lists instead of stacking."""
            waveforms = [item[0] for item in batch]
            params = torch.stack([item[1] for item in batch])
            return waveforms, params

        # Create DataLoaders with custom collate function
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        print(f"\nReady! Variable-length DataLoaders with batch_size={batch_size}")
        print(f"  Note: Waveforms in each batch are returned as a list of tensors")

        return {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'metadata': {
                'parameter_names': param_names,
                'num_samples': num_success,
                'num_failed': num_failed,
                'channels': detector_names,
                'train_size': train_size,
                'val_size': val_size,
                'test_size': test_size,
                'batch_size': batch_size,
                'time_resolution': time_resolution,
                'approximant': approximant,
                'f_lower': f_lower,
                'detectors': detector_names,
                'original_length_range': (int(min_len), int(max_len)),
                'chunk_size': chunk_size,
                'sky_params_provided': sky_params_provided,
                'allow_padding': allow_padding,
                'variable_length': True
            }
        }

    # Apply padding if requested or if lengths are uniform
    if min_len != max_len:
        print(f"  Applying zero-padding (padding at beginning before signal starts).")

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
            'allow_padding': allow_padding,
            'variable_length': False
        }
    }
