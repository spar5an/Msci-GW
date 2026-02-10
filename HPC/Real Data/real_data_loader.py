"""
Real GW event data loading from GWOSC/PyCBC.

This module provides functions to load real gravitational wave detector strain
data from the Gravitational Wave Open Science Center via PyCBC's Merger class.
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, Subset
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm

from pycbc.catalog import Merger
from pycbc.types import TimeSeries
from pycbc.filter import resample_to_delta_t

try:
    from .catalog_utils import get_event_parameters, parameters_to_jhpy_format
except ImportError:
    from catalog_utils import get_event_parameters, parameters_to_jhpy_format


def load_real_event(
    event_name: str,
    detectors: List[str] = None,
    sample_rate: int = 4096,
    duration: float = 32.0,
    data_source: str = 'gwosc'
) -> Dict:
    """
    Load real detector strain for a specific GW event.

    Parameters
    ----------
    event_name : str
        Event name (e.g., 'GW150914')
    detectors : List[str], optional
        Detector names. Default: ['H1', 'L1']
    sample_rate : int
        Desired sample rate in Hz. Default: 4096
    duration : float
        Duration of data to load (seconds). Default: 32.0
        Data is centered on the coalescence time.
    data_source : str
        'gwosc' for Gravitational Wave Open Science Center data

    Returns
    -------
    Dict containing:
        - strains: Dict[str, TimeSeries] mapping detector -> strain data
        - gps_times: Dict[str, float] mapping detector -> GPS start time
        - sample_rate: int
        - duration: float
        - coalescence_time: float (GPS)
        - event_name: str
        - event_params: Dict from get_event_parameters()

    Examples
    --------
    >>> data = load_real_event('GW150914', detectors=['H1', 'L1'])
    >>> print(f"H1 strain length: {len(data['strains']['H1'])} samples")
    H1 strain length: 131072 samples
    """
    if detectors is None:
        detectors = ['H1', 'L1']

    merger = Merger(event_name)
    coalescence_time = merger.time

    strains = {}
    gps_times = {}

    for det in detectors:
        try:
            # Load strain data from GWOSC
            # By default, loads data centered on the event
            strain = merger.strain(det)

            # Resample if needed
            current_rate = 1.0 / strain.delta_t
            if abs(current_rate - sample_rate) > 1:
                target_delta_t = 1.0 / sample_rate
                strain = resample_to_delta_t(strain, target_delta_t)

            # Get available data bounds
            data_start = float(strain.start_time)
            data_end = float(strain.end_time)
            available_duration = data_end - data_start

            # Calculate requested window, clipped to available data
            if duration >= available_duration:
                # Use all available data
                pass  # Keep strain as-is
            else:
                # Try to center on coalescence, but clip to available bounds
                start_time = coalescence_time - duration / 2
                end_time = coalescence_time + duration / 2

                # Clip to available data
                if start_time < data_start:
                    start_time = data_start
                    end_time = start_time + duration
                if end_time > data_end:
                    end_time = data_end
                    start_time = end_time - duration

                # Ensure we're within bounds
                start_time = max(start_time, data_start)
                end_time = min(end_time, data_end)

                # Crop to requested duration
                strain = strain.time_slice(start_time, end_time)

            strains[det] = strain
            gps_times[det] = float(strain.start_time)

        except Exception as e:
            print(f"Warning: Could not load {det} data for {event_name}: {e}")
            continue

    if not strains:
        raise ValueError(f"Could not load any detector data for {event_name}")

    # Get event parameters
    event_params = get_event_parameters(event_name)

    return {
        'strains': strains,
        'gps_times': gps_times,
        'sample_rate': sample_rate,
        'duration': duration,
        'coalescence_time': coalescence_time,
        'event_name': event_name,
        'event_params': event_params
    }


def load_multiple_events(
    event_names: List[str],
    detectors: List[str] = None,
    sample_rate: int = 4096,
    duration: float = 32.0,
    show_progress: bool = True
) -> List[Dict]:
    """
    Load real detector strain for multiple GW events.

    Parameters
    ----------
    event_names : List[str]
        List of event names
    detectors : List[str], optional
        Detector names. Default: ['H1', 'L1']
    sample_rate : int
        Desired sample rate. Default: 4096
    duration : float
        Duration per event. Default: 32.0
    show_progress : bool
        Show progress bar. Default: True

    Returns
    -------
    List[Dict]
        List of data dictionaries from load_real_event()
    """
    if detectors is None:
        detectors = ['H1', 'L1']

    results = []
    iterator = tqdm(event_names, desc="Loading events") if show_progress else event_names

    for event_name in iterator:
        try:
            data = load_real_event(
                event_name,
                detectors=detectors,
                sample_rate=sample_rate,
                duration=duration
            )
            results.append(data)
        except Exception as e:
            print(f"Warning: Skipping {event_name}: {e}")
            continue

    return results


def extract_signal_window(
    strain_data: Dict,
    window_duration: float = 2.0,
    center_on_merger: bool = True
) -> Dict:
    """
    Extract a smaller time window around the GW signal.

    Parameters
    ----------
    strain_data : Dict
        Output from load_real_event()
    window_duration : float
        Duration of window to extract (seconds). Default: 2.0
    center_on_merger : bool
        If True, window ends at coalescence time (merger at end).
        If False, window is centered on coalescence. Default: True

    Returns
    -------
    Dict with extracted strain windows and updated metadata
    """
    coalescence_time = strain_data['coalescence_time']

    if center_on_merger:
        # Window ends at merger (keeps merger at signal end)
        start_time = coalescence_time - window_duration
        end_time = coalescence_time
    else:
        # Window centered on merger
        start_time = coalescence_time - window_duration / 2
        end_time = coalescence_time + window_duration / 2

    extracted_strains = {}
    extracted_gps_times = {}

    for det, strain in strain_data['strains'].items():
        extracted = strain.time_slice(start_time, end_time)
        extracted_strains[det] = extracted
        extracted_gps_times[det] = float(extracted.start_time)

    return {
        'strains': extracted_strains,
        'gps_times': extracted_gps_times,
        'sample_rate': strain_data['sample_rate'],
        'duration': window_duration,
        'coalescence_time': coalescence_time,
        'event_name': strain_data['event_name'],
        'event_params': strain_data['event_params']
    }


def create_real_data_dataloaders(
    event_names: Union[str, List[str]],
    detectors: List[str] = None,
    signal_length: float = 2.0,
    time_resolution: float = 1/4096,
    batch_size: int = 1,
    train_split: float = 0.7,
    val_split: float = 0.15,
    param_names: List[str] = None,
    show_progress: bool = True
) -> Dict:
    """
    Create PyTorch DataLoaders from real GW events.

    Produces output compatible with JHPY's DataLoader format:
    - train_loader, val_loader, test_loader
    - metadata with same structure as pycbc_data_generator()

    Parameters
    ----------
    event_names : str or List[str]
        Single event name or list of events
    detectors : List[str], optional
        Detector names. Default: ['H1', 'L1']
    signal_length : float
        Signal window duration (seconds). Default: 2.0
    time_resolution : float
        Time step (1/sample_rate). Default: 1/4096
    batch_size : int
        DataLoader batch size. Default: 1 (real events are few)
    train_split : float
        Training fraction. Default: 0.7
    val_split : float
        Validation fraction. Default: 0.15
    param_names : List[str], optional
        Parameters to include in y tensor.
        Default: ['mass1', 'mass2', 'spin1z', 'spin2z', 'distance']
    show_progress : bool
        Show progress bar. Default: True

    Returns
    -------
    Dict compatible with JHPY format:
        - train_loader: DataLoader
        - val_loader: DataLoader
        - test_loader: DataLoader
        - metadata: Dict containing:
            - parameter_names: List[str]
            - num_samples: int
            - waveform_shape: Tuple
            - channels: List[str]
            - time_resolution: float
            - signal_length: float
            - target_length: int
            - data_source: 'real_gwtc'
            - event_names: List[str]
            - preprocessing: Dict

    Examples
    --------
    >>> # Load single event
    >>> result = create_real_data_dataloaders('GW150914')
    >>> X, y = next(iter(result['train_loader']))
    >>> print(f"Shape: {X.shape}")  # (1, 2, 8192)

    >>> # Load multiple events
    >>> events = ['GW150914', 'GW151226', 'GW170104']
    >>> result = create_real_data_dataloaders(events)
    """
    if detectors is None:
        detectors = ['H1', 'L1']

    if param_names is None:
        param_names = ['mass1', 'mass2', 'spin1z', 'spin2z', 'distance']

    # Handle single event
    if isinstance(event_names, str):
        event_names = [event_names]

    sample_rate = int(1.0 / time_resolution)
    target_length = int(signal_length * sample_rate)

    # Load all events
    all_waveforms = []
    all_params = []
    loaded_events = []

    iterator = tqdm(event_names, desc="Loading events") if show_progress else event_names

    for event_name in iterator:
        try:
            # Load full data
            data = load_real_event(
                event_name,
                detectors=detectors,
                sample_rate=sample_rate,
                duration=max(32.0, signal_length + 4.0)  # Extra buffer
            )

            # Extract signal window
            windowed = extract_signal_window(
                data,
                window_duration=signal_length,
                center_on_merger=True
            )

            # Convert to numpy arrays
            detector_strains = []
            for det in detectors:
                if det in windowed['strains']:
                    strain_array = np.array(windowed['strains'][det])

                    # Ensure exact length
                    if len(strain_array) > target_length:
                        strain_array = strain_array[-target_length:]
                    elif len(strain_array) < target_length:
                        # Pad at the beginning
                        pad_length = target_length - len(strain_array)
                        strain_array = np.pad(strain_array, (pad_length, 0), mode='constant')

                    detector_strains.append(strain_array)
                else:
                    # Fill with zeros if detector not available
                    detector_strains.append(np.zeros(target_length))

            waveform = np.stack(detector_strains, axis=0)  # Shape: (num_detectors, time_length)
            all_waveforms.append(waveform)

            # Get parameters
            params = parameters_to_jhpy_format(data['event_params'], param_names)
            all_params.append(params)

            loaded_events.append(event_name)

        except Exception as e:
            print(f"Warning: Skipping {event_name}: {e}")
            continue

    if not all_waveforms:
        raise ValueError("No events could be loaded successfully")

    # Stack into tensors
    X = torch.tensor(np.stack(all_waveforms, axis=0), dtype=torch.float32)
    y = torch.tensor(np.stack(all_params, axis=0), dtype=torch.float32)

    num_samples = len(all_waveforms)

    # Create train/val/test splits
    indices = list(range(num_samples))
    np.random.shuffle(indices)

    train_end = int(train_split * num_samples)
    val_end = int((train_split + val_split) * num_samples)

    # Ensure at least 1 sample in each split for small datasets
    if num_samples == 1:
        train_indices = indices
        val_indices = indices
        test_indices = indices
    elif num_samples == 2:
        train_indices = indices[:1]
        val_indices = indices[1:]
        test_indices = indices[1:]
    else:
        train_indices = indices[:max(1, train_end)]
        val_indices = indices[max(1, train_end):max(2, val_end)]
        test_indices = indices[max(2, val_end):] if val_end < num_samples else indices[-1:]

    # Create dataset and subsets
    full_dataset = TensorDataset(X, y)
    train_data = Subset(full_dataset, train_indices)
    val_data = Subset(full_dataset, val_indices)
    test_data = Subset(full_dataset, test_indices)

    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Build metadata (compatible with JHPY format)
    metadata = {
        'parameter_names': param_names,
        'num_samples': num_samples,
        'waveform_shape': (len(detectors), target_length),
        'channels': detectors,
        'detectors': detectors,
        'time_resolution': time_resolution,
        'signal_length': signal_length,
        'target_length': target_length,
        'batch_size': batch_size,
        'data_source': 'real_gwtc',
        'event_names': loaded_events,
        'preprocessing': {
            'whitened': False,
            'normalized': False,
            'resampled': False,
            'truncated': False
        }
    }

    print(f"\nLoaded {num_samples} real GW events")
    print(f"  Waveform shape: {X.shape}")
    print(f"  Parameters shape: {y.shape}")
    print(f"  Events: {loaded_events}")

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'metadata': metadata
    }


def save_real_dataloaders(result: Dict, save_path: str) -> None:
    """
    Save real data DataLoaders to disk.

    Uses same format as JHPY.save_dataloaders() for compatibility.

    Parameters
    ----------
    result : Dict
        Output from create_real_data_dataloaders()
    save_path : str
        Path to save file (e.g., 'real_data.pt')
    """
    print(f"Saving real data to {save_path}...")

    # Extract the underlying datasets and indices from the DataLoaders
    train_dataset = result['train_loader'].dataset
    val_dataset = result['val_loader'].dataset
    test_dataset = result['test_loader'].dataset

    # Get the full tensors from the base dataset
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


def load_real_dataloaders(
    load_path: str,
    batch_size: int = None,
    shuffle_train: bool = True
) -> Dict:
    """
    Load previously saved real data DataLoaders.

    Parameters
    ----------
    load_path : str
        Path to saved file
    batch_size : int, optional
        Override batch size
    shuffle_train : bool
        Whether to shuffle training data. Default: True

    Returns
    -------
    Dict with train_loader, val_loader, test_loader, metadata
    """
    print(f"Loading real data from {load_path}...")

    save_data = torch.load(load_path, weights_only=False)

    X = save_data['X']
    y = save_data['y']
    train_indices = save_data['train_indices']
    val_indices = save_data['val_indices']
    test_indices = save_data['test_indices']
    metadata = save_data['metadata']

    # Use saved batch_size if not provided
    if batch_size is None:
        batch_size = metadata.get('batch_size', 1)
    else:
        metadata = metadata.copy()
        metadata['batch_size'] = batch_size

    print(f"  Tensors: X={X.shape}, y={y.shape}")
    print(f"  Splits: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")

    # Recreate the dataset
    full_dataset = TensorDataset(X, y)

    # Create subsets using the saved indices
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
