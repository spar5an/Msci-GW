"""
DataPipeline — unified dataset generation, saving, and loading.

This is the single entry point for all data operations across all physics
modes (GR, Modified Gravity, Lorentz Violation) and noise backends
(standard analytical PSD, real O4 PSDs).

Usage
-----
Generate and save::

    from DataPipeline import generate_dataset, save_dataset, build_dataset_path
    import numpy as np

    config = {
        'mode': 'gr',
        'generator_backend': 'standard',
        'num_samples': 30000,
        'add_noise': True,
        'whiten': True,
        'detectors': ['H1', 'L1'],
        'num_workers': 4,
    }
    result = generate_dataset(config)
    path = build_dataset_path(config, output_dir='../datasets')
    save_dataset(result, path, config)

Load for training::

    from DataPipeline import load_dataset
    result = load_dataset('../datasets/dataset_gr_standard_N30000_abc123.pt')
    train_loader = result['train_loader']

Config keys
-----------
Required:
    mode (str)              : 'gr', 'mg', or 'lv'
    num_samples (int)       : Number of waveforms to generate

Optional (defaults shown):
    generator_backend (str) : 'standard' (default) or 'real_psd'
    psd_csv (str)           : Path to PSD CSV; required when backend='real_psd'
    approximant (str)       : Waveform approximant
                              gr/mg default: 'IMRPhenomXP'/'IMRPhenomD'
    f_lower (float)         : Lower frequency cutoff. Default: 40.0 (gr), 30.0 (mg/lv)
    f_final (float)         : Upper cutoff for mg/lv. Default: 2048.0
    signal_length (float)   : Waveform duration in seconds. Default: 2.0
    detectors (list)        : Detector list. Default: ['H1', 'L1']
    add_noise (bool)        : Inject detector noise. Default: True
    whiten (bool)           : Whiten waveforms. Default: True
    num_workers (int)       : Multiprocessing workers. Default: auto (<=8)
    batch_size (int)        : DataLoader batch size. Default: 256
    chunk_size (int)        : Generation chunk size. Default: 5000
    train_split (float)     : Fraction for training. Default: 0.8
    val_split (float)       : Fraction for validation. Default: 0.1

MG-specific (mode='mg'):
    lambda_g (float)        : Graviton Compton wavelength in metres (fixed); if
                              None, 'lambda_g' must be a per-sample key in config
    lambda_g_range (tuple)  : (min, max) for per-sample uniform sampling

LV-specific (mode='lv'):
    alpha_lv (float)        : LV dispersion exponent (required for lv mode)
    a_lv_range (tuple)      : (min, max) for per-sample A_lv sampling
    lambda_g_range (tuple)  : (min, max) for per-sample lambda_g sampling (optional)
"""

import os
import sys
import hashlib
import json
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Any, Optional

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _hpc_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _import_generators():
    hpc = _hpc_dir()
    if hpc not in sys.path:
        sys.path.insert(0, hpc)
    import DataGenerator as dg
    import DataGeneratorRealPSD as dg_psd
    return dg, dg_psd


def _make_serialisable(value):
    if isinstance(value, dict):
        return {str(k): _make_serialisable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_make_serialisable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if callable(value):
        return '<callable>'
    return value


def _settings_hash(settings: dict) -> str:
    serialisable = _make_serialisable(settings)
    blob = json.dumps(serialisable, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _extract_tensors_from_loader(loader: DataLoader):
    """Iterate through a DataLoader and return (X, y) tensors."""
    Xs, ys = [], []
    for batch in loader:
        Xs.append(batch[0])
        ys.append(batch[1])
    return torch.cat(Xs, dim=0), torch.cat(ys, dim=0)


def _build_param_config(config: dict) -> dict:
    """
    Build the parameter sampling config dict for the generators.
    Separates pipeline-level keys from per-parameter distribution lambdas.
    """
    _pipeline_keys = {
        'mode', 'generator_backend', 'num_samples', 'psd_csv',
        'approximant', 'f_lower', 'f_final', 'signal_length',
        'detectors', 'add_noise', 'whiten', 'num_workers',
        'batch_size', 'chunk_size', 'train_split', 'val_split',
        'lambda_g', 'lambda_g_range', 'alpha_lv', 'a_lv_range',
    }
    return {k: v for k, v in config.items() if k not in _pipeline_keys}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_dataset_path(config: dict, output_dir: str = None) -> str:
    """
    Build a deterministic file path for a given config.

    Format: dataset_{mode}_{backend}_N{n}_{hash12}.pt
    Default output_dir: <repo>/Msci-GW/datasets/
    """
    mode    = config.get('mode', 'gr')
    backend = config.get('generator_backend', 'standard')
    n       = config.get('num_samples', 0)

    settings = {
        'mode':              mode,
        'generator_backend': backend,
        'num_samples':       int(n),
        'approximant':       config.get('approximant'),
        'f_lower':           config.get('f_lower'),
        'f_final':           config.get('f_final'),
        'signal_length':     config.get('signal_length', 2.0),
        'detectors':         config.get('detectors', ['H1', 'L1']),
        'add_noise':         config.get('add_noise', True),
        'whiten':            config.get('whiten', True),
        'psd_csv':           config.get('psd_csv') if backend == 'real_psd' else None,
        'lambda_g':          config.get('lambda_g'),
        'lambda_g_range':    config.get('lambda_g_range'),
        'alpha_lv':          config.get('alpha_lv'),
        'a_lv_range':        config.get('a_lv_range'),
    }
    h = _settings_hash(settings)
    filename = f"dataset_{mode}_{backend}_N{n}_{h}.pt"

    if output_dir is None:
        output_dir = os.path.join(_hpc_dir(), '..', 'datasets')
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, filename)


def generate_dataset(config: dict) -> dict:
    """
    Generate a dataset for any physics mode and noise backend.

    Returns the standard result dict::

        {
            'train_loader': DataLoader,
            'val_loader':   DataLoader,
            'test_loader':  DataLoader,
            'metadata':     dict,
        }

    See module docstring for full config key reference.
    """
    dg, dg_psd = _import_generators()

    mode    = config.get('mode', 'gr').lower()
    backend = config.get('generator_backend', 'standard').lower()
    n       = config['num_samples']

    param_config = _build_param_config(config)

    # Common kwargs shared across generators
    common = dict(
        num_samples=n,
        detectors=config.get('detectors', ['H1', 'L1']),
        add_noise=config.get('add_noise', True),
        num_workers=config.get('num_workers', None),
        batch_size=config.get('batch_size', 256),
        chunk_size=config.get('chunk_size', 5000),
        train_split=config.get('train_split', 0.8),
        val_split=config.get('val_split', 0.1),
        signal_length=config.get('signal_length', 2.0),
        show_progress=True,
    )

    if mode == 'gr':
        common['approximant'] = config.get('approximant', 'IMRPhenomXP')
        common['f_lower']     = config.get('f_lower', 40.0)
        common['whiten']      = config.get('whiten', True)

        if backend == 'real_psd':
            psd_csv = config.get('psd_csv')
            if not psd_csv:
                raise ValueError("config['psd_csv'] is required when generator_backend='real_psd'")
            result = dg_psd.pycbc_data_generator_real_psd(
                param_config, psd_csv=psd_csv, **common)
        else:
            result = dg.pycbc_data_generator(param_config, **common)

    elif mode == 'mg':
        common['approximant'] = config.get('approximant', 'IMRPhenomD')
        common['f_lower']     = config.get('f_lower', 30.0)
        common['f_final']     = config.get('f_final', 2048.0)
        common['whiten']      = config.get('whiten', True)

        # Inject lambda_g range into param_config if not already there
        if 'lambda_g' not in param_config and config.get('lambda_g_range'):
            lo, hi = config['lambda_g_range']
            param_config['lambda_g'] = lambda size, lo=lo, hi=hi: np.random.uniform(lo, hi, size=size)

        if backend == 'real_psd':
            psd_csv = config.get('psd_csv')
            if not psd_csv:
                raise ValueError("config['psd_csv'] is required when generator_backend='real_psd'")
            result = dg_psd.pycbc_modified_data_generator_real_psd(
                param_config, psd_csv=psd_csv,
                lambda_g=config.get('lambda_g'), **common)
        else:
            result = dg.pycbc_modified_data_generator(
                param_config, lambda_g=config.get('lambda_g'), **common)

    elif mode == 'lv':
        if backend == 'real_psd':
            raise ValueError(
                "generator_backend='real_psd' is not yet supported for mode='lv'. "
                "Use generator_backend='standard'.")

        alpha_lv = config.get('alpha_lv')
        if alpha_lv is None:
            raise ValueError("config['alpha_lv'] is required for mode='lv'")

        common['approximant'] = config.get('approximant', 'IMRPhenomD')
        common['f_lower']     = config.get('f_lower', 30.0)
        common['f_final']     = config.get('f_final', 2048.0)
        # LV generator does not accept 'whiten' kwarg

        # Inject per-sample A_lv range if not already in param_config
        if 'A_lv' not in param_config and config.get('a_lv_range'):
            lo, hi = config['a_lv_range']
            param_config['A_lv'] = lambda size, lo=lo, hi=hi: np.random.uniform(lo, hi, size=size)

        # Inject per-sample lambda_g range if provided
        if 'lambda_g' not in param_config and config.get('lambda_g_range'):
            lo, hi = config['lambda_g_range']
            param_config['lambda_g'] = lambda size, lo=lo, hi=hi: np.random.uniform(lo, hi, size=size)

        result = dg.pycbc_lorentz_violation_data_generator(
            param_config,
            alpha_lv=alpha_lv,
            lambda_g=config.get('lambda_g'),
            **common,
        )

    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'gr', 'mg', or 'lv'.")

    return result


def save_dataset(result: dict, path: str, config: dict = None) -> None:
    """
    Save a dataset result dict to a .pt file.

    Extracts tensors from the DataLoader objects (DataLoaders are not
    directly serialisable) and stores everything needed to reconstruct them.

    Parameters
    ----------
    result : dict
        Output of generate_dataset() — must contain 'train_loader',
        'val_loader', 'test_loader', 'metadata'.
    path : str
        Destination file path (should end in .pt).
    config : dict, optional
        The config used to generate this dataset; stored alongside the data
        for provenance and future reference.
    """
    print(f"Extracting tensors from DataLoaders...")
    train_X, train_y = _extract_tensors_from_loader(result['train_loader'])
    val_X,   val_y   = _extract_tensors_from_loader(result['val_loader'])
    test_X,  test_y  = _extract_tensors_from_loader(result['test_loader'])

    payload = {
        'train_X':  train_X,
        'train_y':  train_y,
        'val_X':    val_X,
        'val_y':    val_y,
        'test_X':   test_X,
        'test_y':   test_y,
        'metadata': result.get('metadata', {}),
        'config':   _make_serialisable(config) if config is not None else {},
    }

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(payload, path)
    sizes = (len(train_X), len(val_X), len(test_X))
    print(f"Dataset saved to: {path}")
    print(f"  Samples — train: {sizes[0]}, val: {sizes[1]}, test: {sizes[2]}")
    print(f"  Waveform dim: {train_X.shape[1]}, Param dim: {train_y.shape[1]}")


def load_dataset(path: str, batch_size: int = 256) -> dict:
    """
    Load a previously saved dataset and reconstruct DataLoaders.

    Returns the same dict structure as generate_dataset()::

        {
            'train_loader': DataLoader,
            'val_loader':   DataLoader,
            'test_loader':  DataLoader,
            'metadata':     dict,
            'config':       dict,   # original generation config (if saved)
        }

    Parameters
    ----------
    path : str
        Path to a .pt file saved by save_dataset().
    batch_size : int
        Batch size for the reconstructed DataLoaders. Default: 256.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    print(f"Loading dataset from: {path}")
    payload = torch.load(path, map_location='cpu', weights_only=False)

    def _make_loader(X, y, shuffle):
        ds = TensorDataset(X, y)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=False)

    result = {
        'train_loader': _make_loader(payload['train_X'], payload['train_y'], shuffle=True),
        'val_loader':   _make_loader(payload['val_X'],   payload['val_y'],   shuffle=False),
        'test_loader':  _make_loader(payload['test_X'],  payload['test_y'],  shuffle=False),
        'metadata':     payload.get('metadata', {}),
        'config':       payload.get('config', {}),
    }

    train_X = payload['train_X']
    print(f"Dataset loaded.")
    print(f"  Samples — train: {len(payload['train_X'])}, "
          f"val: {len(payload['val_X'])}, test: {len(payload['test_X'])}")
    print(f"  Waveform dim: {train_X.shape[1]}, "
          f"Param dim: {payload['train_y'].shape[1]}")
    return result
