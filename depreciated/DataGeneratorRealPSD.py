"""
DataGeneratorRealPSD — Gravitational wave waveform generator using real detector PSDs.

Instead of using the analytical aLIGOZeroDetHighPower PSD (from PyCBC) to
generate noise, this module reads cleaned PSDs exported by the
RealDataGenerator*.py scripts and uses them to colour Gaussian noise.

This produces training data whose noise closely matches the actual O4
detector noise seen at the time of real events.

Usage:
    from DataGeneratorRealPSD import pycbc_data_generator_real_psd

    config = {
        'mass1': lambda size: np.random.uniform(10, 50, size=size),
        'mass2': lambda size: np.random.uniform(10, 50, size=size),
    }

    result = pycbc_data_generator_real_psd(
        config,
        num_samples=1000,
        psd_csv='test7_o4_psds_all.csv',      # or any PSD CSV from the generators
    )
    train_loader = result['train_loader']
"""

import numpy as np
import pandas as pd
import torch
from pycbc.waveform import get_td_waveform, get_fd_waveform
from pycbc.detector import Detector
from multiprocessing import Pool, cpu_count
from functools import partial
from typing import Dict, Callable, List, Tuple, Union
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, random_split
from pycbc.types import TimeSeries, FrequencySeries
from pycbc.psd import interpolate
from pycbc.noise import noise_from_psd
from pathlib import Path
import warnings
from time import perf_counter

# Re-use normalisation and validation helpers from DataGenerator
from DataGenerator import (
    _validate_config,
    _generate_parameter_sets,
    normalize_waveforms,
    normalize_parameters,
    whiten_waveform,
    _chirp_mass,
    _D_alpha,
    _additional_phase,
)


def _format_elapsed_time(seconds: float) -> str:
    """Format elapsed time for readable runtime logging."""
    total_seconds = max(0.0, float(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours >= 1:
        return f"{int(hours)}h {int(minutes)}m {secs:.2f}s"
    if minutes >= 1:
        return f"{int(minutes)}m {secs:.2f}s"
    return f"{secs:.2f}s"


# ═══════════════════════════════════════════════════════════════════════════════
# PSD LOADING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def load_real_psds(psd_csv: str) -> Dict[str, List[Dict]]:
    """
    Load cleaned PSDs from a CSV file produced by the RealDataGenerator scripts.

    The CSV is expected to have columns:
        event_rank, event_name, gps, detector, frequency, psd

    Returns
    -------
    dict mapping detector name -> list of {'event_name', 'gps', 'freqs', 'psd'}
    """
    path = Path(psd_csv)
    if not path.exists():
        raise FileNotFoundError(f"PSD CSV not found: {psd_csv}")

    df = pd.read_csv(path)
    required = {'detector', 'frequency', 'psd'}
    if not required.issubset(df.columns):
        raise ValueError(
            f"PSD CSV must contain columns {required}, got {set(df.columns)}"
        )

    psds_by_det: Dict[str, List[Dict]] = {}

    for (det, ename), grp in df.groupby(['detector', 'event_name']):
        entry = {
            'event_name': ename,
            'gps': grp['gps'].iloc[0] if 'gps' in grp.columns else 0.0,
            'freqs': grp['frequency'].values.astype(np.float64),
            'psd': grp['psd'].values.astype(np.float64),
        }
        psds_by_det.setdefault(det, []).append(entry)

    if not psds_by_det:
        raise ValueError(f"No PSD data found in {psd_csv}")

    for det, entries in psds_by_det.items():
        print(f"  Loaded {len(entries)} PSDs for {det}")

    return psds_by_det


def _pick_random_psd(psds_by_det: Dict[str, List[Dict]], det_name: str,
                     rng: np.random.RandomState) -> Dict:
    """Pick a random PSD entry for a given detector."""
    entries = psds_by_det.get(det_name)
    if entries is None or len(entries) == 0:
        raise ValueError(f"No PSDs available for detector {det_name}")
    idx = rng.randint(0, len(entries))
    return entries[idx]


def _build_pycbc_psd(psd_entry: Dict, flen: int, delta_f: float,
                     f_lower: float) -> FrequencySeries:
    """
    Interpolate a real PSD onto the frequency grid required by noise_from_psd.

    Parameters
    ----------
    psd_entry : dict with 'freqs' and 'psd' arrays
    flen : int  —  number of frequency bins (target_length // 2 + 1)
    delta_f : float  —  frequency resolution
    f_lower : float  —  lower frequency cutoff

    Returns
    -------
    PyCBC FrequencySeries suitable for noise_from_psd
    """
    freqs_target = np.arange(flen) * delta_f
    psd_interp = np.interp(freqs_target, psd_entry['freqs'], psd_entry['psd'],
                           left=0.0, right=0.0)

    # Zero below f_lower (no noise power there)
    psd_interp[freqs_target < f_lower] = 0.0

    # Ensure no negative or NaN values
    psd_interp = np.maximum(np.nan_to_num(psd_interp, nan=0.0), 0.0)

    return FrequencySeries(psd_interp, delta_f=delta_f)


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE-WAVEFORM WORKER (REAL PSD NOISE)
# ═══════════════════════════════════════════════════════════════════════════════

# Module-level storage that workers populate via initializer (avoids pickling)
_worker_psds = None
_worker_rng = None


def _init_worker(psds_by_det, seed):
    """Pool initializer — each worker gets a copy of PSDs and its own RNG."""
    global _worker_psds, _worker_rng
    _worker_psds = psds_by_det
    _worker_rng = np.random.RandomState(seed)


def _generate_single_waveform_real_psd(
    params: Dict, time_resolution: float, approximant: str,
    f_lower: float, detectors: List[str], target_length: int,
    add_noise: bool, whiten: bool,
) -> Dict:
    """Worker: generate one waveform, inject noise coloured by a real PSD."""
    global _worker_psds, _worker_rng
    try:
        hp, hc = get_td_waveform(
            approximant=approximant,
            mass1=params['mass1'],
            mass2=params['mass2'],
            spin1z=params.get('spin1z', 0.0),
            spin2z=params.get('spin2z', 0.0),
            inclination=params.get('inclination', 0.0),
            coa_phase=params.get('coa_phase', 0.0),
            distance=params.get('distance', 410.0),
            delta_t=time_resolution,
            f_lower=f_lower,
        )

        gps_time = params.get('gps_time', 1126259462.4)
        hp.start_time += gps_time
        hc.start_time += gps_time

        ra = params.get('ra', 0.0)
        dec = params.get('dec', np.pi / 2)
        polarization = params.get('polarization', 0.0)

        detector_signals = {}
        for det_name in detectors:
            detector = Detector(det_name)
            signal = detector.project_wave(hp, hc, ra, dec, polarization, method='lal')

            signal_len = len(signal)
            if signal_len >= target_length:
                signal = signal[-target_length:]
            else:
                pad_len = target_length - signal_len
                padded = np.zeros(target_length, dtype=signal.dtype)
                if signal_len > 0:
                    padded[pad_len:] = signal.data[:]
                padded_epoch = signal.start_time - pad_len * signal.delta_t
                signal = TimeSeries(padded, delta_t=signal.delta_t, epoch=padded_epoch)

            detector_signals[det_name] = signal

        # ── Inject noise coloured by a real PSD ──
        if add_noise:
            for det_name in detectors:
                signal = detector_signals[det_name]
                delta_t = signal.delta_t
                duration = target_length * delta_t
                delta_f = 1.0 / duration
                flen = target_length // 2 + 1

                psd_entry = _pick_random_psd(_worker_psds, det_name, _worker_rng)
                psd = _build_pycbc_psd(psd_entry, flen, delta_f, f_lower)

                noise = noise_from_psd(target_length, delta_t, psd)
                noise._epoch = signal._epoch
                detector_signals[det_name] = signal.inject(noise)

        # ── Whiten ──
        if whiten:
            for det_name in detectors:
                signal = detector_signals[det_name]
                whitened_data, _, _ = whiten_waveform(
                    signal,
                    delta_t=time_resolution,
                    f_lower=f_lower,
                    apply_bandpass=True,
                    apply_tukey=True,
                    tukey_side='left',
                )
                detector_signals[det_name] = whitened_data

        return {'success': True, 'detectors': detector_signals, 'params': params}

    except Exception as e:
        return {'success': False, 'error': str(e), 'params': params}


# ═══════════════════════════════════════════════════════════════════════════════
# MODIFIED (MASSIVE GRAVITON) WORKER — REAL PSD NOISE
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_single_modified_waveform_real_psd(
    params: Dict, time_resolution: float, approximant: str,
    f_lower: float, detectors: List[str], target_length: int,
    add_noise: bool, whiten: bool, lambda_g: float, f_final: float,
) -> Dict:
    """Worker: generate one FD modified waveform, inject real-PSD noise."""
    global _worker_psds, _worker_rng
    try:
        delta_f = 1.0 / 256

        hp_fd, hc_fd = get_fd_waveform(
            approximant=approximant,
            mass1=params['mass1'],
            mass2=params['mass2'],
            spin1z=params.get('spin1z', 0.0),
            spin2z=params.get('spin2z', 0.0),
            inclination=params.get('inclination', 0.0),
            coa_phase=params.get('coa_phase', 0.0),
            distance=params.get('distance', 410.0),
            delta_f=delta_f,
            f_lower=f_lower,
            f_final=f_final,
        )

        m1, m2 = params['mass1'], params['mass2']
        chirp_mass = (m1 * m2) ** (3.0 / 5.0) / (m1 + m2) ** (1.0 / 5.0)
        z = params.get('redshift', 0.1)

        freqs = hp_fd.sample_frequencies.numpy()[1:]
        lg = params.get('lambda_g', lambda_g)
        hp_fd_amp = np.abs(hp_fd.numpy()[1:])
        f_c = float(np.max(freqs[np.nonzero(hp_fd_amp)]))
        phase_shift = _additional_phase(freqs, chirp_mass, z, lg, f_c)

        hp_array = hp_fd.numpy().copy()
        hc_array = hc_fd.numpy().copy()
        hp_array[1:] *= np.exp(1j * phase_shift)
        hc_array[1:] *= np.exp(1j * phase_shift)

        hp_raw = np.fft.irfft(hp_array)
        hc_raw = np.fft.irfft(hc_array)
        N = len(hp_raw)
        hp_raw *= delta_f * N
        hc_raw *= delta_f * N

        n_ringdown = 500
        if target_length <= N:
            n_pre = target_length - n_ringdown
            hp_arr = np.concatenate([hp_raw[N - n_pre:], hp_raw[:n_ringdown]])
            hc_arr = np.concatenate([hc_raw[N - n_pre:], hc_raw[:n_ringdown]])
        else:
            hp_arr = np.concatenate([np.zeros(target_length - N), hp_raw])
            hc_arr = np.concatenate([np.zeros(target_length - N), hc_raw])

        hp_ts = TimeSeries(hp_arr.astype(np.float64), delta_t=time_resolution)
        hc_ts = TimeSeries(hc_arr.astype(np.float64), delta_t=time_resolution)

        gps_time = params.get('gps_time', 1126259462.4)
        hp_ts.start_time += gps_time
        hc_ts.start_time += gps_time

        ra = params.get('ra', 0.0)
        dec = params.get('dec', np.pi / 2)
        polarization = params.get('polarization', 0.0)

        detector_signals = {}
        for det_name in detectors:
            detector = Detector(det_name)
            signal = detector.project_wave(hp_ts, hc_ts, ra, dec, polarization, method='lal')

            sig_array = np.array(signal)
            signal_len = len(sig_array)
            if signal_len > target_length:
                sig_array = sig_array[signal_len - target_length:]
            elif signal_len < target_length:
                sig_array = np.concatenate([
                    np.zeros(target_length - signal_len, dtype=sig_array.dtype),
                    sig_array,
                ])

            detector_signals[det_name] = TimeSeries(
                sig_array, delta_t=signal.delta_t, epoch=signal.start_time)

        # ── Inject noise coloured by a real PSD ──
        if add_noise:
            for det_name in detectors:
                signal = detector_signals[det_name]
                delta_t = signal.delta_t
                duration = target_length * delta_t
                delta_f_noise = 1.0 / duration
                flen = target_length // 2 + 1

                psd_entry = _pick_random_psd(_worker_psds, det_name, _worker_rng)
                psd = _build_pycbc_psd(psd_entry, flen, delta_f_noise, f_lower)

                noise = noise_from_psd(target_length, delta_t, psd)
                noise._epoch = signal._epoch
                detector_signals[det_name] = signal.inject(noise)

        if whiten:
            for det_name in detectors:
                signal = detector_signals[det_name]
                whitened_data, _, _ = whiten_waveform(
                    signal,
                    delta_t=time_resolution,
                    f_lower=f_lower,
                    apply_bandpass=True,
                    apply_tukey=True,
                    tukey_side='left',
                )
                detector_signals[det_name] = whitened_data

        return {'success': True, 'detectors': detector_signals, 'params': params}

    except Exception as e:
        return {'success': False, 'error': str(e), 'params': params}


def _generate_waveforms_parallel_real_psd(
    param_dicts, time_resolution, approximant, f_lower, num_workers,
    show_progress, detectors, target_length, add_noise, whiten,
    psds_by_det,
):
    worker_func = partial(
        _generate_single_waveform_real_psd,
        time_resolution=time_resolution,
        approximant=approximant,
        f_lower=f_lower,
        detectors=detectors,
        target_length=target_length,
        add_noise=add_noise,
        whiten=whiten,
    )

    seed = np.random.randint(0, 2**31)
    with Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(psds_by_det, seed),
    ) as pool:
        if show_progress:
            results = list(tqdm(
                pool.imap_unordered(worker_func, param_dicts, chunksize=100),
                total=len(param_dicts),
                desc="Generating waveforms (real PSD noise)",
            ))
        else:
            results = list(pool.imap_unordered(worker_func, param_dicts, chunksize=100))

    return results


def _generate_modified_waveforms_parallel_real_psd(
    param_dicts, time_resolution, approximant, f_lower, num_workers,
    show_progress, detectors, target_length, add_noise, whiten,
    psds_by_det, lambda_g, f_final,
):
    worker_func = partial(
        _generate_single_modified_waveform_real_psd,
        time_resolution=time_resolution,
        approximant=approximant,
        f_lower=f_lower,
        detectors=detectors,
        target_length=target_length,
        add_noise=add_noise,
        whiten=whiten,
        lambda_g=lambda_g,
        f_final=f_final,
    )

    seed = np.random.randint(0, 2**31)
    with Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(psds_by_det, seed),
    ) as pool:
        if show_progress:
            results = list(tqdm(
                pool.imap_unordered(worker_func, param_dicts, chunksize=100),
                total=len(param_dicts),
                desc="Generating modified waveforms (real PSD noise)",
            ))
        else:
            results = list(pool.imap_unordered(worker_func, param_dicts, chunksize=100))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API — TD GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def pycbc_data_generator_real_psd(
    config: Dict[str, Callable],
    num_samples: int,
    psd_csv: str,
    time_resolution: float = 1 / 4096,
    approximant: str = 'IMRPhenomXP',
    f_lower: float = 40.0,
    num_workers: int = None,
    signal_length: float = 2.0,
    batch_size: int = 256,
    chunk_size: int = 10000,
    train_split: float = 0.8,
    val_split: float = 0.1,
    show_progress: bool = True,
    detectors: List[str] = None,
    add_noise: bool = True,
    whiten: bool = False,
) -> Dict:
    """
    Generate PyCBC waveforms with noise coloured by real detector PSDs.

    Identical interface to ``pycbc_data_generator`` in DataGenerator.py, except
    noise is generated from real O4 PSDs loaded from *psd_csv* instead of the
    analytical aLIGOZeroDetHighPower model.

    Parameters
    ----------
    psd_csv : str or Path
        Path to PSD CSV file (produced by RealDataGenerator*.py).
    (all other parameters are the same as pycbc_data_generator)

    Returns
    -------
    dict with 'train_loader', 'val_loader', 'test_loader', 'metadata'
    """
    _validate_config(config)
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if not 0 < train_split < 1 or not 0 < val_split < 1:
        raise ValueError("train_split and val_split must be between 0 and 1")
    if train_split + val_split >= 1:
        raise ValueError("train_split + val_split must be < 1")

    if num_workers is None:
        num_workers = 1
    if detectors is None:
        detectors = ['H1', 'L1']

    # ── Load real PSDs ──
    print(f"Loading real PSDs from {psd_csv} ...")
    psds_by_det = load_real_psds(psd_csv)
    for det in detectors:
        if det not in psds_by_det:
            raise ValueError(
                f"Detector {det} not found in PSD CSV. "
                f"Available: {list(psds_by_det.keys())}"
            )

    target_length = int(signal_length / time_resolution)
    overall_start_time = perf_counter()
    print(f"Generating {num_samples} waveforms with REAL PSD noise")
    print(f"  Detectors: {detectors}")
    print(f"  Target signal length: {target_length} samples ({signal_length}s)")
    print(f"  Noise injection: {'enabled (real PSD)' if add_noise else 'disabled'}")
    print(f"  Whitening: {'enabled' if whiten else 'disabled'}")

    param_dicts = _generate_parameter_sets(config, num_samples)

    all_successful = []
    all_failed = []
    num_chunks = (num_samples + chunk_size - 1) // chunk_size

    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, num_samples)
        chunk_params = param_dicts[chunk_start:chunk_end]
        chunk_start_time = perf_counter()

        if num_chunks > 1:
            print(f"\nChunk {chunk_idx + 1}/{num_chunks} ({len(chunk_params)} waveforms)...")

        chunk_results = _generate_waveforms_parallel_real_psd(
            chunk_params, time_resolution, approximant, f_lower,
            num_workers, show_progress, detectors, target_length,
            add_noise, whiten, psds_by_det,
        )

        for r in chunk_results:
            (all_successful if r['success'] else all_failed).append(r)

        chunk_successes = sum(1 for r in chunk_results if r['success'])
        chunk_elapsed = perf_counter() - chunk_start_time
        total_elapsed = perf_counter() - overall_start_time

        print(f"  Chunk: {chunk_successes} successful")
        print(f"  Chunk generation time: {_format_elapsed_time(chunk_elapsed)}")
        print(f"  Total runtime so far: {_format_elapsed_time(total_elapsed)}")

    if not all_successful:
        print(f"\nALL {len(all_failed)} waveforms failed!")
        for i, fail in enumerate(all_failed[:5]):
            print(f"  Error #{i+1}: {fail.get('error', 'unknown')}")
        raise RuntimeError("No waveforms were successfully generated!")

    num_success = len(all_successful)
    num_failed = len(all_failed)
    print(f"\nGeneration complete: {num_success} successful, {num_failed} failed")
    print(f"Total waveform generation runtime: {_format_elapsed_time(perf_counter() - overall_start_time)}")

    param_names = list(all_successful[0]['params'].keys())
    num_params = len(param_names)
    detector_names = list(all_successful[0]['detectors'].keys())
    num_detectors = len(detector_names)

    print(f"  Detector channels: {detector_names}")

    signal_array = np.empty((num_success, num_detectors, target_length), dtype=np.float32)
    param_array = np.empty((num_success, num_params), dtype=np.float32)

    for i, waveform_data in enumerate(all_successful):
        for j, param_name in enumerate(param_names):
            param_array[i, j] = waveform_data['params'][param_name]
        for k, det_name in enumerate(detector_names):
            signal_array[i, k, :] = waveform_data['detectors'][det_name]

    param_array, param_norm_info = normalize_parameters(
        param_array, param_names, method='zscore'
    )

    X = torch.from_numpy(signal_array)
    y = torch.from_numpy(param_array)
    print(f"  Tensors: X={X.shape}, y={y.shape}")

    dataset = TensorDataset(X, y)
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size

    train_data, val_data, test_data = random_split(
        dataset, [train_size, val_size, test_size]
    )
    print(f"  Splits: train={train_size}, val={val_size}, test={test_size}")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    print(f"\nReady! DataLoaders with batch_size={batch_size}")
    print(f"Total generator runtime: {_format_elapsed_time(perf_counter() - overall_start_time)}")

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'metadata': {
            'parameter_names': param_names,
            'parameter_normalization': param_norm_info,
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
            'signal_length': signal_length,
            'chunk_size': chunk_size,
            'add_noise': add_noise,
            'whiten': whiten,
            'noise_source': 'real_psd',
            'psd_csv': str(psd_csv),
            'preprocessing': {},
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API — FD MODIFIED (MASSIVE GRAVITON) GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def pycbc_modified_data_generator_real_psd(
    config: Dict[str, Callable],
    num_samples: int,
    psd_csv: str,
    lambda_g: float = None,
    time_resolution: float = 1 / 4096,
    approximant: str = 'IMRPhenomD',
    f_lower: float = 30.0,
    f_final: float = 2048.0,
    num_workers: int = None,
    signal_length: float = 2.0,
    batch_size: int = 256,
    chunk_size: int = 10000,
    train_split: float = 0.8,
    val_split: float = 0.1,
    show_progress: bool = True,
    detectors: List[str] = None,
    add_noise: bool = True,
    whiten: bool = True,
) -> Dict:
    """
    Generate modified (massive graviton) waveforms with real-PSD noise.

    Identical to ``pycbc_modified_data_generator`` but uses real detector PSDs.

    Parameters
    ----------
    psd_csv : str or Path
        Path to PSD CSV produced by the RealDataGenerator scripts.
    lambda_g : float or None
        Graviton Compton wavelength in metres. If None, must be in config.
    (all other parameters identical to pycbc_modified_data_generator)
    """
    _validate_config(config)
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if not 0 < train_split < 1 or not 0 < val_split < 1:
        raise ValueError("train_split and val_split must be between 0 and 1")
    if train_split + val_split >= 1:
        raise ValueError("train_split + val_split must be < 1")
    lambda_g_in_config = 'lambda_g' in config
    if lambda_g is None and not lambda_g_in_config:
        raise ValueError("lambda_g must be provided either as an argument or in config")
    if lambda_g is not None and lambda_g <= 0:
        raise ValueError("lambda_g must be positive")

    if num_workers is None:
        num_workers = 1
    if detectors is None:
        detectors = ['H1', 'L1']

    # ── Load real PSDs ──
    print(f"Loading real PSDs from {psd_csv} ...")
    psds_by_det = load_real_psds(psd_csv)
    for det in detectors:
        if det not in psds_by_det:
            raise ValueError(
                f"Detector {det} not found in PSD CSV. "
                f"Available: {list(psds_by_det.keys())}"
            )

    target_length = int(signal_length / time_resolution)
    overall_start_time = perf_counter()
    if lambda_g_in_config:
        print(f"Generating {num_samples} MODIFIED waveforms (lambda_g=per-sample, real PSD noise)")
    else:
        print(f"Generating {num_samples} MODIFIED waveforms (lambda_g={lambda_g:.2e} m, real PSD noise)")
    print(f"  Approximant: {approximant} (frequency domain)")
    print(f"  Frequency range: {f_lower}-{f_final} Hz")
    print(f"  Detectors: {detectors}")
    print(f"  Target signal length: {target_length} samples ({signal_length}s)")
    print(f"  Noise: real PSD from {psd_csv}")

    param_dicts = _generate_parameter_sets(config, num_samples)

    all_successful = []
    all_failed = []
    num_chunks = (num_samples + chunk_size - 1) // chunk_size

    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, num_samples)
        chunk_params = param_dicts[chunk_start:chunk_end]
        chunk_start_time = perf_counter()

        if num_chunks > 1:
            print(f"\nChunk {chunk_idx + 1}/{num_chunks} ({len(chunk_params)} waveforms)...")

        chunk_results = _generate_modified_waveforms_parallel_real_psd(
            chunk_params, time_resolution, approximant, f_lower,
            num_workers, show_progress, detectors, target_length,
            add_noise, whiten, psds_by_det, lambda_g, f_final,
        )

        for r in chunk_results:
            (all_successful if r['success'] else all_failed).append(r)

        chunk_successes = sum(1 for r in chunk_results if r['success'])
        chunk_elapsed = perf_counter() - chunk_start_time
        total_elapsed = perf_counter() - overall_start_time

        print(f"  Chunk: {chunk_successes} successful")
        print(f"  Chunk generation time: {_format_elapsed_time(chunk_elapsed)}")
        print(f"  Total runtime so far: {_format_elapsed_time(total_elapsed)}")

    if not all_successful:
        raise RuntimeError("No waveforms were successfully generated!")

    num_success = len(all_successful)
    num_failed = len(all_failed)
    print(f"\nGeneration complete: {num_success} successful, {num_failed} failed")
    print(f"Total waveform generation runtime: {_format_elapsed_time(perf_counter() - overall_start_time)}")

    param_names = list(all_successful[0]['params'].keys())
    num_params = len(param_names)
    detector_names = list(all_successful[0]['detectors'].keys())
    num_detectors = len(detector_names)

    signal_array = np.empty((num_success, num_detectors, target_length), dtype=np.float32)
    param_array = np.empty((num_success, num_params), dtype=np.float32)

    for i, waveform_data in enumerate(all_successful):
        for j, param_name in enumerate(param_names):
            param_array[i, j] = waveform_data['params'][param_name]
        for k, det_name in enumerate(detector_names):
            signal_array[i, k, :] = waveform_data['detectors'][det_name]

    param_array, param_norm_info = normalize_parameters(
        param_array, param_names, method='zscore'
    )

    X = torch.from_numpy(signal_array)
    y = torch.from_numpy(param_array)
    print(f"  Tensors: X={X.shape}, y={y.shape}")

    dataset = TensorDataset(X, y)
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size

    train_data, val_data, test_data = random_split(
        dataset, [train_size, val_size, test_size]
    )
    print(f"  Splits: train={train_size}, val={val_size}, test={test_size}")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    print(f"\nReady! DataLoaders with batch_size={batch_size}")
    print(f"Total generator runtime: {_format_elapsed_time(perf_counter() - overall_start_time)}")

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'metadata': {
            'parameter_names': param_names,
            'parameter_normalization': param_norm_info,
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
            'f_final': f_final,
            'detectors': detector_names,
            'target_length': target_length,
            'signal_length': signal_length,
            'chunk_size': chunk_size,
            'add_noise': add_noise,
            'whiten': whiten,
            'lambda_g': lambda_g,
            'lambda_g_varied': lambda_g_in_config,
            'modified': True,
            'noise_source': 'real_psd',
            'psd_csv': str(psd_csv),
            'preprocessing': {},
        },
    }
