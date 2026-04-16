# generate_data.py
# Script 1 of 3: Generate and save the waveform dataset to disk.
# Run this script first, then train_model.py, then plot_results.py.
#
# Output: prepared_data.pt containing train/val/test tensors and normalisation info.

import torch
import numpy as np
from scipy import stats
import math
import time
from multiprocessing import Pool
import gc
import psutil
import os
import json
import hashlib
import DataGenerator as data_generator
import DataGeneratorRealPSD as data_generator_real_psd

print("Libraries imported successfully")
print(f"PyTorch version: {torch.__version__}")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

pi = np.pi

WAVEFORM_CACHE_VERSION = 1

import DataPipeline
from DataPipeline import generate_dataset, save_dataset, load_dataset, build_dataset_path


# ==============================================================================
# MEMORY UTILITIES
# ==============================================================================

def get_memory_usage():
    """Get current memory usage for CPU and GPU."""
    process = psutil.Process(os.getpid())
    cpu_mem_gb = process.memory_info().rss / (1024**3)
    gpu_mem_gb = 0
    if torch.cuda.is_available():
        gpu_mem_gb = torch.cuda.memory_allocated() / (1024**3)
    return cpu_mem_gb, gpu_mem_gb


def print_memory_status():
    """Print current memory usage."""
    cpu_mem, gpu_mem = get_memory_usage()
    print(f"\n{'='*60}")
    print("MEMORY STATUS:")
    print(f"  CPU Memory: {cpu_mem:.2f} GB")
    if torch.cuda.is_available():
        print(f"  GPU Memory: {gpu_mem:.2f} GB")
    print(f"{'='*60}\n")


def clear_memory(verbose=True):
    """Clear Python memory cache, garbage collection, and PyTorch cache."""
    if verbose:
        print("\nClearing memory...")
        cpu_before, gpu_before = get_memory_usage()
        print(f"  Before: CPU={cpu_before:.2f}GB", end="")
        if torch.cuda.is_available():
            print(f", GPU={gpu_before:.2f}GB", end="")
        print()

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if verbose:
        cpu_after, gpu_after = get_memory_usage()
        print(f"  After:  CPU={cpu_after:.2f}GB", end="")
        if torch.cuda.is_available():
            print(f", GPU={gpu_after:.2f}GB", end="")
        print()
        cpu_freed = cpu_before - cpu_after
        print(f"  Freed: {cpu_freed:.2f} GB")
        if torch.cuda.is_available():
            gpu_freed = gpu_before - gpu_after
            print(f"  GPU Freed: {gpu_freed:.2f} GB")


# ==============================================================================
# WAVEFORM CACHE FUNCTIONS
# ==============================================================================

def _make_cache_key(num_samples, generator_backend=None, psd_csv=None):
    """Generate a deterministic hash key from the current data-generation config."""
    key_dict = {
        'version': WAVEFORM_CACHE_VERSION,
        'num_samples': num_samples,
        'generator_backend': generator_backend or GENERATOR_BACKEND,
        'psd_csv': psd_csv or REAL_PSD_CSV,
        'modified_gravity': MODIFIED_GRAVITY,
        'lambda_g_min': LAMBDA_G_MIN if MODIFIED_GRAVITY else None,
        'lambda_g_max': LAMBDA_G_MAX if MODIFIED_GRAVITY else None,
        'add_noise': ADD_NOISE,
        'whiten': WHITEN,
        'model_params': sorted(MODEL_PARAMS),
        'param_set': PARAM_SET,
    }
    key_str = json.dumps(key_dict, sort_keys=True)
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]


def load_prepared_waveform_cache(num_samples, generator_backend=None, psd_csv=None):
    """
    Load a previously saved prepared waveform cache if it exists and matches
    the current configuration. Returns None if caching is disabled or no match.
    """
    if not USE_WAVEFORM_CACHE:
        return None

    cache_key = _make_cache_key(num_samples, generator_backend, psd_csv)
    cache_path = f"waveform_cache_{cache_key}.pt"

    if os.path.exists(cache_path):
        print(f"  Loading waveform cache: {cache_path}")
        try:
            cached = torch.load(cache_path, map_location='cpu')
            outputs = (
                cached['train_data'],
                cached['train_params'],
                cached['val_data'],
                cached['val_params'],
                cached['test_data'],
                cached['test_params'],
                cached['param_norm_info'],
                cached['param_names'],
                cached['gps_time_delay'],
            )
            print(f"  ✓ Loaded from cache (key={cache_key})")
            return outputs
        except Exception as e:
            print(f"  ⚠ Failed to load cache: {e}. Regenerating...")
            return None

    print(f"  No matching waveform cache found (key={cache_key}). Generating fresh data.")
    return None


def save_prepared_waveform_cache(num_samples, prepared_outputs, generator_metadata,
                                  component_param_names, generator_backend=None, psd_csv=None):
    """Save prepared waveform outputs to a cache file keyed by the current config."""
    if not USE_WAVEFORM_CACHE:
        return

    cache_key = _make_cache_key(num_samples, generator_backend, psd_csv)
    cache_path = f"waveform_cache_{cache_key}.pt"

    (all_data, all_params, all_val_data, all_val_params,
     all_test_data, all_test_params, param_norm_info, param_names, gps_time_delay) = prepared_outputs

    try:
        torch.save({
            'train_data': all_data,
            'train_params': all_params,
            'val_data': all_val_data,
            'val_params': all_val_params,
            'test_data': all_test_data,
            'test_params': all_test_params,
            'param_norm_info': param_norm_info,
            'param_names': param_names,
            'gps_time_delay': gps_time_delay,
            'generator_metadata': generator_metadata,
            'component_param_names': component_param_names,
            'cache_key': cache_key,
        }, cache_path)
        print(f"  ✓ Waveform cache saved: {cache_path}")
    except Exception as e:
        print(f"  ⚠ Failed to save waveform cache: {e}")


# ==============================================================================
# PHYSICS & PARAMETER UTILITIES
# ==============================================================================

def denormalize_params(normalized_params, param_norm_info, param_names):
    """Convert normalized parameters back to physical space."""
    if isinstance(normalized_params, torch.Tensor):
        physical_params = normalized_params.clone()
    else:
        physical_params = np.array(normalized_params, copy=True)

    for j, param_name in enumerate(param_names):
        if param_name in param_norm_info:
            info = param_norm_info[param_name]
            original_mean = info['mean']
            original_std = info['std']
            if isinstance(physical_params, torch.Tensor):
                physical_params[..., j] = physical_params[..., j] * original_std + original_mean
            else:
                physical_params[..., j] = physical_params[..., j] * original_std + original_mean

    return physical_params


def normalize_params_with_reference(param_array, param_names, reference_norm_info=None):
    """
    Z-score normalize parameter matrix.
    If reference_norm_info is provided, use those fixed stats (for test/inference).
    Otherwise compute stats from the provided array (for training set only).
    """
    arr = np.array(param_array, dtype=np.float32, copy=True)
    normed = np.zeros_like(arr, dtype=np.float32)

    norm_info = {}
    for j, name in enumerate(param_names):
        if reference_norm_info is None:
            mean = float(arr[:, j].mean())
            std = float(arr[:, j].std())
            if std <= 0:
                std = 1.0
        else:
            mean = float(reference_norm_info[name]['mean'])
            std = float(reference_norm_info[name]['std'])
            if std <= 0:
                std = 1.0

        normed[:, j] = (arr[:, j] - mean) / std
        norm_info[name] = {
            'mean': mean,
            'std': std,
            'min': float(arr[:, j].min()),
            'max': float(arr[:, j].max()),
            'method': 'zscore'
        }

    return normed, norm_info


def estimate_redshift_from_luminosity_distance(distance_mpc, h0_km_s_mpc=67.74):
    """Approximate redshift from luminosity distance using low-z Hubble law: z ≈ H0 * D_L / c."""
    if distance_mpc is None or distance_mpc <= 0:
        raise ValueError("distance_mpc must be positive to estimate redshift")
    c_km_s = 299792.458
    z = (h0_km_s_mpc * float(distance_mpc)) / c_km_s
    return float(max(z, 0.0))


def convert_source_to_detector_frame_masses(mass1_source, mass2_source, redshift=None,
                                            distance_mpc=None, h0_km_s_mpc=67.74):
    """
    Convert source-frame component masses to detector-frame masses.
    Uses m_det = m_src * (1 + z). If redshift is not provided, estimates z from
    luminosity distance via low-z Hubble law.
    """
    if redshift is None:
        redshift = estimate_redshift_from_luminosity_distance(distance_mpc, h0_km_s_mpc=h0_km_s_mpc)
    if redshift < 0:
        raise ValueError(f"redshift must be non-negative, got {redshift}")

    scale = 1.0 + float(redshift)
    mass1_detector = float(mass1_source) * scale
    mass2_detector = float(mass2_source) * scale
    return mass1_detector, mass2_detector, float(redshift)


# Extra (non-mass/spin) parameters passed through unchanged during reparameterization.
# Overridden in __main__ based on PARAM_SET config.
EXTRA_PARAM_NAMES = []


def transform_component_params_to_reparameterized(param_array, param_names):
    """
    Convert component parameters to symmetric target basis:
    [mass1, mass2, spin1z, spin2z, ...extra...] -> [chirp_mass, q, chi_eff, chi_a, ...extra...].
    Extra parameters (coa_phase, distance, etc.) are passed through unchanged.
    """
    required = ['mass1', 'mass2', 'spin1z', 'spin2z']
    missing = [p for p in required if p not in param_names]
    if missing:
        raise ValueError(f"Cannot reparameterize: missing parameters {missing}")

    i_m1 = param_names.index('mass1')
    i_m2 = param_names.index('mass2')
    i_s1 = param_names.index('spin1z')
    i_s2 = param_names.index('spin2z')

    arr = np.array(param_array, dtype=np.float32, copy=True)
    m1_raw = arr[:, i_m1]
    m2_raw = arr[:, i_m2]
    s1_raw = arr[:, i_s1]
    s2_raw = arr[:, i_s2]

    heavy_is_1 = m1_raw >= m2_raw
    m1 = np.where(heavy_is_1, m1_raw, m2_raw)
    m2 = np.where(heavy_is_1, m2_raw, m1_raw)
    s1 = np.where(heavy_is_1, s1_raw, s2_raw)
    s2 = np.where(heavy_is_1, s2_raw, s1_raw)

    total_mass = np.clip(m1 + m2, 1e-8, None)
    chirp_mass = np.power(np.clip(m1 * m2, 1e-12, None), 3.0 / 5.0) / np.power(total_mass, 1.0 / 5.0)
    q = np.clip(m2 / np.clip(m1, 1e-8, None), 1e-4, 1.0)
    chi_eff = (m1 * s1 + m2 * s2) / total_mass
    chi_a = 0.5 * (s1 - s2)

    columns = [chirp_mass, q, chi_eff, chi_a]
    transformed_names = ['chirp_mass', 'q', 'chi_eff', 'chi_a']

    param_names_list = list(param_names)
    for ep in EXTRA_PARAM_NAMES:
        if ep in param_names_list:
            columns.append(arr[:, param_names_list.index(ep)])
            transformed_names.append(ep)

    transformed = np.stack(columns, axis=1).astype(np.float32)
    return transformed, transformed_names


def transform_component_dict_to_reparameterized(param_dict):
    """Convert a single component-parameter dict to reparameterized dict."""
    required = ['mass1', 'mass2', 'spin1z', 'spin2z']
    arr = np.array([[param_dict[k] for k in required]], dtype=np.float32)
    transformed, names = transform_component_params_to_reparameterized(arr, required)
    result = {n: float(transformed[0, i]) for i, n in enumerate(names)}
    for ep in EXTRA_PARAM_NAMES:
        if ep in param_dict:
            result[ep] = float(param_dict[ep])
    return result


def transform_reparameterized_to_component_samples(param_array, param_names):
    """
    Convert [chirp_mass, q, chi_eff, chi_a, ...extra...] samples back to
    [mass1, mass2, spin1z, spin2z, ...extra...] where mass1 >= mass2.
    """
    required = ['chirp_mass', 'q', 'chi_eff', 'chi_a']
    missing = [p for p in required if p not in param_names]
    if missing:
        raise ValueError(f"Cannot invert reparameterization: missing parameters {missing}")

    idx_mc = param_names.index('chirp_mass')
    idx_q = param_names.index('q')
    idx_ce = param_names.index('chi_eff')
    idx_ca = param_names.index('chi_a')

    arr = np.array(param_array, dtype=np.float32, copy=True)
    mc = np.clip(arr[:, idx_mc], 1e-6, None)
    q = np.clip(arr[:, idx_q], 1e-4, 1.0)
    chi_eff = arr[:, idx_ce]
    chi_a = arr[:, idx_ca]

    total_mass = mc * np.power(1.0 + q, 6.0 / 5.0) / np.power(q, 3.0 / 5.0)
    m1 = total_mass / (1.0 + q)
    m2 = q * m1

    s1 = chi_eff + 2.0 * (m2 / np.clip(total_mass, 1e-8, None)) * chi_a
    s2 = chi_eff - 2.0 * (m1 / np.clip(total_mass, 1e-8, None)) * chi_a

    s1 = np.clip(s1, -0.999, 0.999)
    s2 = np.clip(s2, -0.999, 0.999)

    columns = [m1, m2, s1, s2]
    comp_names = ['mass1', 'mass2', 'spin1z', 'spin2z']

    param_names_list = list(param_names)
    for ep in EXTRA_PARAM_NAMES:
        if ep in param_names_list:
            columns.append(arr[:, param_names_list.index(ep)])
            comp_names.append(ep)

    comp = np.stack(columns, axis=1).astype(np.float32)
    return comp, comp_names


def convert_samples_to_eval_space(samples, source_param_names, output_component_distributions=False):
    """Convert parameter samples to chosen output space."""
    if output_component_distributions:
        converted, eval_names = transform_reparameterized_to_component_samples(samples, source_param_names)
        return converted, eval_names
    return samples, list(source_param_names)


def convert_vector_to_eval_space(param_vector, source_param_names, output_component_distributions=False):
    """Convert a single parameter vector to chosen output space."""
    vector_2d = np.array(param_vector, dtype=np.float32, copy=True).reshape(1, -1)
    converted, eval_names = convert_samples_to_eval_space(
        vector_2d, source_param_names,
        output_component_distributions=output_component_distributions
    )
    return converted[0], eval_names


def split_samples_into_symmetric_and_component(samples, source_param_names):
    """
    Return both symmetric and component representations for posterior samples.
    Extra parameters (coa_phase, distance, lambda_g) are included in both outputs.
    """
    name_set = set(source_param_names)
    sym_required = {'chirp_mass', 'q', 'chi_eff', 'chi_a'}
    comp_required = {'mass1', 'mass2', 'spin1z', 'spin2z'}

    if sym_required.issubset(name_set):
        symmetric = np.array(samples, dtype=np.float32, copy=True)
        symmetric_names = list(source_param_names)
        component, component_names = transform_reparameterized_to_component_samples(symmetric, source_param_names)
        return symmetric, symmetric_names, component, component_names

    if comp_required.issubset(name_set):
        component = np.array(samples, dtype=np.float32, copy=True)
        component_names = list(source_param_names)
        symmetric, symmetric_names = transform_component_params_to_reparameterized(component, source_param_names)
        return symmetric, symmetric_names, component, component_names

    arr = np.array(samples, dtype=np.float32, copy=True)
    names = list(source_param_names)
    return arr, names, arr, names


def split_vector_into_symmetric_and_component(param_vector, source_param_names):
    """Return both symmetric and component representations for one parameter vector."""
    vector_2d = np.array(param_vector, dtype=np.float32, copy=True).reshape(1, -1)
    symmetric, symmetric_names, component, component_names = split_samples_into_symmetric_and_component(
        vector_2d, source_param_names
    )
    return symmetric[0], symmetric_names, component[0], component_names


def calculate_detector_time_delay(ra: float, dec: float, det1: str = 'H1', det2: str = 'L1',
                                   gps_time: float = 0.0) -> float:
    """
    Calculate the GPS time delay between two detectors for a GW arriving from (ra, dec).
    Uses PyCBC's built-in time_delay_from_earth_center method.
    Returns time delay in seconds (det2 relative to det1).
    """
    try:
        from pycbc.detector import Detector
        detector1 = Detector(det1)
        detector2 = Detector(det2)
        delay1 = detector1.time_delay_from_earth_center(ra, dec, gps_time)
        delay2 = detector2.time_delay_from_earth_center(ra, dec, gps_time)
        return float(delay2 - delay1)
    except Exception as e:
        print(f"Warning: Failed to calculate time delay: {e}. Returning 0.0")
        return 0.0


# ==============================================================================
# DATA PREPARATION FUNCTION
# ==============================================================================

def prepare_pycbc_data(num_samples=10000, generator_backend=None, psd_csv=None):
    cached_outputs = load_prepared_waveform_cache(num_samples, generator_backend=generator_backend, psd_csv=psd_csv)
    if cached_outputs is not None:
        return cached_outputs

    config = {
        'mass1': lambda size: np.random.uniform(10, 180, size=size),
        'mass2': lambda size: np.random.uniform(10, 150, size=size),
        'spin1z': lambda size: np.random.uniform(0, 0.88, size=size),
        'spin2z': lambda size: np.random.uniform(0, 0.88, size=size),
        'coa_phase': lambda size: np.random.uniform(0, 2 * np.pi, size=size),
        'distance': lambda size: np.random.uniform(100, 2000, size=size),
    }

    if MODIFIED_GRAVITY:
        config['lambda_g'] = lambda size: np.random.uniform(LAMBDA_G_MIN, LAMBDA_G_MAX, size=size)

    default_ra = 0.0
    default_dec = np.pi / 2.0
    time_delay_default = calculate_detector_time_delay(default_ra, default_dec, 'H1', 'L1')

    print(f"\nCalling pycbc_data_generator with {num_samples} samples...")
    print(f"  GPS time delay (north pole): {time_delay_default*1000:.3f} ms")
    print(f"  Generator backend: {generator_backend or GENERATOR_BACKEND}")
    try:
        t0 = time.time()

        gen_backend = generator_backend or GENERATOR_BACKEND
        gen_psd_csv = psd_csv if psd_csv is not None else REAL_PSD_CSV

        if gen_backend == 'real_psd':
            if not gen_psd_csv:
                raise ValueError("psd_csv must be provided when generator_backend='real_psd'")
            print(f"  PSD CSV: {gen_psd_csv}")
            if MODIFIED_GRAVITY:
                result = data_generator_real_psd.pycbc_modified_data_generator_real_psd(
                    config, psd_csv=gen_psd_csv, num_samples=num_samples, batch_size=16,
                    num_workers=NUM_WORKERS, chunk_size=5000, add_noise=ADD_NOISE, whiten=WHITEN,
                )
            else:
                result = data_generator_real_psd.pycbc_data_generator_real_psd(
                    config, psd_csv=gen_psd_csv, num_samples=num_samples, batch_size=16,
                    num_workers=NUM_WORKERS, chunk_size=5000, add_noise=ADD_NOISE, whiten=WHITEN,
                )
        else:  # 'standard' backend
            if MODIFIED_GRAVITY:
                result = data_generator.pycbc_modified_data_generator(
                    config, num_samples=num_samples, batch_size=16, num_workers=NUM_WORKERS,
                    chunk_size=5000, add_noise=ADD_NOISE,
                )
                print("  MG mode: lambda_g sampled and passed to FD generator.")
            else:
                result = data_generator.pycbc_data_generator(
                    config, num_samples=num_samples, batch_size=16, num_workers=NUM_WORKERS,
                    chunk_size=5000, add_noise=ADD_NOISE, whiten=WHITEN,
                )
        t1 = time.time()
        print(f"✓ Waveform generation took {t1-t0:.2f} seconds.")
        print("✓ Data generator completed successfully")
    except Exception as e:
        print(f"❌ Error in data_generator: {e}")
        import traceback
        traceback.print_exc()
        raise

    print("Accessing dataloaders and metadata from result...")
    try:
        train_loader = result['train_loader']
        val_loader = result['val_loader']
        test_loader = result['test_loader']
        metadata = result['metadata']
        component_param_names = metadata['parameter_names']
        param_norm_info = metadata['parameter_normalization']
        print(f"✓ Got dataloaders: train={len(train_loader)} batches, val={len(val_loader)} batches, test={len(test_loader)} batches")
        print(f"✓ Parameter normalization info stored")
        print(f"✓ Component parameter names: {component_param_names}")
    except Exception as e:
        print(f"❌ Error accessing dataloaders/metadata: {e}")
        print(f"Result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
        raise

    print("\nProcessing training data from dataloaders...")
    print("  (TWO-DETECTOR MODE: Concatenating H1 and L1 streams)")
    data = []
    params = []
    for i, (waveforms, batch_params) in enumerate(train_loader):
        if i % 1000 == 0:
            print(f"  Processing batch {i}/{len(train_loader)}")
        train_params = torch.FloatTensor(batch_params)
        train_data = torch.FloatTensor(waveforms)
        h1_data = train_data[:, 0, :].reshape(train_data.shape[0], -1)
        l1_data = train_data[:, 1, :].reshape(train_data.shape[0], -1)
        concatenated_data = torch.cat([h1_data, l1_data], dim=1)
        data.append(concatenated_data)
        params.append(train_params)
    all_data = torch.cat(data, dim=0)
    all_params = torch.cat(params, dim=0)

    print("  Normalizing waveform data (per-sample zero mean, unit variance)...")
    data_mean = all_data.mean(dim=1, keepdim=True)
    data_std = all_data.std(dim=1, keepdim=True).clamp(min=1e-8)
    all_data = (all_data - data_mean) / data_std

    nan_mask = torch.isnan(all_data).any(dim=1) | torch.isinf(all_data).any(dim=1)
    if nan_mask.any():
        n_bad = nan_mask.sum().item()
        print(f"  ⚠ Removing {n_bad} samples with NaN/Inf values")
        all_data = all_data[~nan_mask]
        all_params = all_params[~nan_mask]

    print(f"  Training data: {all_data.shape}, params: {all_params.shape}")
    print(f"  Data range: [{all_data.min():.2f}, {all_data.max():.2f}], mean={all_data.mean():.4f}, std={all_data.std():.4f}")

    print("\nProcessing validation data from dataloaders...")
    print("  (TWO-DETECTOR MODE: Concatenating H1 and L1 streams)")
    data_val = []
    params_val = []
    for i, (waveforms, batch_params) in enumerate(val_loader):
        val_params_batch = torch.FloatTensor(batch_params)
        val_data_batch = torch.FloatTensor(waveforms)
        h1_data = val_data_batch[:, 0, :].reshape(val_data_batch.shape[0], -1)
        l1_data = val_data_batch[:, 1, :].reshape(val_data_batch.shape[0], -1)
        concatenated_data = torch.cat([h1_data, l1_data], dim=1)
        data_val.append(concatenated_data)
        params_val.append(val_params_batch)
    all_val_data = torch.cat(data_val, dim=0)
    all_val_params = torch.cat(params_val, dim=0)

    print("  Normalizing validation waveform data (per-sample)...")
    val_data_mean = all_val_data.mean(dim=1, keepdim=True)
    val_data_std = all_val_data.std(dim=1, keepdim=True).clamp(min=1e-8)
    all_val_data = (all_val_data - val_data_mean) / val_data_std

    nan_mask_val = torch.isnan(all_val_data).any(dim=1) | torch.isinf(all_val_data).any(dim=1)
    if nan_mask_val.any():
        n_bad = nan_mask_val.sum().item()
        print(f"  ⚠ Removing {n_bad} val samples with NaN/Inf values")
        all_val_data = all_val_data[~nan_mask_val]
        all_val_params = all_val_params[~nan_mask_val]

    print("\nProcessing test data from dataloaders...")
    print("  (TWO-DETECTOR MODE: Concatenating H1 and L1 streams)")
    data_test = []
    params_test = []
    for i, (waveforms, batch_params) in enumerate(test_loader):
        if i % 100 == 0:
            print(f"  Processing test batch {i}/{len(test_loader)}")
        test_params = torch.FloatTensor(batch_params)
        test_data = torch.FloatTensor(waveforms)
        h1_data = test_data[:, 0, :].reshape(test_data.shape[0], -1)
        l1_data = test_data[:, 1, :].reshape(test_data.shape[0], -1)
        concatenated_data = torch.cat([h1_data, l1_data], dim=1)
        data_test.append(concatenated_data)
        params_test.append(test_params)
    all_test_data = torch.cat(data_test, dim=0)
    all_test_params = torch.cat(params_test, dim=0)

    print("  Normalizing test waveform data (per-sample)...")
    test_data_mean = all_test_data.mean(dim=1, keepdim=True)
    test_data_std = all_test_data.std(dim=1, keepdim=True).clamp(min=1e-8)
    all_test_data = (all_test_data - test_data_mean) / test_data_std

    nan_mask_test = torch.isnan(all_test_data).any(dim=1) | torch.isinf(all_test_data).any(dim=1)
    if nan_mask_test.any():
        n_bad = nan_mask_test.sum().item()
        print(f"  ⚠ Removing {n_bad} test samples with NaN/Inf values")
        all_test_data = all_test_data[~nan_mask_test]
        all_test_params = all_test_params[~nan_mask_test]

    all_params_physical = denormalize_params(all_params.numpy(), param_norm_info, component_param_names)
    all_test_params_physical = denormalize_params(all_test_params.numpy(), param_norm_info, component_param_names)

    if USE_REPARAMETERIZED_TARGETS:
        print("  Reparameterizing targets: [mass1, mass2, spin1z, spin2z] -> [chirp_mass, q, chi_eff, chi_a]")
        train_targets_physical, all_reparam_names = transform_component_params_to_reparameterized(
            all_params_physical, component_param_names
        )
        test_targets_physical, _ = transform_component_params_to_reparameterized(
            all_test_params_physical, component_param_names
        )
        target_param_names = list(all_reparam_names)
    else:
        print("  Using original component targets: [mass1, mass2, spin1z, spin2z]")
        train_targets_physical = np.array(all_params_physical, copy=True)
        test_targets_physical = np.array(all_test_params_physical, copy=True)
        target_param_names = list(component_param_names)

    selected_indices = []
    for mp in MODEL_PARAMS:
        if mp in target_param_names:
            selected_indices.append(target_param_names.index(mp))
        else:
            raise ValueError(
                f"MODEL_PARAMS entry '{mp}' not found in available target params: {target_param_names}"
            )
    train_targets_physical = train_targets_physical[:, selected_indices]
    test_targets_physical = test_targets_physical[:, selected_indices]
    target_param_names = list(MODEL_PARAMS)
    print(f"  Selected model target columns: {target_param_names}")

    train_targets_norm, target_param_norm_info = normalize_params_with_reference(
        train_targets_physical, target_param_names, reference_norm_info=None
    )
    test_targets_norm, _ = normalize_params_with_reference(
        test_targets_physical, target_param_names, reference_norm_info=target_param_norm_info
    )

    all_params = torch.from_numpy(train_targets_norm).float()
    all_test_params = torch.from_numpy(test_targets_norm).float()

    all_val_params_physical = denormalize_params(all_val_params.numpy(), param_norm_info, component_param_names)
    if USE_REPARAMETERIZED_TARGETS:
        val_targets_physical, _ = transform_component_params_to_reparameterized(
            all_val_params_physical, component_param_names
        )
    else:
        val_targets_physical = np.array(all_val_params_physical, copy=True)
    val_targets_physical = val_targets_physical[:, selected_indices]
    val_targets_norm, _ = normalize_params_with_reference(
        val_targets_physical, target_param_names, reference_norm_info=target_param_norm_info
    )
    all_val_params = torch.from_numpy(val_targets_norm).float()

    print(f"  Test data: {all_test_data.shape}, params: {all_test_params.shape}")
    print("✓ Data preparation complete (two-detector concatenation, normalized)\n")

    prepared_outputs = (
        all_data,
        all_params,
        all_val_data,
        all_val_params,
        all_test_data,
        all_test_params,
        target_param_norm_info,
        target_param_names,
        time_delay_default,
    )

    save_prepared_waveform_cache(
        num_samples=num_samples,
        prepared_outputs=prepared_outputs,
        generator_metadata=metadata,
        component_param_names=component_param_names,
        generator_backend=generator_backend,
        psd_csv=psd_csv,
    )

    return prepared_outputs


# ==============================================================================
# MAIN: Configuration + Data Generation + Save to Disk
# ==============================================================================

if __name__ == '__main__':
    # -----------------------------------------------------------------------
    # CONFIGURATION
    # -----------------------------------------------------------------------
    DENORMALIZE_PARAMETERS = True

    #  symmetric : Train on reparameterized params (chirp_mass, q, chi_eff, chi_a + extras)
    #  component : Train on component params (mass1, mass2, spin1z, spin2z, coa_phase, distance)
    PARAM_SET = 'symmetric'  # 'symmetric' or 'component'

    MODIFIED_GRAVITY = True
    LAMBDA_G_MIN = 1e14
    LAMBDA_G_MAX = 1e16
    if MODIFIED_GRAVITY:
        print(f"MG mode enabled: lambda_g range [{LAMBDA_G_MIN:.1e}, {LAMBDA_G_MAX:.1e}] m")

    # Choose which parameters the model trains on and infers.
    # For PARAM_SET='symmetric', choose from: chirp_mass, q, chi_eff, chi_a
    # For PARAM_SET='component', choose from: mass1, mass2, spin1z, spin2z, coa_phase, distance
    # If MODIFIED_GRAVITY=True, you can also include: lambda_g
    MODEL_PARAMS = ['chirp_mass', 'q', 'chi_eff', 'chi_a', 'lambda_g']

    NUM_TRAINING_SAMPLES = 30000

    # Data generation parameters
    ADD_NOISE = True           # Add realistic detector noise to waveforms
    WHITEN = True              # Needed — noise is colored (aLIGOZeroDetHighPower PSD)
    USE_WAVEFORM_CACHE = True  # Reuse prepared waveform datasets when config matches exactly
    GENERATOR_BACKEND = 'standard'  # 'standard' or 'real_psd'
    REAL_PSD_CSV = 'test7_o4_psds_all.csv'
    NUM_WORKERS = 2

    DATA_GENERATOR_BATCH_SIZE = 8
    DATA_GENERATOR_DETECTORS = ['H1', 'L1']

    # Set this once to switch both the "_first2.csv" and "_all.csv" inputs used below.
    REAL_DATA_CSV_PREFIX = 'test7_o4_whitened'

    # Path to a pre-generated dataset from GenerateDataset.py.
    # Leave as None to generate (and optionally cache) data fresh on each run.
    DATASET_PATH = None  # e.g. "../datasets/dataset_mg_standard_N30000_abc.pt"

    # -----------------------------------------------------------------------
    # DERIVED CONFIG
    # -----------------------------------------------------------------------
    _SYMMETRIC_ALL = ['chirp_mass', 'q', 'chi_eff', 'chi_a']
    _COMPONENT_ALL = ['mass1', 'mass2', 'spin1z', 'spin2z', 'coa_phase', 'distance']
    _MG_PARAM = ['lambda_g'] if MODIFIED_GRAVITY else []

    if PARAM_SET == 'symmetric':
        USE_REPARAMETERIZED_TARGETS = True
        _VALID_PARAMS = _SYMMETRIC_ALL + _MG_PARAM
        EXTRA_PARAM_NAMES = [p for p in MODEL_PARAMS if p not in _SYMMETRIC_ALL]
    elif PARAM_SET == 'component':
        USE_REPARAMETERIZED_TARGETS = False
        _VALID_PARAMS = _COMPONENT_ALL + _MG_PARAM
        EXTRA_PARAM_NAMES = [p for p in MODEL_PARAMS if p not in ['mass1', 'mass2', 'spin1z', 'spin2z']]
    else:
        raise ValueError(f"Unknown PARAM_SET: {PARAM_SET}. Must be 'symmetric' or 'component'.")

    _invalid = [p for p in MODEL_PARAMS if p not in _VALID_PARAMS]
    if _invalid:
        raise ValueError(
            f"Invalid MODEL_PARAMS for PARAM_SET='{PARAM_SET}': {_invalid}\n"
            f"Valid choices: {_VALID_PARAMS}"
        )

    PARAM_DIM = len(MODEL_PARAMS)

    print("\n" + "="*70)
    print("DATA GENERATION CONFIGURATION")
    print(f"  Parameter Set (PARAM_SET):               {PARAM_SET}")
    print(f"  Model Parameters (MODEL_PARAMS):         {MODEL_PARAMS}")
    print(f"  Parameter Dimension (PARAM_DIM):         {PARAM_DIM}")
    print(f"  Training Samples (NUM_TRAINING_SAMPLES): {NUM_TRAINING_SAMPLES:,}")
    print(f"  Reparameterized Targets:                 {USE_REPARAMETERIZED_TARGETS}")
    print(f"  Extra Params (pass-through):             {EXTRA_PARAM_NAMES}")
    print(f"  Add Noise (ADD_NOISE):                   {ADD_NOISE}")
    print(f"  Whiten Data (WHITEN):                    {WHITEN}")
    print(f"  Modified Gravity (MODIFIED_GRAVITY):     {MODIFIED_GRAVITY}")
    if MODIFIED_GRAVITY:
        print(f"    lambda_g range:                        [{LAMBDA_G_MIN:.1e}, {LAMBDA_G_MAX:.1e}] m")
    print(f"  Generator Backend:                       {GENERATOR_BACKEND}")
    print(f"  Use Waveform Cache:                      {USE_WAVEFORM_CACHE}")
    print("="*70 + "\n")

    # -----------------------------------------------------------------------
    # DATA GENERATION
    # -----------------------------------------------------------------------
    if DATASET_PATH:
        print(f"Loading pre-generated dataset from: {DATASET_PATH}")
        _ds = DataPipeline.load_dataset(DATASET_PATH, batch_size=64)
        def _all_tensors(loader):
            Xs, ys = [], []
            for b in loader:
                Xs.append(b[0]); ys.append(b[1])
            return torch.cat(Xs), torch.cat(ys)
        pycbc_data,      pycbc_params      = _all_tensors(_ds["train_loader"])
        pycbc_val_data,  pycbc_val_params  = _all_tensors(_ds["val_loader"])
        pycbc_test_data, pycbc_test_params = _all_tensors(_ds["test_loader"])
        param_norm_info   = _ds["metadata"].get("param_norm_info", {})
        model_param_names = _ds["metadata"].get("param_names", MODEL_PARAMS)
        GPS_TIME_DELAY    = _ds["metadata"].get("gps_time_delay", 0.0)
    else:
        (pycbc_data, pycbc_params,
         pycbc_val_data, pycbc_val_params,
         pycbc_test_data, pycbc_test_params,
         param_norm_info, model_param_names,
         GPS_TIME_DELAY) = prepare_pycbc_data(
            num_samples=NUM_TRAINING_SAMPLES,
            generator_backend=GENERATOR_BACKEND,
            psd_csv=REAL_PSD_CSV
        )

    print(f"\n✓ Data ready:")
    print(f"  Training samples: {len(pycbc_params)}")
    print(f"  Validation samples: {len(pycbc_val_params)}")
    print(f"  Test samples: {len(pycbc_test_params)}")
    print(f"  Data dimension: {pycbc_data.shape[1]} (2 detectors concatenated)")
    print(f"  Target parameters: {model_param_names}")
    print(f"  GPS time delay: {GPS_TIME_DELAY*1000:.3f} ms")

    # -----------------------------------------------------------------------
    # SAVE PREPARED DATA TO DISK
    # -----------------------------------------------------------------------
    PREPARED_DATA_PATH = 'prepared_data.pt'
    torch.save({
        'train_data':    pycbc_data,
        'train_params':  pycbc_params,
        'val_data':      pycbc_val_data,
        'val_params':    pycbc_val_params,
        'test_data':     pycbc_test_data,
        'test_params':   pycbc_test_params,
        'param_norm_info': param_norm_info,
        'param_names':   model_param_names,
        'gps_time_delay': GPS_TIME_DELAY,
        # Store config metadata for downstream scripts
        'config': {
            'param_set': PARAM_SET,
            'model_params': MODEL_PARAMS,
            'modified_gravity': MODIFIED_GRAVITY,
            'lambda_g_min': LAMBDA_G_MIN if MODIFIED_GRAVITY else None,
            'lambda_g_max': LAMBDA_G_MAX if MODIFIED_GRAVITY else None,
            'add_noise': ADD_NOISE,
            'whiten': WHITEN,
            'num_training_samples': NUM_TRAINING_SAMPLES,
            'extra_param_names': EXTRA_PARAM_NAMES,
            'use_reparameterized_targets': USE_REPARAMETERIZED_TARGETS,
        },
    }, PREPARED_DATA_PATH)
    print(f"\n✓ Prepared data saved to: {PREPARED_DATA_PATH}")
    print("  Next step: run train_model.py")
