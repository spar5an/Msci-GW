"""
GenerateDataset — standalone data generation script.

Run this independently of any training script to pre-generate and save a
dataset. The saved .pt file can then be loaded by Model.py,
Model_LV.py, or dingo_npe_gpu_two_detectors.py by setting DATASET_PATH.

Usage
-----
Edit the configuration block below, then run::

    cd Msci-GW/HPC
    python GenerateDataset.py

On the HPC, submit as a CPU-only job before the training job::

    qsub GenerateDataJob.pbs      # (see TestPycbcJob.pbs for PBS template)

The script will print the saved file path on completion. Set that path as
DATASET_PATH in your training script to skip data generation entirely.
"""

import os
import sys
import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────
# Physics mode: 'gr' | 'mg' | 'lv'
MODE                = 'gr'

# Noise backend: 'standard' (analytical aLIGO PSD, always available) |
#                'real_psd' (real O4 PSDs from CSV — 'lv' mode does not support this)
GENERATOR_BACKEND   = 'standard'

# Path to real PSD CSV — only needed when GENERATOR_BACKEND = 'real_psd'
# Expected columns: event_rank, event_name, gps, detector, frequency, psd
REAL_PSD_CSV        = 'test7_o4_psds_all.csv'

# Number of waveforms to generate
NUM_SAMPLES         = 30000

# Detector noise settings
ADD_NOISE           = True
WHITEN              = True

# Detector network
DETECTORS           = ['H1', 'L1']

# Parallelism
NUM_WORKERS         = 4

# Waveform settings (defaults chosen per mode if None)
APPROXIMANT         = None      # None → 'IMRPhenomXP' (gr), 'IMRPhenomD' (mg/lv)
F_LOWER             = None      # None → 40.0 (gr), 30.0 (mg/lv)
F_FINAL             = 2048.0    # Upper cutoff for mg/lv
SIGNAL_LENGTH       = 2.0       # Waveform duration in seconds

# Train/val/test split
TRAIN_SPLIT         = 0.8
VAL_SPLIT           = 0.1

# ── Physics-mode-specific settings ────────────────────────────────────────────

# MG (mode='mg') and LV (mode='lv'): graviton Compton wavelength range
LAMBDA_G_MIN        = 1e14      # metres
LAMBDA_G_MAX        = 1e16

# LV only (mode='lv')
ALPHA_LV            = 0         # LV dispersion exponent; e.g. 2.5, 3.0, 4.0
A_LV_MIN            = 1e-20
A_LV_MAX            = 1e-18

# ── Output ────────────────────────────────────────────────────────────────────
# Directory to save the dataset (relative to this script's location)
OUTPUT_DIR          = '../datasets'

# Override the auto-generated filename (leave None to use the hash-based name)
OUTPUT_PATH         = None

# ── Parameter distributions ───────────────────────────────────────────────────
# Customise the physical parameter distributions here.
# Each value must be a callable: lambda size: np.random.*(low, high, size=size)
PARAM_DISTRIBUTIONS = {
    'mass1':       lambda size: np.random.uniform(10, 80,   size=size),
    'mass2':       lambda size: np.random.uniform(10, 80,   size=size),
    'spin1z':      lambda size: np.random.uniform(-0.99, 0.99, size=size),
    'spin2z':      lambda size: np.random.uniform(-0.99, 0.99, size=size),
    'distance':    lambda size: np.random.uniform(100, 1000, size=size),
    'inclination': lambda size: np.random.uniform(0, np.pi, size=size),
    'coa_phase':   lambda size: np.random.uniform(0, 2*np.pi, size=size),
    'ra':          lambda size: np.random.uniform(0, 2*np.pi, size=size),
    'dec':         lambda size: np.arcsin(np.random.uniform(-1, 1, size=size)),
    'polarization':lambda size: np.random.uniform(0, np.pi, size=size),
}
# ──────────────────────────────────────────────────────────────────────────────


if __name__ == '__main__':
    # Add HPC directory to path so DataPipeline can be imported
    hpc_dir = os.path.dirname(os.path.abspath(__file__))
    if hpc_dir not in sys.path:
        sys.path.insert(0, hpc_dir)

    from DataPipeline import generate_dataset, save_dataset, build_dataset_path

    # Resolve defaults
    mode = MODE.lower()
    approximant = APPROXIMANT or ('IMRPhenomXP' if mode == 'gr' else 'IMRPhenomD')
    f_lower     = F_LOWER     or (40.0 if mode == 'gr' else 30.0)

    # Build full config
    config = {
        **PARAM_DISTRIBUTIONS,
        'mode':               mode,
        'generator_backend':  GENERATOR_BACKEND,
        'num_samples':        NUM_SAMPLES,
        'add_noise':          ADD_NOISE,
        'whiten':             WHITEN,
        'detectors':          DETECTORS,
        'num_workers':        NUM_WORKERS,
        'approximant':        approximant,
        'f_lower':            f_lower,
        'f_final':            F_FINAL,
        'signal_length':      SIGNAL_LENGTH,
        'train_split':        TRAIN_SPLIT,
        'val_split':          VAL_SPLIT,
    }

    if GENERATOR_BACKEND == 'real_psd':
        config['psd_csv'] = REAL_PSD_CSV

    if mode in ('mg', 'lv'):
        config['lambda_g_range'] = (LAMBDA_G_MIN, LAMBDA_G_MAX)

    if mode == 'lv':
        config['alpha_lv']  = ALPHA_LV
        config['a_lv_range'] = (A_LV_MIN, A_LV_MAX)

    # Resolve output path
    out_dir  = os.path.abspath(os.path.join(hpc_dir, OUTPUT_DIR))
    out_path = OUTPUT_PATH or build_dataset_path(config, output_dir=out_dir)

    # Skip if already exists
    if os.path.exists(out_path):
        print(f"Dataset already exists, skipping generation: {out_path}")
        sys.exit(0)

    print("=" * 60)
    print("DATASET GENERATION")
    print("=" * 60)
    print(f"  Mode:              {mode.upper()}")
    print(f"  Backend:           {GENERATOR_BACKEND}")
    print(f"  Samples:           {NUM_SAMPLES:,}")
    print(f"  Detectors:         {DETECTORS}")
    print(f"  Add noise:         {ADD_NOISE}")
    print(f"  Whiten:            {WHITEN}")
    print(f"  Approximant:       {approximant}")
    print(f"  f_lower:           {f_lower} Hz")
    if mode in ('mg', 'lv'):
        print(f"  lambda_g range:    [{LAMBDA_G_MIN:.1e}, {LAMBDA_G_MAX:.1e}] m")
    if mode == 'lv':
        print(f"  alpha_lv:          {ALPHA_LV}")
        print(f"  A_lv range:        [{A_LV_MIN:.1e}, {A_LV_MAX:.1e}]")
    print(f"  Output:            {out_path}")
    print("=" * 60)

    result = generate_dataset(config)
    save_dataset(result, out_path, config)

    print()
    print(f"Done. To use this dataset in training, set:")
    print(f"  DATASET_PATH = '{out_path}'")
