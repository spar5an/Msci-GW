"""
generate_dataset.py — Generate and save a gravitational wave dataset.

Edit the CONFIG SECTION below, then run:
    python generate_dataset.py

The dataset is saved as a .pt file and can be reloaded with:
    from gw_datagen import load_dataloaders
    result = load_dataloaders('dataset.pt')
"""

import numpy as np
from gw_datagen import (
    pycbc_data_generator,
    pycbc_massive_gravity_data_generator,
    pycbc_lorentz_violation_data_generator,
    pycbc_data_generator_real_psd,
    pycbc_modified_data_generator_real_psd,
    save_dataloaders,
)

# ── CONFIG SECTION — edit everything below this line ─────────────────────────

# Physics mode:
#   'gr'          — General Relativity (o4_psd GWOSC-cached noise)
#   'mg'          — Massive Graviton modified gravity (o4_psd noise)
#   'lv'          — Lorentz Violation (o4_psd noise)
#   'gr_real_psd' — GR with CSV-based real PSD noise
#   'mg_real_psd' — Massive Graviton with CSV-based real PSD noise
MODE = 'gr'

NUM_SAMPLES   = 100
OUTPUT_PATH   = 'dataset.pt'
ADD_NOISE     = True
NUM_WORKERS   = 4
# Noise backend: 'aligo' uses the fast analytical aLIGO PSD (no network access required).
#                'o4_psd' uses real O4 PSD data cached from GWOSC (slower, requires internet).
NOISE_BACKEND = 'aligo'

# Parameter distributions — add/remove keys to change what is sampled.
# Each value must be a function of (size,) returning a numpy array.
CONFIG = {
    'mass1':       lambda size: np.random.uniform(10, 50, size=size),
    'mass2':       lambda size: np.random.uniform(10, 50, size=size),
    'spin1z':      lambda size: np.random.uniform(-0.99, 0.99, size=size),
    'spin2z':      lambda size: np.random.uniform(-0.99, 0.99, size=size),
    'distance':    lambda size: np.random.uniform(100, 1000, size=size),
    'inclination': lambda size: np.random.uniform(0, np.pi, size=size),
    'coa_phase':   lambda size: np.random.uniform(0, 2 * np.pi, size=size),
    'ra':          lambda size: np.random.uniform(0, 2 * np.pi, size=size),
    'dec':         lambda size: np.arcsin(np.random.uniform(-1, 1, size=size)),
    'polarization':lambda size: np.random.uniform(0, np.pi, size=size),
    'redshift':    lambda size: np.random.uniform(0.01, 0.5, size=size),
}

# ── Waveform settings ────────────────────────────────────────────────────────
TIME_RESOLUTION = 1 / 4096   # seconds per sample
SIGNAL_LENGTH   = 2.0        # seconds
F_LOWER         = 30.0       # Hz
F_FINAL         = 2048.0     # Hz
APPROXIMANT     = 'IMRPhenomD'

# ── MG / LV parameters (used only for 'mg', 'lv', 'mg_real_psd') ────────────
# Set LAMBDA_G = None to sample per-waveform from CONFIG (add 'lambda_g' key)
LAMBDA_G = 1e15   # graviton Compton wavelength in metres

# LV parameters (used only for MODE = 'lv')
ALPHA_LV = 3.0    # dispersion exponent (e.g. 3 = doubly special relativity)
A_LV     = 1e15   # LV Compton wavelength in metres (np.inf to suppress LV term)

# ── CSV PSD path (used only for 'gr_real_psd' or 'mg_real_psd') ─────────────
PSD_CSV = 'o4_psds.csv'

# ── DataLoader settings ───────────────────────────────────────────────────────
BATCH_SIZE   = 256
TRAIN_SPLIT  = 0.8
VAL_SPLIT    = 0.1

# ── END OF CONFIG SECTION ─────────────────────────────────────────────────────

if __name__ == '__main__':
    common_kwargs = dict(
        config=CONFIG,
        num_samples=NUM_SAMPLES,
        time_resolution=TIME_RESOLUTION,
        approximant=APPROXIMANT,
        f_lower=F_LOWER,
        signal_length=SIGNAL_LENGTH,
        batch_size=BATCH_SIZE,
        train_split=TRAIN_SPLIT,
        val_split=VAL_SPLIT,
        add_noise=ADD_NOISE,
        num_workers=NUM_WORKERS,
    )

    if MODE == 'gr':
        result = pycbc_data_generator(
            f_final=F_FINAL,
            noise_backend=NOISE_BACKEND,
            **common_kwargs,
        )

    elif MODE == 'mg':
        result = pycbc_massive_gravity_data_generator(
            lambda_g=LAMBDA_G,
            f_final=F_FINAL,
            noise_backend=NOISE_BACKEND,
            **common_kwargs,
        )

    elif MODE == 'lv':
        result = pycbc_lorentz_violation_data_generator(
            alpha_lv=ALPHA_LV,
            A_lv=A_LV,
            lambda_g=LAMBDA_G,
            f_final=F_FINAL,
            noise_backend=NOISE_BACKEND,
            **common_kwargs,
        )

    elif MODE == 'gr_real_psd':
        result = pycbc_data_generator_real_psd(
            psd_csv=PSD_CSV,
            **common_kwargs,
        )

    elif MODE == 'mg_real_psd':
        result = pycbc_modified_data_generator_real_psd(
            psd_csv=PSD_CSV,
            lambda_g=LAMBDA_G,
            f_final=F_FINAL,
            **common_kwargs,
        )

    else:
        raise ValueError(
            f"Unknown MODE '{MODE}'. "
            "Choose from: 'gr', 'mg', 'lv', 'gr_real_psd', 'mg_real_psd'"
        )

    save_dataloaders(result, OUTPUT_PATH)
    print(f"\nDataset saved to {OUTPUT_PATH}")
    meta = result['metadata']
    print(f"  Waveform shape : {meta['waveform_shape']}")
    print(f"  Train / val / test : {meta['train_size']} / {meta['val_size']} / {meta['test_size']}")
    print(f"  Parameters : {meta['parameter_names']}")
