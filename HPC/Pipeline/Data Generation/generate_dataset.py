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
    save_dataloaders,
)

# ── CONFIG SECTION — edit everything below this line ─────────────────────────

# Physics mode:
#   'gr' — General Relativity
#   'mg' — Massive Graviton modified gravity
#   'lv' — Lorentz Violation
# MODE drives which generator is called and which physics parameters are sampled.
# All three physics labels (m_g, alpha_lv, A) are always stored in the dataset
# so that all modes produce the same label dimensions for ML; unused params are 0.
MODE = 'gr'

NUM_SAMPLES  = 100
OUTPUT_PATH  = 'dataset.pt'
ADD_NOISE    = True
NUM_WORKERS  = 4

# ── Waveform settings ─────────────────────────────────────────────────────────
TIME_RESOLUTION = 1 / 4096   # seconds per sample
SIGNAL_LENGTH   = 2.0        # seconds
F_LOWER         = 30.0       # Hz
F_FINAL         = 2048.0     # Hz
APPROXIMANT     = 'IMRPhenomD'

# ── Modified-gravity parameter bounds ─────────────────────────────────────────
# Graviton mass (used for MODE = 'mg' and 'lv')
# M_G_MIN = 2.21e-58 kg  ↔  lambda_g ≈ 1e16 m  (weak modification)
# M_G_MAX = 2.21e-56 kg  ↔  lambda_g ≈ 1e14 m  (strong modification)
M_G_MIN  = 2.21e-58   # kg
M_G_MAX  = 2.21e-56   # kg

# LV dispersion exponent (used for MODE = 'lv')
ALPHA_LV = 3.0   # e.g. 3.0 = doubly special relativity (DSR)

# LV dispersion coefficient A bounds in [eV]^{2-alpha} (used for MODE = 'lv')
# For alpha=3: A has units eV^{-1}.  Converted to lambda_A internally.
# A=5.07e21 eV^{-1}  ↔  lambda_A ≈ 1e15 m
A_MIN = 1e20   # eV^{-1} for alpha=3 (strong LV)
A_MAX = 1e22   # eV^{-1} for alpha=3 (weak LV)

# ── DataLoader settings ───────────────────────────────────────────────────────
BATCH_SIZE   = 256
TRAIN_SPLIT  = 0.8
VAL_SPLIT    = 0.1

# ── END OF CONFIG SECTION ─────────────────────────────────────────────────────

if __name__ == '__main__':
    # Astrophysical parameters — sampled identically for all modes
    CONFIG = {
        'mass1':        lambda size: np.random.uniform(10, 50, size=size),
        'mass2':        lambda size: np.random.uniform(10, 50, size=size),
        'spin1z':       lambda size: np.random.uniform(-0.99, 0.99, size=size),
        'spin2z':       lambda size: np.random.uniform(-0.99, 0.99, size=size),
        'distance':     lambda size: np.random.uniform(100, 1000, size=size),
        'inclination':  lambda size: np.random.uniform(0, np.pi, size=size),
        'coa_phase':    lambda size: np.random.uniform(0, 2 * np.pi, size=size),
        'ra':           lambda size: np.random.uniform(0, 2 * np.pi, size=size),
        'dec':          lambda size: np.arcsin(np.random.uniform(-1, 1, size=size)),
        'polarization': lambda size: np.random.uniform(0, np.pi, size=size),
    }

    # Physics labels — always present for consistent ML label dimensions.
    # Unused parameters are zero; they do not affect waveform generation.
    if MODE == 'gr':
        CONFIG['m_g']      = lambda size: np.zeros(size)
        CONFIG['alpha_lv'] = lambda size: np.zeros(size)
        CONFIG['A']        = lambda size: np.zeros(size)
    elif MODE == 'mg':
        CONFIG['m_g']      = lambda size: np.random.uniform(M_G_MIN, M_G_MAX, size=size)
        CONFIG['alpha_lv'] = lambda size: np.zeros(size)
        CONFIG['A']        = lambda size: np.zeros(size)
    elif MODE == 'lv':
        CONFIG['m_g']      = lambda size: np.random.uniform(M_G_MIN, M_G_MAX, size=size)
        CONFIG['alpha_lv'] = lambda size: np.full(size, ALPHA_LV)
        CONFIG['A']        = lambda size: np.random.uniform(A_MIN, A_MAX, size=size)
    else:
        raise ValueError(f"Unknown MODE '{MODE}'. Choose from: 'gr', 'mg', 'lv'")

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
        result = pycbc_data_generator(f_final=F_FINAL, **common_kwargs)

    elif MODE == 'mg':
        result = pycbc_massive_gravity_data_generator(f_final=F_FINAL, **common_kwargs)

    elif MODE == 'lv':
        result = pycbc_lorentz_violation_data_generator(
            alpha_lv=ALPHA_LV, f_final=F_FINAL, **common_kwargs)

    save_dataloaders(result, OUTPUT_PATH)
    print(f"\nDataset saved to {OUTPUT_PATH}")
    meta = result['metadata']
    print(f"  Waveform shape : {meta['waveform_shape']}")
    print(f"  Train / val / test : {meta['train_size']} / {meta['val_size']} / {meta['test_size']}")
    print(f"  Parameters : {meta['parameter_names']}")
