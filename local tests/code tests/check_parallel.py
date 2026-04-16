"""
check_parallel.py — Verify that parallel waveform generation works correctly.

Run from this directory:
    python check_parallel.py
"""

import sys
import time
import numpy as np
import torch

from gw_datagen import (
    pycbc_data_generator,
    pycbc_massive_gravity_data_generator,
    pycbc_lorentz_violation_data_generator,
)

CONFIG = {
    'mass1':        lambda size: np.random.uniform(20, 40, size=size),
    'mass2':        lambda size: np.random.uniform(20, 40, size=size),
    'spin1z':       lambda size: np.zeros(size),
    'spin2z':       lambda size: np.zeros(size),
    'distance':     lambda size: np.random.uniform(200, 500, size=size),
    'inclination':  lambda size: np.zeros(size),
    'coa_phase':    lambda size: np.zeros(size),
    'ra':           lambda size: np.zeros(size),
    'dec':          lambda size: np.zeros(size),
    'polarization': lambda size: np.zeros(size),
    'redshift':     lambda size: np.random.uniform(0.05, 0.15, size=size),
}

KWARGS = dict(
    num_samples=16,
    time_resolution=1/4096,
    approximant='IMRPhenomD',
    f_lower=40.0,
    f_final=2048.0,
    signal_length=2.0,
    batch_size=8,
    train_split=0.7,
    val_split=0.15,
    add_noise=True,
    show_progress=False,
)

CASES = [
    ('GR',  pycbc_data_generator,                   {}),
    ('MG',  pycbc_massive_gravity_data_generator,   {'lambda_g': 1e22}),
    ('LV',  pycbc_lorentz_violation_data_generator, {'lambda_g': 1e22, 'alpha_lv': 3.0, 'A_lv': 1e15}),
]

NUM_SAMPLES = KWARGS['num_samples']

def check_result(result):
    m = result['metadata']
    X = result['train_loader'].dataset.dataset.tensors[0]
    assert m['num_samples'] == NUM_SAMPLES, f"expected {NUM_SAMPLES} samples, got {m['num_samples']}"
    assert m['num_failed'] == 0, f"{m['num_failed']} waveforms failed"
    assert m['train_size'] + m['val_size'] + m['test_size'] == NUM_SAMPLES
    assert torch.isfinite(X).all(), "non-finite values in output"
    assert X.abs().sum().item() > 0, "output is all zeros"

if __name__ == '__main__':
    num_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    print(f"Testing parallel generation with num_workers={num_workers}, num_samples=32\n")

    all_ok = True
    for name, fn, extra in CASES:
        print(f"--- {name} ---")
        t0 = time.perf_counter()
        try:
            result = fn(CONFIG, **extra, **KWARGS, num_workers=num_workers)
            check_result(result)
            print(f"PASS  ({time.perf_counter()-t0:.1f}s)\n")
        except Exception as e:
            print(f"FAIL  ({time.perf_counter()-t0:.1f}s): {e}\n")
            all_ok = False

    sys.exit(0 if all_ok else 1)
