# gw_datagen — Code Reference

**File:** `gw_datagen.py`  
**Purpose:** Gravitational wave dataset generation for neural network training. Produces labelled PyTorch `DataLoader` objects containing simulated detector strain signals with optional realistic noise. Covers three physical models: standard GR, massive graviton (MG), and generalised Lorentz violation (LV).

---

## Architecture Overview

```
config (parameter distributions)
        │
        ▼
_generate_parameter_sets()          ← draw N param dicts up-front
        │
        ▼
[per waveform worker]               ← one of three pipelines below
        │
        ├── GR:  _generate_single_waveform()
        ├── MG:  _generate_single_modified_waveform()
        └── LV:  _generate_single_lv_waveform()
        │
        ▼
_generate_*_parallel()              ← serial (num_workers=1) or Pool
        │
        ▼
assemble signal_array (N, D, T)     ← numpy float32
        │
        ▼
normalize_parameters() [z-score]
        │
        ▼
TensorDataset → random_split → DataLoaders
        │
        ▼
{ train_loader, val_loader, test_loader, metadata }
```

---

## Top-Level Generators

All three return the same dict structure:

```python
{
    'train_loader': DataLoader,
    'val_loader':   DataLoader,
    'test_loader':  DataLoader,
    'metadata':     dict,
}
```

### `pycbc_data_generator` — GR baseline

```python
pycbc_data_generator(
    config,
    num_samples,
    time_resolution = 1/4096,      # delta_t in seconds → sample rate 4096 Hz
    approximant     = 'IMRPhenomD',
    f_lower         = 40.0,        # Hz
    f_final         = 2048.0,      # Hz
    highpass_fc     = 35.0,        # Hz — applied after IRFFT
    num_workers     = None,        # None → 1 (serial); >1 uses multiprocessing.Pool
    signal_length   = 2.0,         # seconds
    batch_size      = 256,
    chunk_size      = 10000,       # waveforms processed per memory chunk
    train_split     = 0.8,
    val_split       = 0.1,         # test = 1 - train - val
    show_progress   = True,
    detectors       = ['H1','L1'],
    add_noise       = True,
    noise_backend   = 'aligo',     # 'aligo' or 'o4_psd'
    psd_cache_dir   = <default>,   # path to .npz cache used by o4_psd backend
)
```

The GR worker uses a **FD→IRFFT** pipeline (not PyCBC's native TD generator) so that GR and modified-gravity datasets are assembled identically. No phase modification is applied.

### `pycbc_massive_gravity_data_generator` — Massive graviton (MG)

Same interface as GR plus:

```python
m_g = ...    # graviton mass in kg (required)
```

Adds the massive graviton phase shift `δΨ = −β · u⁻¹` in the frequency domain before IRFFT (see Physics section). The parameter `m_g` can also be included in `config` to vary it per sample.

Metadata extras: `m_g`, `m_g_varied` (bool).

### `pycbc_lorentz_violation_data_generator` — Lorentz violation (LV)

Same interface as GR plus:

```python
alpha_lv = ...    # dispersion exponent α (required)
A        = ...    # LV Compton wavelength in metres (optional)
m_g      = ...    # graviton mass in kg; omit or set None to suppress MG term (optional)
```

Applies the generalised LV phase from Mirshekari, Yunes & Will (2011). Both `m_g` and `A` can be included in `config` for per-sample variation.

Metadata extras: `alpha_lv`, `A`, `m_g`, `m_g_varied`, `A_varied`.

---

## The `config` Dict

Every generator takes a `config` dict mapping parameter names to **callables** that accept a `size` keyword and return a numpy array of that length. All values are sampled up-front before any waveform generation begins.

```python
config = {
    # Required
    'mass1':        lambda size: np.random.uniform(10, 80, size=size),
    'mass2':        lambda size: np.random.uniform(10, 80, size=size),

    # Optional — defaults used if absent
    'spin1z':       lambda size: np.zeros(size),           # default 0.0
    'spin2z':       lambda size: np.zeros(size),           # default 0.0
    'distance':     lambda size: np.random.uniform(100, 1000, size=size),  # Mpc; default 410
    'inclination':  lambda size: np.zeros(size),           # radians; default 0
    'coa_phase':    lambda size: np.zeros(size),           # radians; default 0
    'ra':           lambda size: ...,                      # radians; default 0
    'dec':          lambda size: ...,                      # radians; default π/2
    'polarization': lambda size: ...,                      # radians; default 0
    'gps_time':     lambda size: ...,                      # seconds; default 1126259462.4
    'redshift':     lambda size: np.random.uniform(0.05, 0.5, size=size),  # default 0.1

    # MG/LV only — can be fixed or varied per sample
    'm_g':          lambda size: ...,                      # kg  (graviton mass)
    'A':            lambda size: ...,                      # metres  (LV Compton wavelength)
}
```

`_validate_config()` checks that every value is callable and that a test call with `size=2` returns a numpy array.

---

## Waveform Generation Pipeline (per sample)

All three workers share the same FD→TD pipeline. The MG and LV workers additionally apply a phase modification before the IRFFT.

### Step 1: FD waveform generation

```python
hp_fd, hc_fd = get_fd_waveform(
    approximant=approximant,   # 'IMRPhenomD' default
    delta_f=1/256,             # hardcoded frequency resolution
    f_lower=f_lower,
    f_final=f_final,           # GR/MG/LV all pass f_final
    mass1, mass2, spin1z, spin2z, inclination, coa_phase, distance
)
```

The `delta_f = 1/256 Hz` is hardcoded in all three workers (256-second effective observation window in frequency space).

### Step 2: Phase modification (MG and LV only)

```python
freqs = hp_fd.sample_frequencies.numpy()[1:]   # drop DC bin
phase_shift = _additional_phase(freqs, chirp_mass, z, lambda_g, f_pn_cutoff=f_pn)
# or for LV:
phase_shift = _additional_phase_lv(freqs, chirp_mass, z, lambda_g, alpha_lv, A_lv, f_c, f_pn_cutoff=f_pn)

hp_fd.numpy()[1:] *= exp(1j * phase_shift)
hc_fd.numpy()[1:] *= exp(1j * phase_shift)
```

The PN breakdown frequency `f_pn = 0.1 / (π · (m1+m2) · M_sun_sec)` is computed per sample; above this frequency the phase is tapered to zero via `_cos_taper()` because the PN dispersion formula is not valid in the merger/ringdown regime.

### Step 3: IRFFT and amplitude rescaling

```python
hp_raw = np.fft.irfft(hp_fd.numpy())
hp_raw *= delta_f * N   # restore correct strain amplitude (delta_f * N = 1/dt)
```

This is equivalent to the standard IFFT normalisation convention used by PyCBC.

### Step 4: Time-domain windowing

The IRFFT array has length `N = 2*(len(hp_fd)-1)`. The target output is `target_length = int(signal_length / time_resolution)` samples (e.g. 8192 for 2 s at 4096 Hz).

```
if target_length <= N:
    # Take the last (target_length - 500) samples [late inspiral]
    # then the first 500 samples [merger + early ringdown]
    hp_arr = concat(hp_raw[N - n_pre:], hp_raw[:500])
else:
    # Zero-pad the beginning (signal shorter than window)
    hp_arr = concat(zeros(target_length - N), hp_raw)
```

The `n_ringdown = 500` sample wrap ensures the merger peak appears at the end of the window, which is the conventional layout for matched-filter based studies.

### Step 5: End taper

```python
hp_arr = _apply_end_taper(hp_arr)
```

A cosine taper over the last 128 samples (`_RINGDOWN_TAPER_LEN`) forces the signal to zero at the array boundary. Without this the FIR highpass filter sees a hard step discontinuity and produces edge-effect ringing.

### Step 6: Highpass filter

```python
hp_ts = highpass_fir(TimeSeries(hp_arr, delta_t=time_resolution), highpass_fc=35.0, order=128)
```

Applied to both polarisations **before** detector projection. Removes sub-35 Hz content which is dominated by seismic noise in real detectors.

### Step 7: Detector projection

```python
signal = Detector(det_name).project_wave(hp_ts, hc_ts, ra, dec, polarization, method='lal')
```

Uses PyCBC's LAL-backed antenna pattern functions. The resulting `signal` is trimmed or zero-padded to exactly `target_length` samples.

### Step 8: Noise injection

```python
if noise_backend == 'aligo':
    psd = aLIGOZeroDetHighPower(flen, delta_f, f_lower)
else:
    psd = load_random_o4_psd(flen, delta_f, f_lower, det_name, ...)

noise = noise_from_psd(target_length, delta_t, psd)
noise._epoch = signal._epoch          # align timestamps
signal = signal.inject(noise)
```

The epoch alignment is essential: `inject()` performs a time-domain addition and requires both TimeSeries to share the same start time.

---

## Physics: Modified Dispersion Relations

### Massive Graviton — `_additional_phase`

Will (1997), arXiv:gr-qc/9709011. A graviton with non-zero mass propagates with a frequency-dependent velocity, causing low-frequency components to arrive later than high-frequency ones.

```
δΨ_MG = −β · u⁻¹

β = π² c D_0(z) M / (λ_g² (1+z))
u = π M f          (dimensionless PN frequency parameter)
M = M_chirp · M_sun_sec · (1+z)    (detector-frame chirp mass in seconds)
```

`D_0(z)` is the cosmological distance integral (`_D_alpha` with α=0), computed numerically with `scipy.integrate.quad` using a flat ΛCDM cosmology (H₀=67.4 km/s/Mpc, Ω_m=0.315).

### Lorentz Violation — `_additional_phase_lv`

Mirshekari, Yunes & Will (2011), arXiv:1110.2720. A parametrised post-Einsteinian (ppE) framework allowing generic modifications to the graviton dispersion relation:

```
E² = p²c² + m_g²c⁴ + A_physical · p^α · c^α
```

The total phase correction is:

```
δΨ = δΨ_MG + δΨ_LV

δΨ_LV = −ζ · (u^{α−1} − u_c^{α−1})     [α ≠ 1, 2]
δΨ_LV = +ζ · (ln u − ln u_c)             [α = 1]

ζ = π^{2−α}/(1−α) · c^{1−α} · D_α(z) · M^{1−α} / (A^{2−α} · (1+z)^{1−α})
```

The phase is normalised to zero at `u_c = π M f_c` where `f_c` is the maximum non-zero frequency of the waveform (peak of inspiral amplitude). This removes an unphysical divergence in `u^{α−1}` for α < 1.

**Key values of α:**

| α | Theory |
|---|--------|
| 0 | Degenerate with massive graviton |
| 1 | Logarithmic correction |
| 2 | Degenerate with time of coalescence — no observable effect |
| 2.5 | Non-commutative geometry |
| 3 | Doubly special relativity (DSR) |
| 4 | Extra dimensions / Hořava-Lifshitz gravity |

Setting `m_g = None` (or omitting it) suppresses the MG term; setting `A = np.inf` suppresses the LV term; `alpha_lv = 2` is also suppressed automatically.

---

## O4 PSD Cache

The `o4_psd` noise backend uses real Advanced LIGO O4a power spectral densities fetched from GWOSC. The cache must be built once before use.

### Building the cache

```python
from gw_datagen import build_o4_psd_cache
build_o4_psd_cache('H1', n_segments=100, sample_rate=4096)
build_o4_psd_cache('L1', n_segments=100, sample_rate=4096)
```

Or from the command line:

```bash
python download_o4_psds.py --detectors H1 L1 --n-segments 100
```

**What it does:**
1. Queries `gwosc` for O4a science-mode GPS segments (2023-05-24 → 2024-01-16)
2. Selects `n_segments` GPS start times spread across the run
3. Fetches 256-second data segments via `gwpy.timeseries.TimeSeries.fetch_open_data`
4. Computes a Welch PSD (4 s FFT, 50% overlap, Hann window) for each segment
5. Saves all PSDs to `o4_psd_cache/o4_psds_{detector}_{rate}Hz.npz`

The `.npz` file contains two arrays: `freqs` shape `(M,)` and `psds` shape `(N, M)` where N is the number of successfully fetched segments.

### Using cached PSDs — `load_random_o4_psd`

```python
psd = load_random_o4_psd(flen, delta_f, f_lower, detector, sample_rate, cache_dir)
```

On each call, a PSD is **randomly selected** from the cache and log-log interpolated onto the required frequency grid. Below `f_lower` the PSD is held constant at its value at `f_lower` (prevents catastrophic sub-band noise energy). The DC bin is set to 1.0 (arbitrary non-zero value, never used in practice).

Falls back silently to `aLIGOZeroDetHighPower` if the cache file is missing.

### Auto-build at generation time

When `noise_backend='o4_psd'` is passed to any top-level generator, `build_o4_psd_cache()` is called automatically for each requested detector before waveform generation begins. If the cache already exists the call is a no-op.

---

## Parallelism

Each generator has a `num_workers` parameter:

- `num_workers=1` (default): serial `map()` — no subprocess spawning. **Required inside pytest on WSL2/Linux** to avoid multiprocessing deadlocks.
- `num_workers=N` (N > 1): `multiprocessing.Pool(processes=N)` with `imap_unordered` and `chunksize=100`. Workers receive pre-computed parameter dicts via `functools.partial`.

All waveform errors are caught per-sample and returned as `{'success': False, 'error': str(e)}`. The top-level generator raises `RuntimeError("No waveforms were successfully generated!")` only if every single sample fails; partial failure is silently tolerated.

---

## Parameter and Signal Normalisation

### Parameters — `normalize_parameters`

Applied automatically inside every top-level generator with `method='zscore'`:

```
y_norm = (y − mean) / std     per column
```

The normalisation statistics (`mean`, `std`, `min`, `max`) are stored in `metadata['param_norm_info']` so they can be inverted at inference time.

### Signals — `normalize_waveforms` (optional, not called by default)

Available methods: `global_standardize` (recommended), `per_sample_minmax`, `per_sample_standardize`, `global_minmax`, `scale_constant`, `none`. Not called inside the top-level generators — apply manually if needed.

### Single-waveform utilities

- `normalize_waveform(waveform, scale_factor=1e21)` — multiply by a fixed scalar (useful for plotting; strain ~1e-21 → O(1))
- `whiten_waveform(waveform, ...)` — PSD-based whitening using Welch + bandpass; returns `(whitened, psd_array, freqs)`
- `resample_waveform(waveform, original_delta_t, target_delta_t, ...)` — anti-aliased resampling via PyCBC

---

## Output Structure and Metadata

```python
result = pycbc_data_generator(...)

result['train_loader']   # DataLoader — batches of (X, y)
result['val_loader']     # DataLoader
result['test_loader']    # DataLoader

result['metadata'] = {
    'parameter_names':    list[str],        # column order of y
    'num_samples':        int,              # successful waveforms
    'num_failed':         int,
    'waveform_shape':     (D, T),           # (num_detectors, target_length)
    'channels':           list[str],        # e.g. ['H1', 'L1']
    'train_size':         int,
    'val_size':           int,
    'test_size':          int,
    'batch_size':         int,
    'time_resolution':    float,            # delta_t
    'approximant':        str,
    'f_lower':            float,
    'f_final':            float,
    'highpass_fc':        float,
    'detectors':          list[str],
    'target_length':      int,
    'signal_length':      float,
    'chunk_size':         int,
    'sky_params_provided':dict,
    'add_noise':          bool,
    'preprocessing':      dict,             # empty by default

    # MG only
    'm_g':                float,
    'm_g_varied':         bool,

    # LV only
    'alpha_lv':           float,
    'A':                  float,
    'm_g':                float,
    'm_g_varied':         bool,
    'A_varied':           bool,
    'waveform_type':      'lorentz_violation',
}
```

Tensor shapes:
- `X` — `(N, D, T)` float32, where D = number of detectors, T = `target_length`
- `y` — `(N, P)` float32, z-score normalised, where P = number of parameters in config

A batch from a DataLoader has shape `(batch_size, D, T)` for X and `(batch_size, P)` for y.

---

## Save / Load

```python
from gw_datagen import save_dataloaders, load_dataloaders

# Save
save_dataloaders(result, 'dataset.pt')

# Load — recreates DataLoaders with the original train/val/test split
loaded = load_dataloaders('dataset.pt')
loaded = load_dataloaders('dataset.pt', batch_size=64)  # override batch size
```

**What is saved** (via `torch.save`):
- `X` and `y` tensors (full dataset, not split)
- `train_indices`, `val_indices`, `test_indices` — integer index lists
- `metadata` dict

Loading reconstructs the exact same split using `torch.utils.data.Subset`. The tensor round-trip is exact (`torch.equal` passes).

---

## Testing

Tests live in `Msci-GW/local tests/code tests/`.

| File | Covers |
|------|--------|
| `test_analytic.py` | GR, MG, LV with analytic aLIGO noise; save/load round-trip; plots |
| `test_o4_real.py` | O4 PSD cache validation; GR, MG, LV with real O4 noise; save/load; plots |
| `conftest.py` | Shared fixtures: `small_config`, `base_kwargs`, `aligo_kwargs`, `psd_cache_dir` |

Run with:

```bash
cd "Msci-GW/local tests/code tests"
python -m pytest -v
```

Key fixture defaults (`conftest.py`):
- `num_samples=16`, `num_workers=1` (serial to avoid WSL2 deadlocks), `batch_size=8`
- `train_split=0.7`, `val_split=0.15`
- `psd_cache_dir` fixture auto-skips all O4 tests if the cache is not populated

---

## Key Constants and Defaults

| Constant | Value | Meaning |
|----------|-------|---------|
| `_HIGHPASS_FC` | 35 Hz | FIR highpass cutoff applied after IRFFT |
| `_RINGDOWN_TAPER_LEN` | 128 samples | End-taper length before highpass |
| `_M_OMEGA_PN` | 0.1 | PN breakdown: (m1+m2)·ω = 0.1 (geometric units) |
| `_TAPER_FRACTION` | 0.50 | Phase taper: smooth to zero over top 50% of f_pn |
| `delta_f` (workers) | 1/256 Hz | Hardcoded FD waveform frequency resolution |
| `n_ringdown` | 500 samples | Ringdown window size in time-domain assembly |
| `_O4A_GPS_START` | 1369166418 | O4a run start (2023-05-24 18:00 UTC) |
| `_O4A_GPS_END` | 1389744018 | O4a run end (2024-01-16 16:00 UTC) |
| `_FETCH_DUR` | 256 s | Duration of each GWOSC data fetch |
| `_DEFAULT_N_SEGS` | 100 | Default number of O4 PSD segments to cache |
| `_C` | 2.998×10⁸ m/s | Speed of light |
| `_M_SUN_SEC` | ~4.926×10⁻⁶ s | Solar mass in geometric units (G·M☉/c³) |
| `_H0` | 67.4 km/s/Mpc | Hubble constant |
| `_OMEGA_M` / `_OMEGA_LAMBDA` | 0.315 / 0.685 | Flat ΛCDM cosmological parameters |
