# Local Test Suite — `gw_datagen`

Tests live in `local tests/code tests/`. The `physics tests/` directory is currently empty.

---

## Directory layout

```
local tests/
├── TESTS.md                   ← this file
├── code tests/
│   ├── conftest.py            ← shared pytest fixtures
│   ├── pytest.ini             ← pytest configuration and markers
│   ├── test_analytic.py       ← Batch 1: synthetic (aLIGO) PSD tests
│   ├── test_o4_real.py        ← Batch 2: real O4 GWOSC PSD tests
│   ├── check_parallel.py      ← standalone parallel-generation smoke test
│   └── plots/                 ← diagnostic plots written by TestPlots
└── physics tests/             ← (empty, reserved for future use)
```

---

## Prerequisites

### Python path

`conftest.py` automatically inserts `Msci-GW/HPC/Pipeline/Data Generation/` into
`sys.path`, so `from gw_datagen import ...` works without any manual path setup.

### O4 PSD cache (Batch 2 only)

`test_o4_real.py` requires a pre-downloaded cache of real GWOSC O4 PSDs.
Run **once** from the `Data Generation/` directory before running those tests:

```bash
python download_o4_psds.py --detectors H1 L1 --n-segments 100
```

If the cache is absent, every test in `test_o4_real.py` is **automatically skipped** —
no failures, just skips.

---

## Running the tests

All commands should be run from `local tests/code tests/`.

### Run everything

```bash
pytest
```

### Run only Batch 1 (no internet / no cache needed)

```bash
pytest test_analytic.py -v
```

### Run only Batch 2 (requires O4 PSD cache)

```bash
pytest test_o4_real.py -v
```

### Run by marker

```bash
pytest -m aligo    # aLIGO synthetic PSD tests
pytest -m o4psd    # real O4 PSD tests
```

### Run the parallel smoke test

`check_parallel.py` is a plain Python script (not collected by pytest). It tests
all three generators with multi-process waveform generation:

```bash
# Default: 4 workers
python check_parallel.py

# Custom worker count
python check_parallel.py 8
```

---

## Test files

### `conftest.py` — shared fixtures

| Fixture | Scope | Description |
|---|---|---|
| `small_config` | session | Minimal BBH parameter space (masses 20–40 M☉, distances 200–500 Mpc, zero spins, random redshift 0.05–0.15). All entries are lambdas as required by `_validate_config`. |
| `base_kwargs` | session | Fast generation settings: 16 samples, 4096 Hz, `IMRPhenomD`, 2 s signal, `num_workers=1` (avoids multiprocessing deadlocks in pytest on WSL2/Linux). `f_final` is **excluded** here. |
| `aligo_kwargs` | session | `base_kwargs` + `f_final=2048.0`. Used by all standard generators. |
| `psd_cache_dir` | session | Points to the fixed O4 PSD cache directory. **Skips the whole test session** if `H1` cache file is missing. |

---

### `test_analytic.py` — Batch 1: synthetic PSDs

No internet access required. All three generators use the `aLIGOZeroDetHighPower` PSD.

#### Session fixtures

| Fixture | Generator | Physics parameters |
|---|---|---|
| `gr` | `pycbc_data_generator` | Standard GR |
| `mg` | `pycbc_massive_gravity_data_generator` | `lambda_g = 1e22 m` |
| `lv` | `pycbc_lorentz_violation_data_generator` | `lambda_g = 1e22 m`, `alpha_lv = 3.0`, `A_lv = 1e15` |

Each dataset is generated **once per test session** via session-scoped fixtures.

#### Test classes

**`TestGenerators`** — output contract for all three generators

| Test | What it checks |
|---|---|
| `test_gr_keys` | Return dict has exactly `{train_loader, val_loader, test_loader, metadata}` |
| `test_gr_splits` | Split sizes sum to `num_samples`; each split is non-empty |
| `test_gr_batch_shape` | Tensor shape is `(batch_size=8, 2, waveform_samples)` for X; label width matches `parameter_names` |
| `test_gr_finite_and_nonzero` | No NaN/Inf; output is not all zeros |
| `test_mg_lambda_g` | Metadata records the correct `lambda_g` |
| `test_mg_finite_and_nonzero` | Same numerical health check for MG |
| `test_lv_alpha_lv` | Metadata records the correct `alpha_lv` |
| `test_lv_finite_and_nonzero` | Same numerical health check for LV |

**`TestSaveLoad`** — `save_dataloaders` / `load_dataloaders` round-trip on GR data

| Test | What it checks |
|---|---|
| `test_file_exists` | `.pt` file is written to disk |
| `test_keys_survive` | Loaded dict has all four keys |
| `test_metadata_survives` | Key metadata fields are identical after reload |
| `test_tensor_exact_roundtrip` | Tensor data is **bit-for-bit identical** (`torch.equal`, not `allclose` — strain ~1e-22 is within `allclose`'s default tolerance) |

**`TestPlots`** — diagnostic visualisation

Generates `plots/analytic_generators.png` showing H1 strain for one sample from each
generator. No assertions on plot content; the test passes as long as matplotlib does
not error.

---

### `test_o4_real.py` — Batch 2: real O4 GWOSC PSDs

Requires the pre-downloaded O4 PSD cache. All tests skip automatically if the cache
is missing. Generators run with `detectors=["H1"]` (single detector) to match the
cache fixture.

#### Session fixtures

| Fixture | Generator | Noise backend |
|---|---|---|
| `gr_o4` | `pycbc_data_generator` | `o4_psd` |
| `mg_o4` | `pycbc_massive_gravity_data_generator` | `o4_psd` |
| `lv_o4` | `pycbc_lorentz_violation_data_generator` | `o4_psd` |
| `saved_path_gr/mg/lv` | — | Paths to saved `.pt` files for each generator |

#### Test classes

**`TestCache`** — validates the downloaded `.npz` file

| Test | What it checks |
|---|---|
| `test_file_exists` | Cache file is present on disk |
| `test_keys_and_shape` | Contains `freqs` and `psds` arrays; shapes are consistent |
| `test_freqs_positive_sorted` | Frequency axis is strictly positive and monotonically increasing |
| `test_psds_positive` | All PSD values are positive |
| `test_psds_distinct` | At least two PSD segments differ (uses `np.array_equal`, not `allclose`) |

**`TestPsdLoading`** — verifies `load_random_o4_psd` interpolation

| Test | What it checks |
|---|---|
| `test_returns_frequency_series` | Returns a PyCBC `FrequencySeries` |
| `test_length_and_delta_f` | Output length equals `FLEN=4097`; `delta_f` matches `0.5 Hz` |
| `test_positive_and_finite` | All interpolated PSD values are positive and finite |

**`TestNoise`** — verifies `noise_from_psd`

| Test | What it checks |
|---|---|
| `test_shape_finite_nonzero` | Noise array has correct length (`8192` samples), is finite, and is non-zero |
| `test_different_seeds_differ` | Two draws with different seeds produce different arrays (uses `array_equal`, not `allclose`) |

**`TestSaveLoad`** — round-trip for all three O4 generators

Same checks as Batch 1 `TestSaveLoad`, applied independently to GR, MG, and LV.

**`TestGeneratorEndToEnd`** — numerical health for all three O4 generators

Checks that every batch in the train loader is finite, and that the first batch is
non-zero.

**`TestPlots`** — saves `plots/o4_real_waveforms.png`

Side-by-side H1 strain comparison for GR, MG, and LV with O4 noise.

---

### `check_parallel.py` — parallel smoke test

A self-contained script (not a pytest file) that directly exercises all three
generators with `num_workers > 1`. Run it to verify that multiprocessing works
correctly on your machine.

**Configuration constants** (edit at the top of the file):

| Name | Default | Description |
|---|---|---|
| `KWARGS['num_workers']` | CLI arg (default 4) | Worker processes for generation |
| `KWARGS['num_samples']` | 16 | Total waveforms to generate |
| `KWARGS['batch_size']` | 8 | DataLoader batch size |

**Exit codes:** `0` = all cases passed, `1` = at least one failed.

> **Note on `num_workers` in pytest:** `conftest.py` sets `num_workers=1` in
> `base_kwargs` to avoid multiprocessing deadlocks inside pytest on WSL2/Linux.
> `check_parallel.py` exists specifically to test multi-worker generation outside
> the pytest process.

---

## Diagnostic plots

Both test files write plots to `code tests/plots/` (created automatically):

| File | Contents |
|---|---|
| `analytic_generators.png` | H1 strain — GR, MG, LV with synthetic aLIGO noise |
| `o4_real_waveforms.png` | H1 strain — GR, MG, LV with real O4 GWOSC noise |

Plots are for visual inspection only; they do not gate test pass/fail.

---

## Common failure modes

| Symptom | Likely cause |
|---|---|
| All `test_o4_real.py` tests **skipped** | O4 PSD cache not populated — run `download_o4_psds.py` |
| `ImportError: cannot import name 'gw_datagen'` | `conftest.py` path resolution failed — check that `HPC/Pipeline/Data Generation/gw_datagen.py` exists |
| `check_parallel.py` hangs or deadlocks | Multiprocessing issue on WSL2 — try `python check_parallel.py 1` to confirm the generators work, then increase workers |
| `test_tensor_exact_roundtrip` fails | Save/load introduced floating-point error — the test intentionally uses `torch.equal` (exact) rather than `allclose` because strain values (~1e-22) are within `allclose`'s default tolerance |
