# 1M data-generation slowdown — findings & fixes

Context: scaling the proven [gen_gr_200k.pbs](../../PBS%20files/gen_gr_200k.pbs) recipe to 1M waveforms via a 5×200k chunked [gen_gr_1M.pbs](../../PBS%20files/gen_gr_1M.pbs) on the `v1_medium24` queue (32 CPU / 450 GB / 24 h). First submission was job **2466219.pbs-7**, killed after 3h29m with only one chunk of 200k saved.

## Observed symptoms

1. Per-batch waveform generation was **~30 % slower** than the 200k run despite doubling `num_workers` (16 → 32).
2. Speed **collapsed mid-chunk** on chunk 1 — from ~53 it/s to 14–20 it/s.
3. After chunk 1 finished generating, the job spent **>1 h in `_whiten_batch` alone** (single-core, 400k serial calls) before being killed.

### Timing evidence

Per-batch speeds (10k inner batches) from [2459226.pbs-7.OU](../../hpc_outputs/2459226.pbs-7.OU) vs [2466219.pbs-7.OU](../../hpc_outputs/2466219.pbs-7.OU):

| Run | Workers | Early batches | Late batches |
|-----|---------|---------------|--------------|
| 200k (2459226) | 16 | 76 it/s | 63–80 it/s (stable) |
| 1M (2466219) chunk 0 | 32 | 53 it/s | 52–55 it/s |
| 1M (2466219) chunk 1 | 32 | 53 it/s | **14–20 it/s** |

Whitening timings (from log tail):
- 200k run total walltime: **1:06:26** (= ~45 min generation + ~20 min whitening + save).
- 1M run: chunk 0 whitened in ~20 min, chunk 1 whitening hung for >1 h before kill.

## Root causes

### 1. BLAS thread oversubscription — explains the 76 → 53 it/s steady-state drop

`pycbc_data_generator` fans out `num_workers=32` via `multiprocessing.Pool`. Each worker is a separate Python process, but inside each worker numpy/scipy/pycbc still default to *all-cores* BLAS (OpenBLAS/MKL). With 32 worker processes × ~32 BLAS threads each, ~1000 threads compete for the 32 physical cores allocated by PBS → cache thrashing and context-switch overhead.

The 200k job had the same BLAS default but only 16 workers, so contention was milder.

### 2. Fork() copy-on-write bloat across chunks — explains the chunk-1 collapse

Python's allocator does not release freed heap back to the OS between chunks. After chunk 0 completes:
- Parent RSS has grown (generation tensors, whitening scratch).
- `del` frees Python references but the arena stays mapped.

When `pycbc_data_generator` forks 32 workers for chunk 1, each child inherits the parent's memory map via COW. Any write by a worker into an inherited page triggers a page copy. With 32 workers all actively writing, we get bursty memory-bandwidth saturation — progressive slowdown as more pages get copied.

RSS at kill time: ~116 GB (well under the 450 GB cap, but still enough to make COW duplication expensive).

### 3. Serial whitening — dominates once generation finishes

`_whiten_batch` in [gw_datagen.py](Data%20Generation/gw_datagen.py) was a plain double `for` loop over `N × D` pairs (400 000 for a 200k chunk), calling pycbc-based `whiten_waveform` one waveform at a time on one core. The other 31 allocated cores sat idle.

Per-call cost is ~3 ms in clean conditions but inflates several-fold under memory pressure, which is why chunk 0 whitened in ~20 min but chunk 1 was still running after 60+ min.

Whitening a full 1M dataset this way would cost **~100 min best case, several hours under pressure**, which dominates the end-to-end runtime.

## Fixes applied

### Fix A — thread caps in the PBS script

[gen_gr_1M.pbs](../../PBS%20files/gen_gr_1M.pbs) now exports before `python`:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export WHITEN_WORKERS=32
```

Each multiprocessing worker is now genuinely single-threaded. 32 processes × 1 BLAS thread = 32 threads on 32 cores, no oversubscription.

### Fix B — parallel `_whiten_batch`

[gw_datagen.py](Data%20Generation/gw_datagen.py) `_whiten_batch` now:
- Flattens the (N, D, T) input into (N·D, T).
- Uses `multiprocessing.get_context('fork').Pool` with an `initializer` that stashes the input array and `delta_t` in module-level state. Fork on Linux shares the input via COW, so workers read without duplication.
- Distributes the N·D indices with `pool.imap_unordered(chunksize≈N·D/(num_workers·8))`.
- Falls back to the original serial loop when `num_workers<=1` or batch is tiny.
- Reads worker count from `WHITEN_WORKERS` env var (default `cpu_count-1`).

Expected: 20 min → ~1 min per 200k chunk on 32 cores.

### Unchanged — the chunked pipeline itself

Still 5×200k sequential chunks with a pre-allocated `(1_000_000, 2, 8192)` accumulator in the stitch phase. Chunked approach is still needed: one-shot 1M would peak at ~500 GB RAM even on the medium24 node.

## Validation

Smoke test [test_whiten_parallel.pbs](../../PBS%20files/test_whiten_parallel.pbs) submitted as job **2467974.pbs-7** on `v1_small24` (8 CPU, 32 GB, 30 min). It:
- Checks parity: serial vs 4-worker outputs must match to `1e-10` absolute.
- Times serial vs 8-worker on a 2000-sample batch and extrapolates to the 400k-call chunk case.

Not resubmitting the 1M generation until the smoke test passes. Output will be at [hpc_outputs/2467974.pbs-7.OU](../../hpc_outputs/).

## Backup

Snapshot of the Data Generation tree as it stood *after* the parallel-whitening edit is at [Data Generation_backup_20260419/](Data%20Generation_backup_20260419/). Revert by swapping directories if a regression surfaces.

## Open follow-ups (not yet done)

- No explicit `gc.collect()` between chunks. Worth adding if, after the thread-cap fix, the chunk-to-chunk slowdown still shows up in the next run.
- `_whiten_batch` currently pickles 8 KB output arrays per index back to the main process (~25 GB of pickle traffic for a full 200k chunk). If this becomes the new bottleneck, switch to `multiprocessing.shared_memory` for the output buffer — workers would write directly, zero pickle cost.
- Long-term: move whitening inside `pycbc_data_generator`'s per-batch loop so raw waveforms never live as a single 200k-sample tensor.
