# Pre-full-run updates for the HPC training pipeline

**Context.** The smoke test passed on HPC and a 200 k-sample training is
running now. Before committing GPU-hours to the full sweep, the items
below are the updates that will either (a) give clear throughput wins,
(b) prevent losing work to wall-time kills, or (c) stop a long sweep
from going off the rails silently. Ordered by expected payoff.

---

## Implementation status (L40 48 GB target)

Shipped in this pre-full-run pass:

- **§1 Throughput** — TF32 matmul + cuDNN TF32 + `cudnn.benchmark`, AMP
  (`torch.autocast(dtype=bfloat16)`, no GradScaler), `torch.compile`
  hook with `drop_last_batch` auto-enabled so CUDA graphs don't
  recompile on the tail batch, VRAM max-allocated logged per epoch.
  `GPU_DEFAULT_CONFIG` now ships with `batch_size=256, val_chunk_size=256`
  (see §2 — safe headroom on a 48 GB L40).
- **§5 Observability** — per-run log file via `_Tee` wrapping stdout
  (`cfg['log_file']`), CUDA memory line per epoch, CSV now carries
  `val_loss_history` (pipe-joined) and `log_file`. `hp_search.py`
  auto-creates `logs/` and injects `log_file=logs/{run_id}.log` per
  trial.
- **§6 Evaluation parity** — both `evaluate_model.py` and
  `evaluate_real.py` now call `resolve_crop_for_embedding(ckpt['config'])`
  (honours per-embedding overrides), forward all new flow/config keys
  into `DINGOModel`, and pass `param_parameterization` into
  `load_dataset_pt`. `evaluate_real.py` additionally derives catalogue
  chirp_mass / mass_ratio from the CSV masses when the model is
  Mc/q-trained, and the residual / z-score / metrics loops iterate
  over the active `matched_params` (not a hardcoded `mass1, mass2` list).
- **§9a Neural Spline Flow** — shipped as an HP-searchable axis.
  `coupling_type ∈ {'affine', 'spline'}` on `NormalizingFlow` +
  `DINGOModel`; spline path is a full rational-quadratic Durkan
  implementation (`_rational_quadratic_spline`, `RQSplineCouplingLayer`)
  with linear tails, zero-init on the final layer, and
  `spline_num_bins` / `spline_tail_bound` as additional axes.
- **§9c Mc/q parameterisation** — toggled via
  `param_parameterization ∈ {'m1_m2', 'Mc_q'}`. `load_dataset_pt`
  rewrites the `mass1/mass2` columns to `chirp_mass/mass_ratio` and
  retrains the normaliser off the transformed splits. Evaluators
  render the new labels and handle real-event truth injection.
- **§9d Volume distance prior** — toggled via
  `distance_prior ∈ {'uniform', 'volume'}`. `run_training` builds
  importance weights `w = d² / mean(d²)` on the train split and the
  trainer consumes them via `train_weights`; no dataset regeneration
  needed. GPU trainer honours the same path.
- **Per-embedding crop overrides** — `simple_crop_half_width`,
  `conv1d_crop_half_width`, `lstm_crop_half_width` shipped as HP axes.
  `resolve_crop_for_embedding` picks the override when set, otherwise
  falls back to `merger_crop_half_width`. `hp_search.prune_for_embedding`
  drops crop keys that don't belong to the active embedding so the
  grid doesn't explode on no-op combinations.
- **Checkpoint filename** — `build_checkpoint_path` now tags the
  active crop, coupling type (`_spline`), parameterisation (`_Mcq`),
  and prior (`_volprior`) so checkpoint stems are unambiguous across
  the sweep.

Still pending (tracked here, not shipped):

- §3 periodic backup `*.latest.pt` + resume-from exercise on real GPU.
- §4 per-combo walltime guard + cheap-first grid sort (narrowing the
  grid itself is a user call at sweep launch).
- §9b SVD basis compression — deferred; out of scope for this run.
- §9e on-the-fly noise per batch — deferred.
- §9f cosine LR schedule with warmup — deferred.

---

## 1. Throughput — free speedups on the existing GPU loop

`train_model_gpu.py` already has gradient clipping, AdamW +
ReduceLROnPlateau, checkpoint-on-improvement, and whole-dataset-on-device.
It is **not** yet using the mixed-precision / TF32 / cuDNN knobs.

Add near the top of `run_training()`:

```python
torch.backends.cuda.matmul.allow_tf32 = True    # free on Ampere+
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark  = True          # LSTM conv front-end benefits
```

Add **automatic mixed precision** (AMP) to `train_dingo_model_gpu` — this
is the single biggest GPU win for this model:

```python
scaler = torch.cuda.amp.GradScaler(enabled=dev.type == 'cuda')
...
with torch.cuda.amp.autocast(enabled=dev.type == 'cuda', dtype=torch.bfloat16):
    log_prob = model(batch_params, batch_data)
    loss = -log_prob.mean()
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer); scaler.update()
```

`bfloat16` is safer than `fp16` for the flow (no loss-scaling gotchas).
The validation path in `_chunked_val_logprob` should also wrap in
`autocast` for consistency but stay in `torch.no_grad()`.

Flip `compile_model=True` in the full-run config **after** AMP is in.
The hook is already in `train_model_gpu.run_training` (wraps the model
in `torch.compile(mode='reduce-overhead')` with a try/except fallback),
so no new code is needed — just set the config flag. Notes:

- First-epoch wall-clock will be 30–90 s longer than subsequent ones
  while Inductor traces the graph; do not panic and kill the job.
- `reduce-overhead` uses CUDA graphs under the hood, which means dynamic
  shapes break it. The LSTM embedding's final batch can be smaller than
  `batch_size` (the `if batch_params.shape[0] < 2: continue` already
  drops size-1 batches, but not size-(N mod batch_size) ones). Either
  drop the remainder batch (`indices[:num_samples - num_samples %
  batch_size]`) or switch to `mode='max-autotune'` if you see
  recompilation warnings.
- Compile pairs well with AMP — do them together, not sequentially.
- If `torch.compile` regresses performance (rare but possible with LSTM
  on older torch builds), the fallback path in the config logs
  `torch.compile: disabled` and continues eagerly.

Finally, push `batch_size` from the current 128 toward 256–512 once AMP
is on; watch `nvidia-smi` for memory headroom.

---

## 2. Memory — 200 k samples borderline for whole-dataset upload

Current pattern uploads the entire dataset to the GPU once. For the
cropped dataset (`merger_crop_half_width=500`, T=1000):

| Split     | Size                                  | VRAM  |
|-----------|---------------------------------------|-------|
| train 200 k | 200 000 × 2 × 1000 × 4 B            | 1.6 GB |
| val 25 k    |  25 000 × 2 × 1000 × 4 B            | 0.2 GB |
| test 25 k   | ditto                               | 0.2 GB |
| params (all three)                            | ~30 MB |

So the raw data is ~2 GB on device. On an A100 (40/80 GB) this is fine.
On a V100 (16 GB) with AMP + compile + batch 512 it gets tight.

**If you see OOMs**, switch to a streaming `TensorDataset + DataLoader`
with `pin_memory=True, num_workers=2, persistent_workers=True` for the
train split only, keep val on-device (the chunked path already handles
it). This is a ~30-line change; do it only if needed.

---

## 3. Robustness — survive wall-time kills

The checkpoint-on-improvement already covers the common case. Two gaps
for long HPC jobs:

- **Periodic backup checkpoint.** Add an "every N epochs regardless of
  improvement" save. If a run flat-lines for 10 epochs and then the job
  is killed, the on-improvement checkpoint is 10 epochs stale. Write to
  a separate `*.latest.pt` path so the canonical `best` file is not
  overwritten.

- **Resume has never been exercised.** Before the full sweep, pick a
  smoke checkpoint, bounce the job, and relaunch with `resume_from`.
  Confirm `epochs_completed`, the LR scheduler state, and the early-stop
  `bad_epochs` counter all pick up correctly. If resume is broken,
  you discover it now, not at hour 14 of a 20-hour run.

---

## 4. HP search at scale — 24 combos × 20 epochs × 200 k

At smoke-test scale a combo is seconds. At 200 k it will be
tens of minutes to hours per combo, so the full 24-combo grid is a
multi-day commitment. Two changes before launching the sweep:

1. **Narrow `SEARCH_SPACE`.** Use the 200 k single-combo result as a
   baseline; only sweep the axes that actually affect that baseline's
   failure modes. A typical starting grid for the real sweep:

   ```python
   SEARCH_SPACE = {
       'embedding_type':   ['conv1d', 'lstm'],   # drop 'simple' once it loses on 200 k
       'context_dim':      [128],
       'hidden_dim':       [64, 128],
       'num_flow_layers':  [4, 6],
       'learning_rate':    [1e-4, 3e-4],
       'batch_size':       [256],
   }
   ```

   That is 8 combos, not 24.

2. **Per-combo walltime guard.** `hp_search.py` catches exceptions but
   not "this combo has been running for 4 hours and is still at epoch 3".
   Add a `max_seconds_per_run` kwarg to `COMMON_CONFIG` and have
   `train_dingo_model_gpu` check elapsed time after each epoch. Record
   the truncation in the CSV `status` column as `timeout`.

3. **Sort combos cheap-first.** Iterate the grid from smallest model to
   largest so the first few CSV rows land fast and you can abort early
   if something is obviously wrong.

---

## 5. Observability — you will look at this sweep offline

- **Tee stdout to a log file per run.** `hp_search.py` currently only
  relies on the PBS stdout redirect. Add
  `tee -a logs/{run_id}.log` at the trainer level, or open a file
  inside `run_training` and mirror print statements.
- **Log CUDA memory after each epoch.** One line:
  `torch.cuda.max_memory_allocated() / 1e9` tells you whether you could
  have pushed batch size higher.
- **Expand the CSV.** Add `val_loss_history` as a pipe-joined string so
  you can plot loss curves from the CSV alone without re-opening each
  checkpoint.

---

## 6. Evaluation parity — don't forget the crop

`evaluate_model.py` and `evaluate_real.py` both already read
`merger_crop_half_width` off the checkpoint's config dict and apply
`crop_to_merger` before inference. After the full run:

- Verify the two scripts run cleanly against one of the 200 k
  checkpoints before launching the real-data eval sweep.
- For real-event eval, confirm the crop is applied to the O4 strain in
  the same way — the sanity check is that `data.shape[-1] == 2 *
  merger_crop_half_width` inside the evaluator.

---

## 7. PBS / scheduler

- **Walltime** — a 200 k × 20-epoch run should take ~2–6 hours on a
  single A100 depending on AMP/compile. Set PBS walltime to 2× that.
- **GPU count** — single-GPU for now; multi-GPU needs DDP and is out of
  scope for this run.
- **Memory** — request at least 32 GB RAM per job (dataset load + Python
  overhead); the GPU memory is separate.
- **Job array for the HP sweep.** If your PBS queue supports it, emit
  one array task per combo (`hp_search.py --max-runs 1 --skip $i`)
  rather than one monolithic job — kills are per-combo, not per-sweep.

---

## 8. Numerical sanity before committing

Before queuing the big sweep, run **one** combo end-to-end on the full
dataset with the updates above applied, and confirm:

- Loss decreases monotonically for the first 3 epochs (no NaN/Inf).
- Per-epoch wall-clock matches your walltime estimate within ±20 %.
- `val_chunk_size` fits in GPU memory at peak — if not, halve it.
- The best checkpoint loads back into `evaluate_model.py` and produces
  sane posterior samples on the test split.

If all four pass, the sweep is safe to launch.

---

## 9. Alignment with DINGO — where this pipeline diverges from the paper

Dax et al. 2021 (*Real-time gravitational-wave inference with DINGO*)
and the public `dingo-gw` codebase do several things that the current
pipeline does not. The ones that are actually worth porting before a
long HPC run, in order of expected accuracy payoff:

### a. Neural Spline Flow instead of affine coupling (big accuracy win)
Current flow = stacked `AffineCouplingLayer` with a scale/translation
MLP. DINGO uses **rational-quadratic neural spline coupling** (RQ-NSF,
Durkan et al. 2019) which has roughly an order of magnitude more
expressive power per coupling layer. On a 13-dim posterior with the
mass / distance degeneracies, affine couplings will consistently
under-fit the tails even at `num_flow_layers=6`.

  - Effort: medium — drop in `nflows` or `zuko`, swap
    `AffineCouplingLayer` for their `PiecewiseRationalQuadraticCoupling`,
    keep the same `context_dim` interface. Flow-level API in
    `NormalizingFlow` stays unchanged.
  - Do this **before** spending GPU-hours on the full HP sweep — tuning
    affine flows further is a dead end if you're going to swap them.

### b. Frequency-domain strain + SVD basis compression (big efficiency win)
DINGO does not feed time-domain strain into the embedding. It (i)
whitens in frequency domain, (ii) projects onto a **reduced SVD basis**
learned from a bank of simulated signals (typically 200–400 basis
coefficients per detector), then (iii) feeds the coefficients into the
embedding. The embedding is then a plain MLP on ~800 inputs, not a
conv/LSTM on 8000+ samples.

  - Why it matters: 10× fewer input features → 10× smaller / faster
    embedding, and the basis projection is a physics-informed prior
    that regularises against noise realisations.
  - Effort: high — requires generating the SVD basis from a separate
    simulation sweep and adding an FD → basis stage upstream of the
    embedding. Not for this run. **But worth knowing** that every
    time-domain architecture you HP-search is fighting a handicap.

### c. Parameterisation: chirp mass + mass ratio, not `(m1, m2)`
Current `generate_dataset.py` samples `mass1` and `mass2` independently
uniform in [5, 90]. The physical posterior is almost always sharper in
`(chirp_mass, mass_ratio) = (Mc, q)` than in `(m1, m2)`, and the flow
training is much faster to converge in that basis because the level
sets align with the data.

  - Fix: add Mc, q computation inside `generate_dataset.py` (or in
    `load_dataset_pt`) and train the flow against those instead of
    `m1, m2`. At inference time, transform back. The label dim stays
    the same.
  - Effort: low (~30 lines). Measurable gain on coverage of the mass
    posterior.

### d. Distance prior: uniform in comoving volume, not uniform in Mpc
Current code samples `distance ~ U(100, 5000) Mpc`. The physical prior
for BBH mergers is **uniform in comoving volume** (∝ D² dD in the
Euclidean limit). Training against the wrong prior biases the flow
towards flat distance posteriors that will not match PE from proper
Bayesian samplers on real events.

  - Fix: in the `CONFIG` dict, replace the uniform `distance` sampler
    with `D ∝ (U)^(1/3) × D_max` (uniform in volume). Also check
    `gw_datagen` does not internally re-weight.
  - Effort: low. Important if your `evaluate_real.py` comparison is
    against published PE posteriors.

### e. On-the-fly noise per batch (instead of baked into the dataset)
`ADD_NOISE = True` in `generate_dataset.py` draws one noise realisation
per sample at dataset-generation time and freezes it. DINGO redraws
noise from the PSD every epoch, giving the flow an effectively infinite
training set for the noise component. With 200 k frozen noise draws
over 20 epochs, the flow will start memorising specific noise
realisations late in training — the `bad_epochs` counter may not catch
this because val shares the same frozen-noise bias.

  - Fix: store the **clean** strain only (the `be06ec3 saving only
    processed waveforms` commit already halved storage this way); draw
    a fresh O4 PSD noise realisation on GPU at batch-load time. This
    is ~20 lines on top of `train_dingo_model_gpu`.
  - Effort: medium. Pairs nicely with AMP — the noise draw is a
    single `torch.randn(...) * sqrt(psd)` kernel.

### f. Cosine LR schedule with warmup
DINGO uses cosine-annealing with linear warmup, not `ReduceLROnPlateau`.
Plateau is defensive (reacts to val stalling) but conservative;
cosine usually converges to a better minimum on big runs where you
trust the training recipe. Worth trying once you have a stable baseline.
  - Effort: low — two-line scheduler swap. Low priority for *this* run.

### What to skip for now
- **GNPE / group equivariance** — big architectural change, not worth
  it for a first full run.
- **5 M+ training samples** — DINGO typically trains on 5 M waveforms;
  200 k–1 M is fine for this project's scale.
- **Full DINGO waveform bank (multiple approximants, full spin
  precession)** — your `IMRPhenomD` aligned-spin setup is adequate for
  comparing embedding architectures; upgrade later.

---

## Update priority (if you only have time for some)

1. **Swap affine coupling for neural spline flow** (§9a) — single largest
   accuracy win; anything else is polish on top of an under-powered flow.
2. **AMP + TF32 + cuDNN benchmark** (§1) — biggest speedup, ~1 h of work.
3. **Chirp-mass / mass-ratio reparameterisation** (§9c) — low effort,
   real coverage gain on the mass posterior.
4. **Distance ∝ volume prior** (§9d) — low effort, real parity with
   Bayesian PE on real events.
5. **Exercise `resume_from`** (§3) — cheap insurance.
6. **Narrow `SEARCH_SPACE` and sort cheap-first** (§4).
7. **Per-run log file + CUDA memory logging** (§5).
8. **On-the-fly noise per batch** (§9e) — moderate effort, pays back
   on longer (>20-epoch) runs.
9. Everything else (SVD basis, cosine schedule, GNPE) is next-phase work.
