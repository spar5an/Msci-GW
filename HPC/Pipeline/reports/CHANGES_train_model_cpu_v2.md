# Changes to `train_model_cpu.py` — v2

Follow-up to [REVIEW_train_model_cpu_v2.md](REVIEW_train_model_cpu_v2.md) and [CHANGES_train_model_cpu.md](CHANGES_train_model_cpu.md). Implements REVIEW-v2 items 2.1 (scheduler), 2.3 (AdamW), 2.4 (dropout), 4.2 (resume/checkpoint state), 4.3 (per-detector vs shared weights option), 4.5 (filename tag). Also bulks the dataset to 10 000 waveforms.

---

## 1. Dataset: 100 → 10 000 waveforms

[Data Generation/generate_dataset.py:31](Data Generation/generate_dataset.py#L31) now sets `NUM_SAMPLES = 10000`. Everything else in the generator is unchanged — noise + waveforms are still computed at generation time and baked into `dataset.pt`, per the user's instruction. The new split is 8000 / 1000 / 1000 (train / val / test).

The dataset now lives at [Data/dataset.pt](Data/dataset.pt) as expected by `load_dataset_pt` ([train_model_cpu.py:345](train_model_cpu.py#L345)).

---

## 2. Training script changes

### 2.1 `CosineAnnealingLR` → `ReduceLROnPlateau`  *(REVIEW v2 §2.1)*

- [train_model_cpu.py:441-443](train_model_cpu.py#L441-L443) now builds `ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=lr*0.01)`.
- `scheduler.step(current_metric)` is called with the epoch's validation log-prob (falling back to the train log-prob when no val set is supplied) — [train_model_cpu.py:519-521](train_model_cpu.py#L519-L521).
- The `mode='max'` flag is important: we are tracking *log-prob*, which we want to **maximise**; `factor=0.5` halves the LR on each plateau and `min_lr` matches the old `eta_min` floor.

Motivation: the v1 cosine schedule dropped the LR to 1e-6 on a fixed 20-epoch budget regardless of whether training had converged. `ReduceLROnPlateau` only drops the LR after `patience=3` consecutive non-improving epochs, so the full learning-rate budget is spent only when progress actually stalls.

### 2.3 `Adam` → `AdamW(weight_decay=1e-4)`  *(REVIEW v2 §2.3)*

- [train_model_cpu.py:440](train_model_cpu.py#L440) swaps `torch.optim.Adam(...)` for `torch.optim.AdamW(..., lr=lr, weight_decay=weight_decay)`.
- `weight_decay` is exposed as a function argument on `train_dingo_model`, with default `1e-4`. Main reads it from the new `WEIGHT_DECAY = 1e-4` config constant ([train_model_cpu.py:554](train_model_cpu.py#L554)).
- The decay is written to the checkpoint config for reproducibility.

### 2.4 Dropout in `SimpleEmbeddingNetwork`  *(REVIEW v2 §2.4)*

[train_model_cpu.py:153-184](train_model_cpu.py#L153-L184):

- `SimpleEmbeddingNetwork.__init__` takes a new `dropout=0.1` argument.
- A `nn.Dropout(dropout)` sits at the end of the per-channel MLP (after the third `ReLU`), and a second one sits inside `merge` between the two final `Linear` layers.
- Dropout probability is plumbed from `DINGOModel(embedding_dropout=0.1)` → `SimpleEmbeddingNetwork(dropout=...)`.

Motivation: the last run's remaining 4.5-nat train/val gap pointed to under-regularised feature extraction. The Conv1D and LSTM embeddings already had dropout — this brings the simple MLP in line.

### 4.3 `share_detector_weights` option  *(REVIEW v2 §4.3)*

`SimpleEmbeddingNetwork` now supports both a **shared per-detector MLP** (default, cheaper, permutation-equivariant between H1 and L1) and **independent per-detector MLPs** (slightly more parameters, lets the network learn detector-specific features that differ between H1 and L1 antenna response or noise):

- `SimpleEmbeddingNetwork.__init__(..., share_detector_weights=True)` ([train_model_cpu.py:153-154](train_model_cpu.py#L153-L154)).
- Shared branch: `self.per_channel = make_per_channel()` applied as `(N*D, T) → (N*D, h)` and reshaped.
- Independent branch: `self.per_channel = nn.ModuleList([make_per_channel() for _ in range(num_detectors)])`, and `forward` loops over each detector through its own MLP.
- The switch is surfaced on `DINGOModel(share_detector_weights=True)` ([train_model_cpu.py:294-296](train_model_cpu.py#L294-L296)).

Kept the default at `True` — for whitened strain the two detectors look statistically similar, so sharing weights halves parameters and trains faster. Flip to `False` for raw data or when experimenting with detector-specific PSDs.

### 4.2 Checkpoint now saves optimizer / scheduler state (+ resume path)  *(REVIEW v2 §4.2)*

Three changes stitched together:

1. `train_dingo_model` accepts `optimizer_state_dict`, `scheduler_state_dict`, `start_epoch`, `best_log_prob_init`, `best_state_init`, `best_epoch_init`, `bad_epochs_init` ([train_model_cpu.py:410-469](train_model_cpu.py#L410-L469)). When the optimizer/scheduler state dicts are provided they are `load_state_dict`-ed onto freshly built instances. The epoch loop now runs over `range(start_epoch, num_epochs)` ([train_model_cpu.py:471](train_model_cpu.py#L471)).
2. `train_dingo_model` returns `(losses, val_losses, best_state, best_log_prob, best_epoch, optimizer, scheduler, bad_epochs)` — the extra trio is what `__main__` needs to write back into the checkpoint.
3. The checkpoint dict saved at the end of `__main__` now includes `optimizer_state_dict`, `scheduler_state_dict`, `bad_epochs`, `epochs_completed`, and a duplicate `best_state` key (same as `model_state_dict`) for clarity ([train_model_cpu.py:633-664](train_model_cpu.py#L633-L664)).

Resume is opt-in via a `RESUME_FROM = None` constant at the top of `__main__` ([train_model_cpu.py:556](train_model_cpu.py#L556)). Setting it to a checkpoint path reads all seven fields back, `load_state_dict`s them, and picks up at `epochs_completed`.

### 4.5 `noisy_whitened` filename tag  *(REVIEW v2 §4.5)*

[train_model_cpu.py:579-589](train_model_cpu.py#L579-L589) replaces the old two-tag `_{noise_str}{whiten_str}` construction with a single `signal_tag` chosen by what the saved data actually looks like:

- `WHITEN=True` → tag is `whitened` (the `noisy` prefix is dropped because whitening flattens most of the coloured noise anyway).
- `WHITEN=False, ADD_NOISE=True` → tag is `noisy`.
- `WHITEN=False, ADD_NOISE=False` → tag is `clean`.

So the filename now reads e.g. `dingo_N8k_F4_C128_H64_E20_simple_whitened_cpu.pt` instead of the old contradictory `..._noisy_whitened_cpu.pt`.

---

## 3. Training run on the 10 000-sample dataset

Run config: 8000 train / 1000 val / 1000 test, 20 epochs, batch 32, lr 1e-4, weight_decay 1e-4, simple embedding (shared per-detector, dropout 0.1). Total parameters: **1 296 616**. Output checkpoint: [dingo_N8k_F4_C128_H64_E20_simple_whitened_cpu.pt](dingo_N8k_F4_C128_H64_E20_simple_whitened_cpu.pt).

| Metric | v1 (100 samples, cosine) | v2 (10 000 samples, AdamW + plateau) |
|---|---:|---:|
| Best val log-prob | −16.55 (ep 20) | **−5.69 (ep 20)** |
| Final-epoch val log-prob | −16.55 | **−5.69** |
| Train–val gap at best | ~4.5 nats | ~3.2 nats |
| LR at end of run | 1e-6 (collapsed) | **1e-4 (still full)** |

Epoch-by-epoch (abridged):

```
Epoch   1/20, Train: -12.4926, Val: -10.5686, LR: 1.00e-04 *
Epoch   5/20, Train:  -7.1746, Val:  -7.3055, LR: 1.00e-04 *
Epoch  10/20, Train:  -5.0861, Val:  -6.2874, LR: 1.00e-04 *
Epoch  15/20, Train:  -3.5968, Val:  -6.0800, LR: 1.00e-04
Epoch  17/20, Train:  -3.1523, Val:  -5.7002, LR: 1.00e-04 *
Epoch  20/20, Train:  -2.4687, Val:  -5.6933, LR: 1.00e-04 *
```

Three observations:

1. **The scheduler stayed at 1e-4 the whole run** — val improved every 1–3 epochs, so the `patience=3` non-improvement streak required for a `ReduceLROnPlateau` step never triggered. This is the intended behaviour (§2.1); the old cosine schedule would have been at ~1e-6 by now.
2. **Val is still improving at epoch 20** (ep 17 was −5.70, ep 20 is −5.69). As with the v1 run this suggests a longer `NUM_EPOCHS` budget is the obvious next knob — now safe to push to 50–100 epochs because the LR will only collapse when progress actually stalls.
3. **Gap narrowed from ~4.5 nats to ~3.2 nats** (train −2.47, val −5.69) — AdamW weight decay + the added dropout are doing something, but 3 nats of gap on a 1.3 M-param model trained on 8 k samples is still meaningful and is probably where RQ-spline couplings or a bigger dataset would bite next.

---

## 4. Checkpoint compatibility

Checkpoints written by v1 `train_model_cpu.py` **will not load** into v2 without a resume-time shim, because:

- The `SimpleEmbeddingNetwork` state dict now contains `Dropout` params (no weights, but affects key ordering) — compatible as long as the same `embedding_dropout`/`share_detector_weights` are used.
- New checkpoint keys (`optimizer_state_dict`, `scheduler_state_dict`, `bad_epochs`, `epochs_completed`) are optional on resume — `.get(key)` with sensible defaults. So resuming from a v1 checkpoint's `model_state_dict` still works; the optimizer/scheduler just start fresh.

If [plot_results.py](plot_results.py) is used downstream, it still needs the v1 patch called out in REVIEW v2 §4.1 (constructor args `num_detectors`/`seq_len`, drop `time_delay_value`).
