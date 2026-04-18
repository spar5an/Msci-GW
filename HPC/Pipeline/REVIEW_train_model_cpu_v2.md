# Review v2: `train_model_cpu.py` (post-refactor)

Follow-up to [REVIEW_train_model_cpu.md](REVIEW_train_model_cpu.md) and [CHANGES_train_model_cpu.md](CHANGES_train_model_cpu.md). This pass looks at the **current** script and flags what's still worth doing, with a mix of leftover v1 items and new observations from the latest training run.

---

## 1. What the new run tells us

Last run on 100 synthetic waveforms (20 epochs, batch 32, lr 1e-4, simple embedding):

| Epoch | Train log-prob | Val log-prob | LR |
|------:|---------------:|-------------:|---:|
|  1    | −21.24         | −22.41       | 9.94e-5 |
|  5    | −16.68         | −18.93       | 8.55e-5 |
| 10    | −13.95         | −17.19       | 5.05e-5 |
| 15    | −12.38         | −16.63       | 1.55e-5 |
| 20    | −12.00         | **−16.55**   | 1.00e-6 |

Three takeaways:

1. **Val was still improving at epoch 20**, but the cosine schedule had already dropped the LR to `1e-6`. Compute was being wasted — the model was no longer learning *because the step size had collapsed*, not because it had converged. This is a pure LR-schedule issue now that LayerNorm + no `reg_loss` removed the acute overfit.
2. **Train–val gap is still ~4.5 nats** (−12.0 vs −16.6). Acute overfit is gone, but there's still a generalisation gap. 1.3M parameters vs 80 samples is still a ~16 000:1 ratio.
3. **Early stopping never triggered** — every epoch was a new best. The `patience=10` logic works but didn't get exercised. That also means the "saved best" and "saved last" are the same model; we don't yet know whether early stopping is actually doing anything useful.

---

## 2. Accuracy improvements

### 2.1 LR-schedule mismatch  *(new, high impact)*

[train_model_cpu.py:424-426](train_model_cpu.py#L424-L426) uses `CosineAnnealingLR(T_max=num_epochs, eta_min=lr*0.01)`, which assumes you want the LR to fall smoothly from `lr` to `0.01·lr` over the full training budget. On the latest run the LR fell to 1e-6 before the model had converged — so the final 5–10 epochs made tiny updates even though validation was still going down.

Two reasonable fixes:

- **Switch to `ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)`** keyed on `avg_val_log_prob`. The schedule only drops the LR when val stops improving, which is what we actually want.
- **Or: longer cosine cycles** — run cosine with `T_max=5` and warm-restart. Less adaptive but deterministic.

Pair either option with a longer `NUM_EPOCHS` (say 100) and the existing early-stopping guardrail.

### 2.2 Generate more data + noise augmentation  *(carried over from v1)*

Still the single biggest remaining lever. The fact that val is still improving at epoch 20 with a 4.5-nat gap to train is a strong signal we're data-limited, not optimisation-limited. Concrete actions:

- Raise `NUM_SAMPLES` in [Data Generation/generate_dataset.py:31](Data Generation/generate_dataset.py#L31) from 100 to at least 10 000.
- Add per-epoch noise re-draw — draw fresh PSD noise inside the training loop rather than baking one realisation into `dataset.pt`. This multiplies effective dataset size by `NUM_EPOCHS`.

### 2.3 `Adam` → `AdamW` with weight decay  *(carried over from v1)*

Still a near-free bump. [train_model_cpu.py:423](train_model_cpu.py#L423) uses vanilla Adam; swap for `torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)`. Helps close the 4.5-nat generalisation gap.

### 2.4 Dropout in the simple embedding  *(new)*

The per-channel MLP in `SimpleEmbeddingNetwork` ([train_model_cpu.py:147-170](train_model_cpu.py#L147-L170)) has no regularisation at all. Given the remaining train-val gap, adding a single `nn.Dropout(0.1)` before the `merge` block is the cheapest regulariser to try. Already present in `Conv1DEmbeddingNetwork` ([train_model_cpu.py:205](train_model_cpu.py#L205)), so it'd be consistent.

### 2.5 Test-split is loaded but never evaluated  *(new)*

[train_model_cpu.py:485](train_model_cpu.py#L485) unpacks `test_data` / `test_params` from the dataset and then never uses them. Worth adding a post-training `with torch.inference_mode(): test_log_prob = model(test_params, test_data).mean().item()` and stashing it in the checkpoint. Otherwise we're quietly ignoring the held-out split that tells us about true generalisation (val is being used for early-stopping, so it's technically contaminated).

### 2.6 Rational-quadratic spline couplings  *(carried over from v1, bigger effort)*

Affine couplings ([train_model_cpu.py:25-82](train_model_cpu.py#L25-L82)) parameterise each flow step as `y = x·exp(s) + t`. That's enough to model roughly-Gaussian posteriors but plateaus on multimodal or heavy-tailed ones — which is exactly what GW posteriors look like (e.g. `coa_phase`, `polarization`). Switching to `nflows` or `zuko` RQ-spline couplings is a bigger refactor but is what the real DINGO paper uses.

---

## 3. Speed improvements (CPU)

### 3.1 Thread-count tuning  *(carried over from v1)*

Still not done. At module top:

```python
import os, torch
torch.set_num_threads(os.cpu_count())
torch.set_num_interop_threads(1)
```

Worth also benchmarking `os.cpu_count() // 2` (physical cores only, skipping hyperthreads).

### 3.2 Downsample waveforms before training  *(carried over from v1)*

Still the biggest easy speedup. 8192 samples/detector @ 4096 Hz → 2048 samples/detector @ 1024 Hz halves `seq_len`, which roughly halves the cost of the `LayerNorm(T)` and `Linear(T, h)` layers in `SimpleEmbeddingNetwork` ([train_model_cpu.py:156-157](train_model_cpu.py#L156-L157)).

Two places it could live:

- Inside `load_dataset_pt` ([train_model_cpu.py:325](train_model_cpu.py#L325)) — e.g. `X = F.avg_pool1d(X, kernel_size=4)` — no re-generation needed.
- Inside `generate_dataset.py` (lower the `TIME_RESOLUTION`) — better numerically, but out of scope per the earlier "don't touch generation" rule.

### 3.3 Stop storing `test_data` on the hot path  *(new, minor)*

Currently `load_dataset_pt` keeps the test split in memory for the whole training run. On a bigger dataset (after §2.2) that's wasted RAM. Either load the test split only after training or skip it until `test` is actually used (§2.5).

### 3.4 `torch.compile(model)`  *(carried over from v1)*

Worth a try once `NUM_EPOCHS` grows beyond ~20 — the warm-up cost amortises over longer runs. Current 20-epoch budget is too short to see a win.

---

## 4. Correctness / housekeeping

### 4.1 `plot_results.py` will break on new checkpoints  *(new)*

The v2 checkpoint schema dropped `gps_time_delay` and the `DINGOModel` constructor signature changed (`num_detectors` / `seq_len` instead of `data_dim`, no `time_delay_value`). [plot_results.py](plot_results.py) still loads with the old schema — it will fail to instantiate the model. Needs a small patch: read `num_detectors` and `seq_len` from `checkpoint['config']`, drop the `time_delay_value` argument.

### 4.2 Optimizer / scheduler state not saved  *(carried over from v1)*

Checkpoint has `model_state_dict` and `best_state` but not optimizer/scheduler state, so training can't resume from a saved checkpoint. Add:

```python
'optimizer_state_dict': optimizer.state_dict(),
'scheduler_state_dict': scheduler.state_dict(),
'bad_epochs': bad_epochs,
```

…and plumb a resume path in `__main__`.

### 4.3 `SimpleEmbeddingNetwork` shares weights across detectors — is that what we want?  *(new)*

The new `per_channel` MLP is shared across detectors ([train_model_cpu.py:153-162](train_model_cpu.py#L153-L162)). That's "permutation-equivariant" between H1 and L1: swapping the two detectors at input produces the same merged context up to the merge layer's weights. For PSD-whitened strain this is probably fine (both detectors have similar spectral content after whitening). But if you ever feed raw detector data with different PSDs, per-detector weights might train better. Not a bug, but worth a sentence in your notes.

### 4.4 `best_state = copy.deepcopy(model.state_dict())` at init  *(new, minor)*

[train_model_cpu.py:440](train_model_cpu.py#L440) deep-copies the randomly-initialised state before training starts. If epoch 1 improves on that (it always will), it's a wasted copy. Trivial cost but clean to initialise `best_state = None` and only copy on the first "improved" epoch.

### 4.5 `noise_str` in filename still misleading when `WHITEN=True`  *(carried over from v1)*

Not addressed yet — [train_model_cpu.py:497](train_model_cpu.py#L497) still emits `noisy_whitened` which reads as "noisy despite being whitened". Either `whitened_from_noisy` or drop the `noisy` tag when `WHITEN=True`.

---

## 5. Quick-win shortlist (post-v1)

Three changes that address what the latest run *actually* showed was broken, each under 30 minutes:

1. **Switch to `ReduceLROnPlateau`** — stops wasting the last 5–10 epochs at lr=1e-6 (§2.1).
2. **`AdamW` + a dash of dropout in the simple embedding** — cheapest attack on the 4.5-nat train-val gap (§2.3, §2.4).
3. **Evaluate and log test log-prob at the end** — free information about generalisation we're currently throwing away (§2.5).

---

## 6. Bigger effort

Same list as v1, trimmed for what's still open:

- More training data + noise augmentation (§2.2) — the blocker on further accuracy gains.
- Rational-quadratic spline couplings (§2.6).
- `torch.compile` (§3.4) — revisit once epoch counts grow.
- Update [plot_results.py](plot_results.py) to the new checkpoint schema (§4.1) — not optional if you want to actually *look at* the model's posteriors.
- Resume-from-checkpoint plumbing (§4.2).
