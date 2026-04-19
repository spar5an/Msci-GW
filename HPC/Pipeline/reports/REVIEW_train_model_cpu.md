# Review: `train_model_cpu.py`

Suggestions for making the script **more accurate** and **faster on CPU**.
Each item points at a specific line in [train_model_cpu.py](train_model_cpu.py) so it can be picked up in isolation.

---

## 1. Evidence from the last run

The 20-epoch smoke test on 80 training / 10 validation samples showed textbook overfit:

| Epoch | Train log-prob | Val log-prob |
|-------|---------------:|-------------:|
| 1     | −21.39         | −17.31       |
| 5     | −17.46         | −17.45       |
| 10    | −15.65         | −17.81       |
| 15    | −15.07         | −19.04       |
| 20    | −15.22         | −20.55       |

- Training log-prob kept climbing; validation peaked around epoch 5 and then steadily worsened.
- The saved checkpoint is the **last** epoch, not the best one — we're keeping the worst val-loss model.
- 2.3M parameters vs 80 training samples is a ratio of ~29 000:1; overfit is structural, not a bug.

This frames the rest of the document: the model *can* fit, but we're under-supplied with data and we're not keeping the right checkpoint.

---

## 2. Accuracy improvements (ranked by expected impact)

### 2.1 Generate more data + noise augmentation  *(biggest lever)*

`NUM_SAMPLES=100` in [Data Generation/generate_dataset.py:31](Data Generation/generate_dataset.py#L31) will never train a 2.3M-parameter flow well. Two complementary fixes:

- **Bulk**: raise `NUM_SAMPLES` to at least 10 000. The generator is CPU-parallel (`num_workers=4`) and can produce ~5–10 waveforms/s, so 10 k samples is an overnight run even on a laptop.
- **Noise re-draw per epoch**: right now one fixed noise realization is baked into `dataset.pt` and reused every epoch. Draw fresh noise per epoch (add a "clean signal" + fresh PSD noise inside the training loop). This is a *free* regularizer — it multiplies the effective dataset size by the number of epochs.

### 2.2 Early stopping + best-val checkpoint

Currently the final epoch is saved unconditionally ([train_model_cpu.py:407-480](train_model_cpu.py#L407-L480)). Track best validation log-prob, cache the corresponding state dict, and stop if it hasn't improved for `patience` epochs. On the current run this alone would have saved the epoch-5 model instead of the epoch-20 one (val ≈ −17.4 vs −20.5 — a ~3 nats/dim difference).

Sketch:

```python
best_val = -float('inf')
best_state = None
patience, bad = 10, 0
# inside epoch loop, after computing avg_val_log_prob:
if avg_val_log_prob > best_val:
    best_val, best_state = avg_val_log_prob, {k: v.detach().clone() for k, v in model.state_dict().items()}
    bad = 0
else:
    bad += 1
    if bad >= patience:
        break
# after loop:
model.load_state_dict(best_state)
```

### 2.3 Swap `BatchNorm1d` → `LayerNorm` inside the coupling layers

The scale/translation nets use `BatchNorm1d(hidden_dim)` ([train_model_cpu.py:39-45, 53-59](train_model_cpu.py#L39-L59)). With `BATCH_SIZE=32` and only 2–3 batches per epoch for the 80-sample set, BN's running statistics are badly estimated — at eval time the model sees a different distribution from training.  `LayerNorm(hidden_dim)` is batch-size-independent and usually the right default for MLPs inside flows.

### 2.4 `Adam` → `AdamW` with weight decay

[train_model_cpu.py:390](train_model_cpu.py#L390) uses vanilla Adam. `AdamW(lr=..., weight_decay=1e-4)` is a near-free accuracy bump and a proper L2 regularizer for an overfitting model.

### 2.5 Drop the dead `time_delay_value` column

[train_model_cpu.py:284-286](train_model_cpu.py#L284-L286) concatenates a scalar `time_delay_value` onto the context. For synthetic data this is `0.0` for every sample, so it's a constant feature that the flow has to learn to ignore. Remove the column (and the `+ 1` at [train_model_cpu.py:272](train_model_cpu.py#L272)) until there's a real per-sample value to feed.

### 2.6 Reconsider the `reg_loss` penalty

[train_model_cpu.py:433-435](train_model_cpu.py#L433-L435):

```python
reg_loss = 10.0 * torch.clamp(1.5 - context_std, min=0)
```

Two issues: the magic `1.5` and `10.0` have no justification in the code, and this penalty *forces* the embedding to spread out even when the data doesn't warrant it. If the embedding is collapsing, it's usually a sign that the network is too big for the data (see §2.1). Drop this entirely as a first experiment; keep `NUM_EPOCHS` logs and see if the embedding still collapses.

### 2.7 Reproducibility

At the top of the script:

```python
torch.manual_seed(0)
np.random.seed(0)
```

Currently `torch.randperm` ([train_model_cpu.py:413](train_model_cpu.py#L413)) and weight init are non-deterministic; two back-to-back runs produce different numbers, so we can't tell if a change helped.

### 2.8 Use the two-detector channel axis properly

`Conv1DEmbeddingNetwork.forward` does `x = data.unsqueeze(1)` ([train_model_cpu.py:188](train_model_cpu.py#L188)) — i.e. it treats the flattened `(N, 16384)` tensor as a **single-channel** sequence. But the raw data is `(N, 2 detectors, 8192)`. By flattening we concatenate H1 and L1 end-to-end, destroying cross-detector coherence — which is exactly the signal you'd use to beat noise in the real pipeline.

Fix: stop flattening in `load_dataset_pt` ([train_model_cpu.py:332](train_model_cpu.py#L332)) for the `conv1d` / `lstm` paths and set `Conv1d(in_channels=2, …)` at [train_model_cpu.py:156](train_model_cpu.py#L156). The `simple` MLP still needs the flatten.

The LSTM path ([train_model_cpu.py:228](train_model_cpu.py#L228)) has the same issue — it treats the sequence as 1-channel and would need an input projection over the 2-detector axis instead.

### 2.9 Consider rational-quadratic spline couplings *(bigger experiment)*

If affine couplings plateau even with plenty of data, the DINGO paper's own reference implementation uses RQ-spline couplings via `nflows` / `zuko`. Likely a follow-up PR, not a quick win.

---

## 3. Speed improvements on CPU (ranked by expected speedup)

### 3.1 Remove the duplicate embedding forward  *(easy ~35% per-step saving)*

Every training batch currently runs the embedding network **twice**:

- once inside `model(batch_params, batch_data)` ([train_model_cpu.py:426](train_model_cpu.py#L426))
- again as `model.embedding_net(batch_data)` for the `context_std` regulariser ([train_model_cpu.py:433](train_model_cpu.py#L433))

For the `simple` embedding, that forward pass is dominated by `Linear(16384, 128)` — the single most expensive op in the step. Easiest fix: have `DINGOModel.forward` also return `context` and reuse it for the regulariser (or, per §2.6, drop the regulariser and the whole problem disappears).

### 3.2 Thread-count tuning

Add near the top:

```python
import os, torch
torch.set_num_threads(os.cpu_count())
torch.set_num_interop_threads(1)
```

PyTorch's intra-op threading is often under-used for small matmul workloads. Worth benchmarking both `cpu_count()` and `cpu_count() // 2` (physical cores); hyperthreads occasionally hurt.

### 3.3 Downsample waveforms before training

Data is 8192 samples/detector @ 4096 Hz. For BBH inspiral–merger–ringdown the useful band is ~20 Hz–1 kHz, so 2048 Hz (2 s ⇒ 4096 samples/detector) or even 1024 Hz is enough. Halving the sample rate halves `data_dim` and roughly halves the first Linear's cost.

Could be done post-hoc inside `load_dataset_pt` ([train_model_cpu.py:306](train_model_cpu.py#L306)) with `F.avg_pool1d` or a decimation filter — no re-generation required.

### 3.4 Bottleneck the `simple` embedding

`nn.Linear(data_dim, hidden_dim * 2)` with `data_dim=16384, hidden_dim=64` is 2.1M params in one layer ([train_model_cpu.py:259](train_model_cpu.py#L259)). That's the whole model. Either:

- Insert a bottleneck: `Linear(16384, 512) → Linear(512, 128) → …`.
- Switch to `EMBEDDING_TYPE='conv1d'` — convolutions downsample cheaply on CPU and often train faster than a huge first Linear.

### 3.5 `torch.inference_mode()` for validation

Swap `torch.no_grad()` ([train_model_cpu.py:453](train_model_cpu.py#L453)) for `torch.inference_mode()`. Small but free, and avoids version-counter bookkeeping entirely.

### 3.6 Run the whole val set in one forward

`len(val_params) == 10` in the current smoke-test. The inner loop ([train_model_cpu.py:456-464](train_model_cpu.py#L456-L464)) adds Python overhead for no reason. Just do `model(val_params, val_data)` once.

### 3.7 `torch.compile(model)` (PyTorch ≥ 2.0)

Worth a try for longer runs. Has a warm-up cost (tens of seconds) so it only pays off for ≫20 epochs and stable shapes — revisit after §2.1 pushes runs longer.

### 3.8 Keep data in memory (already correct)

The whole dataset fits in RAM and is already indexed by tensors. No `DataLoader` / `num_workers` changes needed for CPU.

---

## 4. Correctness nits

- **`model.to(DEVICE)` is never called** ([train_model_cpu.py:540](train_model_cpu.py#L540)). `DEVICE=cpu` makes this harmless today, but any future copy-paste to GPU will silently train on CPU. Add it for defensive clarity.
- **Checkpoint saves no optimizer/scheduler state** ([train_model_cpu.py:558-579](train_model_cpu.py#L558-L579)) — training can't be resumed from a checkpoint.
- **Variable-name confusion around `loss` vs log-prob**: `loss = -log_prob.mean()` ([train_model_cpu.py:427](train_model_cpu.py#L427)) is standard, but then `loss_value = -loss.item()` ([train_model_cpu.py:442](train_model_cpu.py#L442)) flips the sign back. The accumulated `epoch_loss` is therefore an average log-prob, and `best_loss` at [train_model_cpu.py:471](train_model_cpu.py#L471) takes the **max** of that. Works, but reads backwards. Rename to `avg_log_prob` / `best_log_prob` consistently.
- **`log(2π)` recomputed every forward** ([train_model_cpu.py:112-113](train_model_cpu.py#L112-L113)): wrap it as a module constant once, e.g. `self.register_buffer('log_two_pi', torch.tensor(math.log(2 * math.pi)))`.
- **`noisy` tag is misleading when `USE_WHITENED=True`** ([train_model_cpu.py:527](train_model_cpu.py#L527)). Whitening flattens most of the coloured noise, so the file name tells you the opposite of what the data looks like. Either `_whitened_from_noisy` or just drop `noisy` when `WHITEN=True`.

---

## 5. Quick-win shortlist

Four changes, all <1 hour total, that should visibly move both metrics:

1. **Early stopping + best-val checkpoint** — saves the right model (§2.2).
2. **Remove the duplicate embedding forward** — ~35% faster per step (§3.1).
3. **`AdamW` + seeds** — better regularisation, reproducible runs (§2.4, §2.7).
4. **`BatchNorm1d` → `LayerNorm` inside coupling layers** — stable at tiny batch sizes (§2.3).

---

## 6. Bigger-effort items

Flagged separately so they don't get bundled with quick wins:

- Generating ≥10 k samples and adding per-epoch noise re-draw (§2.1).
- Refactoring embeddings to properly handle the 2-detector channel axis (§2.8).
- Switching to rational-quadratic spline couplings (§2.9).
- Eventual GPU port / `torch.compile` (§3.7).
