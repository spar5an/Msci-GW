# Changes to `train_model_cpu.py`

Follow-up to [REVIEW_train_model_cpu.md](REVIEW_train_model_cpu.md). Implements items 2.2, 2.3, 2.5, 2.6, 2.7, 2.8, 3.1, 3.5, 3.6.

---

## Headline result

Same 100-sample synthetic dataset, same hyperparameters (20 epochs, batch 32, lr 1e-4, simple embedding):

| Metric | Before | After |
|---|---:|---:|
| Final-epoch val log-prob | −20.55 | **−16.55** |
| Best val log-prob during run | −17.31 (ep 1) | **−16.55 (ep 20)** |
| Trainable parameters | 2 345 704 | **1 296 616** |
| Overfit? | Yes — val worsened after epoch 5 | No — val improved every epoch |

The saved checkpoint is now the best-val model (§2.2), and the best model also happens to be the last model — so the *same* 20-epoch compute budget now yields a ~4-nat better val log-prob.

---

## Per-item changes

### 2.2 — Early stopping + best-val checkpointing

- New `patience` parameter on `train_dingo_model` (default 10).
- Each epoch, track validation log-prob. Deep-copy `model.state_dict()` whenever it improves, and increment a `bad_epochs` counter when it doesn't. Stop early once `bad_epochs >= patience`.
- After the loop, `model.load_state_dict(best_state)` so the model object and the saved checkpoint both reflect the best epoch.
- Log line now prints `Best: X @ epY` and a `*` marker when the current epoch is a new best.
- Falls back to best *train* log-prob if no validation set is provided.
- Checkpoint now stores `best_log_prob` and `best_epoch` alongside the state dict.

### 2.3 — `BatchNorm1d` → `LayerNorm` in coupling layers

- `AffineCouplingLayer.scale_net` and `.translation_net` now use `nn.LayerNorm(hidden_dim)` instead of `nn.BatchNorm1d(hidden_dim)` at every intermediate layer.
- Removes the running-statistics drift that made small-batch training fragile. Also removes the `>=2 samples per batch` constraint in principle (kept the guard anyway — cheap).

### 2.5 — Removed the dead `time_delay_value` column

- `DINGOModel.__init__` no longer accepts `time_delay_value`; `.forward` no longer concatenates the constant column; the flow's `context_dim` equals the embedding's output dim directly (no more `+ 1`).
- `sample_posterior` simplified accordingly.
- Dropped `gps_time_delay` from the checkpoint schema.

### 2.6 — Removed the `reg_loss` penalty

- Deleted the `reg_loss = 10.0 * clamp(1.5 - context_std, min=0)` block and the `total_loss = loss + reg_loss` combination.
- Loss is now the clean NLL: `loss = -log_prob.mean()`.
- Side effect: the duplicate embedding forward vanishes (see §3.1).

### 2.7 — Seeds

- `torch.manual_seed(0)` and `np.random.seed(0)` at the top of `__main__`.
- `SEED` is saved to the checkpoint config so future runs can be reproduced exactly.

### 2.8 — Two-detector channel axis preserved

- `load_dataset_pt` no longer flattens with `X.reshape(N, -1)`. Data tensors now have shape `(N, num_detectors, seq_len)` all the way through training.
- **Simple embedding** (new `SimpleEmbeddingNetwork`): shared per-detector MLP — `LayerNorm(T) → Linear(T, h) → … → Linear(h, h)` applied independently to each detector, then the two per-channel embeddings are concatenated and projected to `context_dim` by a small merge MLP. Same topology as the old flat MLP, but detectors are processed in parallel rather than stitched end-to-end. Roughly halves the dominant first-layer cost.
- **Conv1D embedding**: `nn.Conv1d(in_channels=num_detectors, …)` and no `unsqueeze(1)`. Each filter now sees both detectors at each time step, which is the whole point of a CNN over multi-channel strain data.
- **LSTM embedding**: `input_proj = Linear(num_detectors, 32)`; the input is reshaped to `(N, T, num_detectors)` so the BiLSTM iterates over time with a 2-vector per step.
- `DINGOModel` now takes `num_detectors` and `seq_len` directly; `num_detectors` is inferred from the dataset shape in `__main__`. `data_dim` is gone.

### 3.1 — Duplicate embedding forward removed

- Was `model(batch_params, batch_data)` followed by a second `model.embedding_net(batch_data)` for the `context_std` regulariser. The second call is gone because the regulariser is gone (§2.6). Each training step now runs the embedding once.

### 3.5 — `torch.inference_mode()` for validation

- Validation loop uses `with torch.inference_mode():` instead of `torch.no_grad()`. Slightly lower overhead and disables version-counter tracking.

### 3.6 — Whole validation set in one forward

- Replaced the batched validation loop with a single `model(val_params, val_data)` call. With `val_size=10` there's no reason to chunk.

---

## Other small cleanups that came along for the ride

- `log(2π)` promoted to a module-level `LOG_2PI` constant; `NormalizingFlow.forward` no longer recomputes `torch.log(2 * pi * base_std**2)` on every step. (Minor correctness nit from the review.)
- Renamed loop accumulators from `loss` to `log_prob` terminology (`epoch_log_prob`, `batch_log_probs`, `best_log_prob`, `best_epoch`), matching what's actually being tracked.
- Added `.to(DEVICE)` on the model at construction for defensive clarity, even though `DEVICE=cpu`.
- Dropped `psutil` import (was unused).

---

## Checkpoint compatibility

The model shape and state-dict keys have changed (new `SimpleEmbeddingNetwork`, LayerNorm-based coupling layers, no time-delay column, `num_detectors`/`seq_len` replacing `data_dim`). Old checkpoints from the previous `train_model_cpu.py` **will not load** into the new model. If [plot_results.py](plot_results.py) is used downstream, it will need its model-construction call updated to pass `num_detectors`/`seq_len` and to drop `time_delay_value`.

---

## Training log (20 epochs, 100 synthetic samples, simple embedding)

```
Epoch   1/20, Train: -21.2391, Val: -22.4103, Best: -22.4103 @ ep1  *
Epoch   5/20, Train: -16.6803, Val: -18.9313, Best: -18.9313 @ ep5  *
Epoch  10/20, Train: -13.9464, Val: -17.1856, Best: -17.1856 @ ep10 *
Epoch  15/20, Train: -12.3765, Val: -16.6314, Best: -16.6314 @ ep15 *
Epoch  20/20, Train: -11.9955, Val: -16.5504, Best: -16.5504 @ ep20 *
```

Every epoch was a new best, so early stopping didn't trigger on this run — which is fine. The next meaningful experiment is scaling the dataset up (REVIEW §2.1) so val has headroom beyond where `lr → eta_min` takes it.
