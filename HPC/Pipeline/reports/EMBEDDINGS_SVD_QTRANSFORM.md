# New embedding models: `svd_mlp` and `qtransform_conv2d`

Companion note to [REPRESENTATIONS.md](REPRESENTATIONS.md). That earlier study argued, from the SNR audit and the per-event representation panels, that a reduced SVD projection and a Q-transform magnitude image are both promising NPE inputs. This note documents the **embedding networks that consume those representations** and the pipeline plumbing that now produces them.

## 1. Why two new embeddings

The three strain-domain embeddings (`simple`, `conv1d`, `lstm`) all operate on `(N, D, T)` whitened strain with `T = 8192` (or a cropped window). The two new embeddings consume different trailing shapes:

| representation | per-detector input | embedding |
|---|---|---|
| `strain`     | `(T,)`        | `simple` / `conv1d` / `lstm` |
| `svd`        | `(K,)`        | `svd_mlp` |
| `qtransform` | `(F, T_q)`    | `qtransform_conv2d` |

`DINGOModel.__init__` now takes `data_shape=(T,) | (K,) | (F, T_q)` instead of a bare `seq_len`, and dispatches to the matching embedding class. The old `seq_len=int` form still works — see [ML/cpu/train_model.py:604–619](../ML/cpu/train_model.py#L604-L619).

Allowed pairings are enforced by [hp_search.prune_for_embedding](../ML/cpu/hp_search.py#L116): the grid drops any `(representation, embedding_type)` combination not in the dispatch table.

## 2. `SVDMLPEmbeddingNetwork`

Source: [ML/cpu/train_model.py:491–541](../ML/cpu/train_model.py#L491-L541).

Input `(N, D, K)` — one coefficient vector per detector per event, produced by `apply_svd_projection` against a cached basis (§4). The network is deliberately simple:

```
LayerNorm(K) → Linear(K, 2H) → ReLU → Linear(2H, 2H) → ReLU → Dropout
                            (shared across detectors by default)
concat across detectors  →  Linear(D·2H, 2H) → ReLU → Dropout → Linear(2H, context_dim) → LayerNorm
```

Default `H = hidden_dim = 128`. With `D = 2`, `K = 200`, `context_dim = 128` this is roughly **260 k parameters** — ~70× smaller than the strain MLP (`simple`) on the full 8 192-sample window, which is the point: the SVD projection already did the heavy dimension reduction, so the embedding can afford to be trivial.

Design notes:
* **LayerNorm on the input** matters. SVD coefficients for low-SNR events are concentrated in the first ~20 modes and near-zero thereafter; raw coefficients have orders-of-magnitude variation that a linear layer would struggle to absorb.
* **Shared-weight per-detector MLP** is on by default (`share_detector_weights=True`). H1 and L1 see the same chirp manifold, so a shared encoder is the right prior; swap to independent encoders only when you genuinely expect detector-specific morphology.
* **No temporal structure.** The SVD basis has already flattened time — there is nothing sequential left to exploit, so we use an MLP, not a 1-D conv.

## 3. `QTransformConv2DEmbeddingNetwork`

Source: [ML/cpu/train_model.py:544–584](../ML/cpu/train_model.py#L544-L584).

Input `(N, D, F, T_q)`. Detectors become **input channels** to a 2-D conv stack — this is the choice `qtransform_conv2d` makes over a two-stream design, and it keeps the network trivially light. Stack:

```
Conv2d(D → 32, 3×3, s=1, p=1) → BN → ReLU → MaxPool(2)
Conv2d(32 → 64, 3×3, s=1, p=1) → BN → ReLU → MaxPool(2)
Conv2d(64 → 128, 3×3, s=1, p=1) → BN → ReLU → MaxPool(2)
AdaptiveAvgPool2d(1) → flatten → Linear(128, 256) → ReLU → Dropout → Linear(256, context_dim) → LayerNorm
```

With default `num_filters = (32, 64, 128)` and `context_dim = 128` this is **~130 k parameters**, again much lighter than the strain embeddings.

Design notes:
* **Global average pooling** at the end of the conv stack means the network is shape-agnostic in `(F, T_q)` — if you switch `qtransform_logfsteps` or `qtransform_delta_t_out`, the same architecture still fits, as long as the tensor survives three 2× max-pools (so `F ≥ 8`, `T_q ≥ 8`).
* **Detectors as channels (not streams)** is a deliberate loss of per-detector individuality. The conv's first-layer kernels see both H1 and L1 pixels simultaneously and the network can learn coincidence features directly. The alternative — two separate encoders merged late — is strictly more expressive but ~2× the params, and worth trying only if the channel-mixing design underperforms.
* **BatchNorm** (not LayerNorm) is used here because Q-transform magnitudes are strictly non-negative and highly peaked along the chirp arc; BN's per-channel scaling handles that dynamic range better than LN over a full image would.

## 4. Data-side plumbing

[`load_dataset_pt`](../ML/cpu/train_model.py#L805) now branches on `input_representation ∈ {strain, svd, qtransform}` *before* the train/val/test split, so the split lines up with whichever trailing shape the embedding expects.

**SVD.** `_load_or_build_svd_basis` ([train_model.py:756](../ML/cpu/train_model.py#L756)):
* Looks for `Data experiments/outputs/svd_basis_H1.npz`. If present, loads it.
* Otherwise builds one on the fly from **2 000 clean templates** drawn from the training prior via `generate_clean_templates`, runs SVD on the `(M, T)` matrix with `k_max = max(svd_k, 600)`, and caches the result to disk. Done once per fresh clone; subsequent runs reload instantly.
* Slices the top `svd_k` rows of the basis and projects via `einsum('kt,ndt->ndk')`. Default `svd_k = 200`, matching the primary-candidate recommendation in [REPRESENTATIONS.md §4](REPRESENTATIONS.md).
* **Gotcha:** the basis is time-axis-specific. If `merger_crop_half_width` is changed after the basis is built, the length check in [train_model.py:906–910](../ML/cpu/train_model.py#L906-L910) fires. Either delete the cached basis to force a rebuild at the new length, or leave `merger_crop_half_width=None` when using SVD.

**Q-transform.** `_apply_qtransform` ([train_model.py:740](../ML/cpu/train_model.py#L740)) is a thin wrapper over `gw_datagen.compute_qtransform_batch`. Defaults:

| knob | default | note |
|---|---|---|
| `qtransform_frange` | `(20.0, 300.0)` Hz | matches the whitening low-cut and covers the chirp band for the prior mass range |
| `qtransform_qrange` | `(4.0, 16.0)` | pycbc's `TimeSeries.qtransform` Q tile search |
| `qtransform_logfsteps` | `50` | log-spaced frequency bins → `F = 50` |
| `qtransform_delta_t_out` | `0.002` s | output grid 500 Hz → `T_q ≈ 1000` for a 2-second input |

Per-event Q-transform is ~100 ms on CPU, so generating it for the whole dataset before training is fine for current dataset sizes. At HPC scale, consider caching the Q-transformed tensor alongside `dataset.pt`.

## 5. Checkpoint interop

Every run now writes `data_shape`, `input_representation`, `svd_k`, `svd_basis_path`, `qtransform_logfsteps`, `qtransform_delta_t_out`, `qtransform_frange`, `qtransform_qrange` into `checkpoint['config']`. [evaluate_model.py](../evaluation/evaluate_model.py) and [evaluate_real.py](../evaluation/evaluate_real.py) read those keys back and re-apply the same representation transform on the held-out split before sampling the flow. That means **an SVD- or Q-transform-trained checkpoint is fully self-describing** — no need to remember which knobs it was trained under.

Checkpoint filenames also get a representation tag: `dingo_..._svd200_...pt`, `dingo_..._qt_...pt`, or no tag for raw strain. Parsed by the embedding-stem logic at [evaluate_model.py:342–347](../evaluation/evaluate_model.py#L342-L347) so per-representation plots land in their own subdirectory under `plots/`.

## 6. What to watch on first full-scale runs

1. **Does the cached SVD basis survive prior changes?** The basis is built from whatever templates `generate_clean_templates` produces at build time. If the training prior is later widened (e.g. new mass range), the cached basis silently no longer spans the new manifold. Delete `outputs/svd_basis_H1.npz` when the prior changes.
2. **Q-transform memory on GPU.** `(N, D, F, T_q) = (N, 2, 50, 1000)` at float32 is 400 kB per event. An 8 k training split is ~3 GB — fine resident in RAM, but tight on smaller GPUs if the whole tensor is uploaded at once. The GPU trainer's `val_chunk_size` path already handles this for validation; training uses the standard DataLoader so per-batch transfer is sufficient.
3. **`svd_mlp` underfitting.** The embedding has no nonlinearity on the K-dim input beyond two dense ReLUs. If the metrics come in flatter than `simple` on strain, the first thing to try is a deeper MLP (pass `hidden_dim=256`) rather than reaching for a conv.
4. **Q-transform phase blindness.** `compute_qtransform_batch` returns `|q|`. Sky-localisation parameters that rely on inter-detector phase difference (RA, dec, polarization) may degrade relative to strain. This is the expected trade-off; worth quantifying explicitly in the next [MODEL_COMPARISON.md](MODEL_COMPARISON.md) update.
