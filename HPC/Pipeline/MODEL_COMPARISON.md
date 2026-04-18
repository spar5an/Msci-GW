## Embedding-network comparison: `simple` vs `conv1d` vs `lstm`

### 1. Context

This report compares two trained embedding networks feeding the same 4-layer affine-coupling normalising flow (`context_dim=128`, `hidden_dim=64`) on an 8 000 / 1 000 / 1 000 train/val/test split of whitened H1+L1 strain (sample rate 4096 Hz, 2 s window, fixed merger at t = 1.0 s). The dataset is unchanged from the baseline described in [ASSESSMENT_real_data_gap.md](ASSESSMENT_real_data_gap.md).

- **`simple`** — per-detector flat MLP, original baseline. Checkpoint: [dingo_N8k_F4_C128_H64_E20_simple_whitened_cpu.pt](dingo_N8k_F4_C128_H64_E20_simple_whitened_cpu.pt). Outputs in [plots/simple/](plots/simple/).
- **`conv1d`** — single 1-D conv stack over the **time-concatenated** `(N, 1, 2·T)` tensor (H1 | L1 placed end-to-end along time). Checkpoint: [dingo_N8k_F4_C128_H64_E20_conv1d_whitened_cpu.pt](dingo_N8k_F4_C128_H64_E20_conv1d_whitened_cpu.pt). Outputs in [plots/conv1d/](plots/conv1d/).
- **`lstm`** — two-stream shared-weight BiLSTM with a per-channel stride-2 conv front-end (see [train_model_cpu.py:246-300](train_model_cpu.py#L246-L300)). **Training was cancelled at epoch 16 before any checkpoint was saved** ([train_lstm.log](train_lstm.log)); see §7 for why and what would be needed to make it competitive.

All three models were trained with identical optimiser settings (AdamW, lr = 1e-4, wd = 1e-4, batch = 32, 20 epochs, patience = 10, ReduceLROnPlateau).

### 2. Architectures at a glance

**`simple`** — each detector's 8192 samples are flattened, fed through an independent 3-layer MLP (8192 → 512 → 256 → 128), and the two 128-dim embeddings concatenated + projected to the 128-dim flow context. Sees every sample but has no notion of temporal locality. Fastest to train; most parameters concentrated in the first dense layer.

**`conv1d`** — the two detector channels are concatenated along the **time** axis to form `(N, 1, 16384)`, then processed by a stack of stride-2 conv + BatchNorm + ReLU blocks (channels 32 → 64 → 128 → 128), global-average-pooled, projected to 128-dim. This is a single-detector model that sees both strains as one long signal with a discontinuity at t = 16384. Translation-equivariant within each detector's block, parameter-efficient.

**`lstm`** — each detector `(N, 1, 8192)` is pushed through a shared stride-2 conv front-end (16 → 32 → 64 features, 3 stride-2 stages) taking 8192 → ~1024 time steps, then a 2-layer bidirectional LSTM (hidden 128) shared across detectors. The final forward+backward hidden state per detector is concatenated (2·2·128 = 512), the two detectors' vectors concatenated (1024), and a 2-layer MLP projects to 128-dim. Designed to model time-ordered dependencies; sequence length even after down-sampling is ~1024, which dominates wall-clock.

### 3. Parameter counts and wall-clock

| Model | Parameters | Wall-clock / epoch (8 k samples) | Best val log-prob | Epoch of best |
|:------|-----------:|---------------------------------:|------------------:|--------------:|
| `simple` | ~1.30 M | ~2 min | -5.69 | 20 |
| `conv1d` | 864 744 | ~2 min | -8.37 | 19 |
| `lstm`   | ~0.95 M | ~8 min | -8.97 (partial) | 15 (of 16 run before cancel) |

`lstm` training reached epoch 16 before being killed ([train_lstm.log:42](train_lstm.log#L42)); no checkpoint was written, so no test-set or real-data evaluation is available. The stopping decision is discussed in §6.

### 4. Synthetic test-split metrics

Mean log-prob over 1 000 held-out synthetic events (higher is better — less negative is better):

| | `simple` | `conv1d` |
|:---|---:|---:|
| Mean log-prob | **-5.97** | -8.52 |
| Std log-prob | 3.93 | 1.54 |

Per-parameter (median posterior σ / median \|z\| / coverage @ 95 %) from [plots/simple/metrics.txt](plots/simple/metrics.txt) and [plots/conv1d/metrics.txt](plots/conv1d/metrics.txt):

| Parameter | `simple` σ / \|z\| / c95 | `conv1d` σ / \|z\| / c95 |
|:---|:---|:---|
| mass1 | 25.53 / 0.77 / 96.9 % | 26.06 / 0.78 / 99.5 % |
| mass2 | 30.79 / 0.64 / 100.0 % | 23.98 / 0.82 / 99.5 % |
| spin1z | 0.926 / 0.43 / 100.0 % | 0.451 / 0.91 / 99.8 % |
| spin2z | 0.418 / 0.96 / 94.9 % | 0.464 / 0.88 / 99.7 % |
| distance | 1099 / 0.90 / 90.5 % | 1929 / 0.56 / 100.0 % |
| ra | 2.48 / 0.69 / 100.0 % | 1.82 / 0.92 / 100.0 % |
| dec | 0.718 / 0.73 / 96.7 % | 0.686 / 0.74 / 97.2 % |

On **synthetic** data `simple` wins overall mean log-prob by ~2.5 nats. `conv1d`'s posteriors are consistently **wider** (higher σ on distance and mass1, lower on mass2/ra), which pushes `|z|` up and KS statistics down — classic under-confidence. No `conv1d` parameter passes KS uniformity at 1 % except `dec` (p ≈ 0.24) and `inclination` (p = 0.04).

### 5. Real O4 metrics (73 events with catalogue params)

From [plots/simple/real_metrics.txt](plots/simple/real_metrics.txt) and [plots/conv1d/real_metrics.txt](plots/conv1d/real_metrics.txt). The "winner" column is whichever model is closer to the catalogue value or has better-calibrated coverage:

| Parameter | `simple` Δ / σ_post / c68 / c95 | `conv1d` Δ / σ_post / c68 / c95 | Winner |
|:---|:---|:---|:---|
| mass1   | -15.30 / 25.89 / 63.0 % / 94.5 % | **-6.87** / 25.14 / 71.2 % / 97.3 % | `conv1d` |
| mass2   | -20.21 / 30.64 / 71.2 % / 98.6 % | -18.32 / **23.59** / 58.9 % / 95.9 % | `conv1d` (tighter σ, better c95) |
| distance | 426 / 1100 / 43.8 % / 83.6 %   | **249** / 1923 / 80.8 % / 98.6 % | `conv1d` (smaller bias, better c95) |
| ra      | -0.17 / 2.42 / 82.2 % / 100 %    | -0.17 / **1.78** / 63.0 % / 100 % | `conv1d` (tighter posteriors) |
| dec     | 0.15 / 0.71 / 37.0 % / 89.0 %    | 0.21 / 0.69 / 28.8 % / 93.2 % | tie (simple lower bias, conv1d better c95) |
| mean log-prob on real events | **-5.48** | -6.63 | `simple` |

**The striking inversion**: `conv1d` lost to `simple` on synthetic log-prob by 2.5 nats, but **beats `simple` on every real-data parameter residual except dec bias**, and halves the mass1 bias (-15 → -7 M⊙), halves the distance bias (426 → 249 Mpc), and narrows the RA posterior width by ~25 %. The only place `simple` wins on real data is the mean-log-prob proxy itself, which rewards tight peaks near the catalogue value — `conv1d`'s wider distance/mass posteriors cost it log-prob but cover the truth more often.

### 6. Qualitative observations from the plots

**`simple`** — [plots/simple/coverage.png](plots/simple/coverage.png) shows mass2 and spin1z over-coverage (posteriors too wide on synthetic), distance and polarization under-coverage tails. Real-data corner plots (e.g. GW150914, [plots/simple/corner_real_GW150914.png](plots/simple/corner_real_GW150914.png)) show a posterior concentrated in a narrow mass ridge, but that ridge is systematically displaced from the catalogue truth by ~15 M⊙ — the model has strong priors that the training data pulled into the wrong place.

**`conv1d`** — [plots/conv1d/coverage.png](plots/conv1d/coverage.png) shows near-uniform over-coverage on everything except dec: the posteriors are broader but the true value is more often inside them. [plots/conv1d/pp_plot.png](plots/conv1d/pp_plot.png) is visibly flatter than simple's, though only `dec` / `inclination` pass formal KS. Real corner plots (e.g. GW230820_212515, [plots/conv1d/corner_real_GW230820_212515-v1.png](plots/conv1d/corner_real_GW230820_212515-v1.png)) show the mass posterior drifting closer to the catalogue value and the distance posterior swelling to acknowledge uncertainty rather than confidently miss.

**Interpretation** — `conv1d` is **less over-fit to the in-distribution noise realisations** of the synthetic dataset. The shared conv kernels across the 16 384-sample concatenated sequence are a much stronger inductive bias for signal-morphology-from-strain than `simple`'s dense layers, which can memorise per-sample quirks of the synthetic noise. On synthetic test data, memorising the quirks pays (lower log-prob); on real O4 strain it hurts, because the real noise quirks are different.

### 7. `lstm`: why it was cancelled and what might help

By epoch 15 the BiLSTM's validation log-prob was **-8.97**, still worse than `conv1d` at the same point in training ([train_conv1d.log:41](train_conv1d.log#L41) was -8.77 at epoch 15, and converged to -8.37 by epoch 19). The epoch-to-epoch improvement had slowed to < 0.1 nats/epoch, and each LSTM epoch was roughly 4× slower than `conv1d`. There was no reason to expect the LSTM to overtake `conv1d` by epoch 20, and the two models share the same inductive-bias problem: **both see the full 8 192 whitened samples per detector, 95 % of which are pre-merger or post-ringdown noise**.

#### The proposed fix: crop to ±500 samples around the merger, no down-sampling

The BBH signal's SNR is dominated by the last few cycles of inspiral, the merger peak, and the ringdown — a window of roughly 50 – 300 ms for O4 events. At 4 096 Hz that is ~200 – 1 200 samples. A **1 000-sample crop centred on the merger** (±500 samples either side of the pipeline's fixed `target_length//2` merger placement) captures essentially all of the waveform information with:

- **8× fewer time steps per detector** (8 192 → 1 000). The LSTM cost scales linearly in sequence length; 8× fewer steps means 8× faster training, so we could run 20 epochs in the time `conv1d` currently takes.
- **No down-sampling.** The conv front-end in the current LSTM code is throwing away information — a stride-2 stack turns 8 192 into ~1 024 before the LSTM ever sees it, and the stride-2 filters are learnt from scratch with no guarantee they preserve the high-frequency merger content. Hand-cropping to the physics-relevant window keeps the native 4 096 Hz resolution, lets the LSTM see every merger-frequency cycle, and removes an entire learnt stage.
- **A much better LSTM task.** Recurrent models struggle on long, mostly-noise sequences because they must carry information across thousands of uninformative steps. A 1 000-step merger-centred window is the sweet spot for a BiLSTM with hidden-dim 128.

Concretely, the change lives in the dataset-construction side ([data_gen.py](data_gen.py), not the model): after whitening and the fixed-merger placement, slice `data[:, :, merger_idx - 500 : merger_idx + 500]` and save the cropped tensor. The model dispatch in [train_model_cpu.py:303-307](train_model_cpu.py#L303-L307) then uses `seq_len=1000` and can drop the conv front-end entirely (feed raw strain to the BiLSTM), or keep a light single-stage conv with stride 1 for feature extraction.

Secondary suggestions that would help any of the three architectures, ranked by expected real-data impact:
1. **Merger-centred crop + retrain all three** — same rationale as above, and `simple`'s flat MLP will also stop wasting 87 % of its parameters on pre-merger samples.
2. **More synthetic events** — at 8 k training samples and 13 parameters, the flow is undersized's data budget, not its capacity (both conv1d and simple still improve monotonically through epoch 19 – 20). Target 40 k events before trying fancier architectures.
3. **Data augmentation with real O4 PSDs not used at training time** — currently the same cached PSD set is reused across all 10 k synthetic events, so the synthetic noise distribution is narrower than real O4 noise. Rotate PSDs per-event.
4. **Hybrid head: `conv1d` masses + merger-cropped LSTM for sky** — mass1 and distance are both better with `conv1d`'s global view (full 16 384-sample sequence picks up the chirp scaling with frequency); RA/dec are dominated by inter-detector timing differences near the merger, which a merger-cropped model should resolve with far less data.

### 8. Ranked next steps (extending [ASSESSMENT_real_data_gap.md §4](ASSESSMENT_real_data_gap.md))

| # | Action | Target parameter(s) | Expected impact | Effort |
|--:|:---|:---|:---|:---|
| 1 | Regenerate dataset with ±500-sample merger crop; retrain `simple` and `conv1d` on it | all | high — removes 87 % of uninformative samples, should tighten every posterior | ~1 day data-gen + 2× 20-epoch training |
| 2 | Retrain `lstm` on the cropped dataset (no conv front-end) and re-evaluate | RA/dec (timing-sensitive), mass ratio | medium — the architecture will finally be on a sequence length it can handle | 1 day |
| 3 | Increase synthetic dataset to 40 k events; re-run simple+conv1d | all | high — both models' val log-prob was still improving at epoch 19 | 1 day data-gen + 2× training |
| 4 | Rotate real O4 PSDs per synthetic event instead of reusing cache | distance coverage, real-data log-prob | medium — closes sim-to-real noise gap that `simple` is most sensitive to | 0.5 day |
| 5 | Hybrid embedding: `conv1d` branch (full 16 384) + `lstm` branch (1 000 cropped) concatenated before flow | mass1 + ra jointly | medium-high, speculative — conditioned on step 1 working | 2 days |
| 6 | Bump `num_flow_layers` from 4 to 8 and `hidden_dim` from 64 to 128 on the winning embedding | sharpness of modes, multi-modality | low-medium — flow capacity isn't the current bottleneck | 1 day |

The primary recommendation is **step 1**: the observed `simple` → `conv1d` improvement on real data is explained by `conv1d` relying less on per-sample memorisation of synthetic noise quirks. Cropping to the merger window removes the largest single source of those quirks (pre-merger detector noise that the flow learns to associate with specific parameters via coincidence). Every architecture benefits, and the LSTM in particular becomes a genuinely viable candidate rather than a slow also-ran.
