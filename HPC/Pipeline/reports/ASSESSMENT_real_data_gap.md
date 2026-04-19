# Why the NPE underperforms on real O4 data — assessment & recommendations

Follow-up to [REVIEW_train_model_cpu_v2.md](REVIEW_train_model_cpu_v2.md) and [CHANGES_train_model_cpu_v2.md](CHANGES_train_model_cpu_v2.md). Written after the first real-data run of [evaluate_real.py](evaluate_real.py) against 112 O4 events using the 10 k-sample checkpoint [dingo_N8k_F4_C128_H64_E20_simple_whitened_cpu.pt](dingo_N8k_F4_C128_H64_E20_simple_whitened_cpu.pt).

## 1. Executive summary

The flow generalises from synthetic to real noisily but with clear systematic biases. The dominant causes, ranked by how much signal they carry:

1. **Fixed coalescence-time placement** — the network has never seen timing jitter ([gw_datagen.py:550–583](Data Generation/gw_datagen.py#L550-L583)).
2. **Frozen noise realisations** — one Gaussian draw baked in at generation time, no per-epoch re-draw ([gw_datagen.py:717–731](Data Generation/gw_datagen.py#L717-L731)).
3. **Flat-vector MLP embedding** — 8192 → 128 in a single learned projection with no temporal or frequency-domain structure ([train_model_cpu.py:141–195](train_model_cpu.py#L141-L195)).
4. **PSD vintage mismatch** — synthetic data uses an O4a-only PSD cache, but real events span O4a/O4b/O4c.

The affine-only coupling flow and the narrow, uniform-in-distance prior are real but secondary. The sample-to-parameter ratio (8 k / 1.3 M) is genuinely tight but masked by the data-generation issues above — fixing those first will reveal how much capacity headroom there actually is.

## 2. Observed symptoms

From [plots/real_metrics.txt](plots/real_metrics.txt) (73 of 112 events have catalogue truth; 5/13 parameters are anchored):

| Parameter | N | median Δ (catalogue − μ_post) | median σ_post | median &#124;z&#124; | cov @ 68 % | cov @ 95 % |
|---|---:|---:|---:|---:|---:|---:|
| mass1 | 73 | **−15.30 M⊙** | 25.89 | 0.75 | 63.0 % | 94.5 % |
| mass2 | 73 | **−20.21 M⊙** | 30.64 | 0.69 | 71.2 % | 98.6 % |
| distance | 73 | **+426.0 Mpc** | 1100 | 1.18 | **43.8 %** | **83.6 %** |
| ra | 73 | −0.17 rad | **2.42 rad** | 0.56 | 82.2 % | 100 % |
| dec | 73 | +0.15 rad | 0.71 | 1.37 | 37.0 % | 89.0 % |

Mean per-event log-prob: synthetic-test −5.97 vs real-data −5.48 (partial-truth proxy — catalogue only covers 5/13 params, so the comparison is not apples-to-apples).

Headline failure modes:
- **mass1/mass2 biased low by 15–20 M⊙** with near-nominal 95 % coverage → shape is right, mean is off.
- **distance over-confident** — 84 % coverage at nominal 95 %, biased low.
- **RA essentially prior-dominated** (σ ≈ 2.4 rad ≈ 3/4 of the full 2π prior width).
- **dec shows mild bias + 68 % under-coverage** despite 95 % coverage sitting near nominal.

## 3. Diagnosis

### 3.1 Fixed merger placement → mass bias

[gw_datagen.py:550–583](Data Generation/gw_datagen.py#L550-L583) IRFFT-rearranges every waveform so coalescence sits at sample `target_length // 2` (t = 1.0 s, exactly). Real events have sub-sample coalescence-time jitter relative to any reference window, and the O4 preprocessed tensor aligns by GPS merger but still carries epoch-level offsets. A flow that has never seen a shifted merger will try to absorb the shift into the parameters it *can* vary — chirp-mass being the most sensitive. A late merger looks like a higher chirp mass (shorter inspiral in-window); correspondingly an early/misaligned merger biases the inferred masses **low**, which is what we see.

### 3.2 Frozen noise realisations → distance over-confidence

[gw_datagen.py:717–731](Data Generation/gw_datagen.py#L717-L731) draws one `noise_from_psd` sample per waveform per detector at generation time and injects it into the signal. The tensor that lands in [Data/dataset.pt](Data/dataset.pt) is that single realisation — no re-draw in `load_dataset_pt` ([train_model_cpu.py:345](train_model_cpu.py#L345)). Effective training set for noise realisations is therefore **8 000**, not "8 000 × N_epochs × ∞ Gaussian draws" as the NPE literature assumes. The flow learns the signal pattern conditional on *these specific noise instances* and consequently predicts posteriors tighter than warranted — exactly the 83.6 % / 95 % coverage gap on distance.

### 3.3 O4a-only PSD cache → distance bias

PSDs are cached from a May 2023 – Jan 2024 GPS window ([Data Generation/download_o4_psds.py](Data Generation/download_o4_psds.py), O4a science segments). Real events in the evaluation set run well past that window (several GW230628–GW250114 events). Detector noise character drifts month-to-month (laser power glitches, seismic input, squeezer state), so whitening a real O4b/O4c event with its own contemporaneous PSD leaves a coloured residual that differs from the synthetic training distribution. Distance is the parameter most sensitive to the absolute noise amplitude — a mild systematic in residual whitening shows up as the observed +426 Mpc bias.

### 3.4 Flat-vector embedding → RA prior-domination

[train_model_cpu.py:141–195](train_model_cpu.py#L141-L195) ingests each detector's 8192-sample whitened strain via `LayerNorm → Linear(8192, 128) → ReLU → Linear(128, 128) → …`. The first linear is a single learned linear combination — no convolution, no pooling, no frequency-domain features. Sky localisation (RA/DEC) is driven by the **sub-ms time-of-arrival difference between H1 and L1**, which lives in the cross-correlation of the two strain streams — exactly the kind of feature a flat MLP cannot extract. The fact that a `Conv1DEmbeddingNetwork` already exists at [train_model_cpu.py:198](train_model_cpu.py#L198) but the training run used `EMBEDDING_TYPE='simple'` means the fix is already mostly implemented; it just isn't being used.

### 3.5 Affine couplings only → multi-modal posterior failure

[train_model_cpu.py:90–133](train_model_cpu.py#L90-L133) stacks four `AffineCouplingLayer`s. Affine transforms can't represent multi-modal 1-D marginals from a unimodal base distribution — GW posteriors have well-known bimodality (distance–inclination degeneracy, RA antipode). The DINGO paper (Dax et al. 2021) uses rational-quadratic spline couplings for precisely this reason. On real data the bimodality is stronger than on synthetic (broader priors in practice), so the limitation bites harder.

### 3.6 Parameter / sample ratio — context, not primary cause

The model has 1 296 616 parameters vs 8 000 training samples (≈ 160× over-parameterised). Weight decay 1e-4 + dropout 0.1 + ~20 epochs keep this from total overfit in-distribution (synthetic train−val gap was ~3.2 nats), but real-data distribution shift amplifies any residual overfitting. This argues for bigger datasets and/or noise augmentation (§4.1 D2) — not necessarily a smaller model, because M1 will likely need more parameters once it stops throwing away features.

## 4. Ranked recommendations

### 4.1 Data generation — highest leverage first

| # | Change | Expected impact | Effort |
|---|---|---|---|
| **D1** | **Randomise merger time** uniformly in `[−0.1, +0.1] s` around the window centre at generation. Replace the hard `n_half = target_length // 2` at [gw_datagen.py:564](Data Generation/gw_datagen.py#L564) with a per-sample jittered offset, and save the jitter alongside truth (either as a new parameter the flow learns to marginalise over, or baked into the waveform pre-whitening). | Directly attacks the mass-bias symptom; also trains the embedding to be translation-tolerant. | **S** — one arg + regenerate. |
| **D2** | **Per-epoch noise re-draw.** Save only the clean signal `X_signal` + per-detector PSD into `dataset.pt`; mix Gaussian noise `noise_from_psd(...)` fresh in the `DataLoader`'s `__getitem__`. Needs schema change to [Data/dataset.pt](Data/dataset.pt) and a matching loader update in [train_model_cpu.py:345](train_model_cpu.py#L345). | Effective noise-realisation count becomes `N_epochs × N_batches × batch` — typically 4–5 orders of magnitude more. Targets the distance over-confidence directly. | **M** — schema + loader. |
| **D3** | **Refresh the PSD cache** to include O4b and O4c segments; sample PSD epoch uniformly per waveform. The `load_random_o4_psd` call at [gw_datagen.py:728](Data Generation/gw_datagen.py#L728) already takes a random draw — the cache just needs more dates. | Matches real-data noise drift; further reduces distance bias and should improve mass2 posteriors (more sensitive to low-frequency noise). | **S** — extension of [Data Generation/download_o4_psds.py](Data Generation/download_o4_psds.py). |
| **D4** | **Widen + reshape the distance prior** to `[50, 10 000] Mpc`, uniform in comoving volume (`p(d) ∝ d²` at low redshift) rather than uniform-in-distance. Update [generate_dataset.py:83](Data Generation/generate_dataset.py#L83). | GWTC-3 spans ~40–8 000 Mpc; the current `[100, 5 000]` uniform truncates both tails and over-represents nearby events. Uniform-in-volume is the physically motivated prior. | **S**. |
| **D5** | **Align spin prior with GWTC.** Narrow `[−0.8, 0.8]` → `[−0.5, 0.5]` at [generate_dataset.py:81–82](Data Generation/generate_dataset.py#L81-L82), or bite the bullet and switch approximant `IMRPhenomD → IMRPhenomXPHM` to model precession. | `|χ| > 0.5` is rare in GWTC — the current range wastes capacity. Precession helps mass bias too because aligned-spin waveforms are an imperfect model of real events. | **S** (narrow) / **M** (XPHM). |
| **D6** | **SNR-augmented draws** — for each generated waveform, emit a second copy rescaled to a target SNR drawn uniformly from `[8, 25]`. Implementable as a post-hoc rescale of `detector_signals` before noise injection at [gw_datagen.py:717](Data Generation/gw_datagen.py#L717). | Cheap way to broaden the effective-SNR distribution without re-generating waveforms; helps distance and mass posteriors on quiet events. | **S**. |

### 4.2 ML model — highest leverage first

| # | Change | Expected impact | Effort |
|---|---|---|---|
| **M1** | **Stop using `SimpleEmbeddingNetwork`; switch to `Conv1DEmbeddingNetwork`** (already implemented at [train_model_cpu.py:198](train_model_cpu.py#L198)). Set `EMBEDDING_TYPE='conv1d'` in [train_model_cpu.py:564](train_model_cpu.py#L564). Re-verify the 3-block setup is deep enough — may need 5 stride-2 blocks to reach 128-dim embedding. | Directly addresses the RA prior-domination. Convolutions see local phase differences between H1/L1 at every time-scale, which is what sky localisation needs. | **XS** (use existing) / **S** (widen to 5 blocks). |
| **M2** | **RQ-spline couplings.** Replace `AffineCouplingLayer` at [train_model_cpu.py:90–123](train_model_cpu.py#L90-L123) with rational-quadratic spline couplings (use `nflows.transforms.PiecewiseRationalQuadraticCouplingTransform`; `nflows` is a small dep). Keep the same 4-layer depth initially. | Fixes multi-modal 1-D marginals (distance–inclination, RA antipodal); the single biggest modelling improvement in the DINGO line of work. | **M** — coupling layer rewrite + dep. |
| **M3** | **Train longer.** Push `NUM_EPOCHS` 20 → 100 and let `ReduceLROnPlateau` cut LR when val log-prob stalls. The last run was still improving at epoch 20 with LR at full 1e-4. | Free gain; costs only compute. | **XS** — config knob. |
| **M4** | **Frequency-domain branch.** Add a second embedding branch operating on `|STFT(strain)|` (hop 128, window 512) — concatenate with the conv1d output before `merge`. Simple to add after M1 lands. | Captures the time-frequency track of the coalescence, which compactly encodes chirp mass. Should reduce the mass bias on top of D1. | **M**. |
| **M5** | **H1↔L1 time-delay feature.** Compute the cross-correlation peak between the two whitened strains per event, feed the lag as an extra scalar into the merge head at [train_model_cpu.py:179](train_model_cpu.py#L179). Sub-ms resolution is what sky localisation needs. | Cheap, targeted fix for RA/DEC that complements M1. | **XS**. |
| **M6** | **Attention pooling over detectors.** Replace `nn.Linear(num_detectors * h, h)` in `merge` with a small transformer block over the per-detector embeddings. Lets the flow down-weight a glitchy detector event-by-event. | Helps events where one IFO is noisier than the other — common in O4b/c. | **S**. |

## 5. Verification strategy (when implementing any of the above)

- Re-run [evaluate_model.py](evaluate_model.py) on the synthetic test split — mean log-prob should not regress significantly from the current −5.97 baseline.
- Re-run [evaluate_real.py](evaluate_real.py). **Primary real-data success criteria** (all measured against the table in §2):
  - mass1/mass2 median residuals shrink to within **±5 M⊙**.
  - distance coverage at 95 % rises back to **≥ 92 %**.
  - RA median σ falls below **1 rad**.
  - dec 68 % coverage rises to **≥ 60 %**.
- Cross-check a few known-loud events individually: GW150914, GW170814, GW230628_231200 posteriors should bracket the published LVK values ([plots/corner_real_GW150914.png](plots/corner_real_GW150914.png) etc.).
- Record the incremental improvement attributable to each change — D1, D2, D3, M1, M2 should each contribute independently, so ablation is worth the compute.

## 6. What NOT to change yet

- **Keep the 13-parameter space** (including the zeroed LV/MG triplet). Dropping the degenerate parameters forces a schema change that propagates through [load_dataset_pt](train_model_cpu.py#L345), the `DINGOModel` constructor, and every downstream plotter. Not worth the churn until the real-data gap is fixed.
- **Keep `share_detector_weights=True`** until M1 has landed and stabilised. Independent per-detector weights are worth re-evaluating once the embedding is capable of producing useful features; doing both at once makes ablation impossible.
- **Keep `context_dim=128`** for the same reason — bottleneck width only matters once the embedding is no longer the bottleneck.
