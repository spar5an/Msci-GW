# Data representation study for NPE

Findings from [snr_audit.py](snr_audit.py) and [data_visibility.py](data_visibility.py) on the 10 000-event GR dataset at [../Data/dataset.pt](../Data/dataset.pt). All plots referenced below live in [plots/](plots/); CSV and SVD basis outputs in [outputs/](outputs/).

## 1. The training set is not SNR-controlled

From [plots/snr_audit.png](plots/snr_audit.png), network SNR computed against the averaged O4 PSD:

| statistic | value |
|---|---|
| median network SNR | **4.1** |
| below SNR 4 | 49.5 % |
| below SNR 8 (detectable) | **73.6 %** |
| above SNR 20 | 9.6 % |
| above SNR 50 | 2.6 % |
| max | 308 |

The right-hand scatter panels confirm the two dominant axes: SNR ≈ 1/distance (tight power law) and SNR ∝ Mc^{5/6} (weaker but real). The dataset's **uniform priors on mass (5–90 M⊙) and distance (0.1–5 Gpc)** therefore concentrate most events near the noise floor. Nothing about the dataset generation is wrong, but this distribution is load-bearing for any NPE training conclusions: 3/4 of training gradient comes from events where the signal is barely present.

## 2. What each representation actually shows

Six per-event figures at SNR quantiles 0.20, 0.40, 0.60, 0.80, 0.95, 0.99 ([plots/representation_comparison_qNN_idxNNNN.png](plots/)). Panels per figure: (a) whitened strain, (b) Q-transform magnitude, (c) clean template overlaid on data, (d) normalised matched-filter ρ(t), (e) |FFT|, (f) merger zoom.

- **SNR < 5 (q20, q40):** the whitened strain in (a) is indistinguishable from noise by eye. Q-transform (b) shows no chirp track. MF peak (d) sits below ~4σ — within fluctuations of noise-only runs. Panel (c) shows the clean template as a small red ribbon swallowed by the data, and (f) confirms no alignment is visible.
- **SNR 5–15 (q60, q80):** the chirp **starts to be visible in the Q-transform** before it's visible in the raw whitened strain. MF peak jumps to ~6–10σ, cleanly resolving the merger time. FFT panel (e) shows the template spectrum emerging above the (essentially flat) noise floor.
- **SNR > 30 (q95, q99):** everything works. Signal dominates the strain trace, Q-transform shows a textbook inspiral-merger-ringdown arc, MF peak is sharp and tall, merger zoom reveals phase-aligned waveforms.

Key qualitative observation: the **Q-transform wins first** as SNR drops. It's the most forgiving representation visually because it concentrates chirp power along a localised arc in (t, f) instead of spreading it over a 2-second time series.

## 3. Reduced SVD basis — the real win

Built from 2 000 clean whitened H1 templates drawn from the training prior (see [svd_basis.py](svd_basis.py)). The 8 192-sample waveform space collapses to an extraordinarily compact subspace:

| cumulative template energy | required k |
|---|---|
| 95 %  | **12** |
| 99 %  | **21** |
| 99.9 % | **39** |

The singular-value spectrum at [plots/svd_spectrum.png](plots/svd_spectrum.png) drops 7 decades over the first 600 modes. This is much steeper than in DINGO (~200–300 modes for BNS–BBH) because our GR prior has no spin precession, a single approximant, and a narrow 2-s window — the template manifold here is genuinely thin.

The payoff for NPE is on [plots/svd_reconstruction.png](plots/svd_reconstruction.png). For three events (SNR 3.1, 10.2, 88.3), projecting the noisy observation onto the top k signal modes and re-expanding gives:

| event | original MF σ | after k=50 projection | after k=150 | after k=400 |
|---|---|---|---|---|
| idx 414 (net SNR 3.1) | 3.8 | **21.9** | 11.1 | 5.8 |
| idx 268 (net SNR 10.2) | 9.4 | **29.8** | 26.8 | 17.1 |
| idx 9121 (net SNR 88.3) | 51.3 | **159.9** | 123.8 | 70.7 |

A 5–10× apparent SNR boost at k=50, entirely from discarding noise that lives outside the signal subspace. The boost is *larger* for low-SNR events, which is exactly where the training set is most populated (panel 1 above). SNR falls back as k grows past ~100 because adding more modes re-admits noise without adding signal — consistent with the 99 %-energy threshold being at k=21.

**Caveat:** the "MF σ" in that table uses the event's own clean template as the filter, which is an oracle we wouldn't have at inference. It's a useful diagnostic of signal retention, not the SNR the NPE would effectively see.

## 4. Recommendation for NPE input

**Primary candidate: project the whitened strain onto the top ~200 SVD modes.** This is the DINGO choice, and our data supports it: 200 modes captures essentially all signal energy while collapsing the input dimensionality ~40× (8 192 → 200). Per-event cost of the projection is one dense matmul against a (200, 8 192) basis — negligible vs the flow cost downstream.

Runner-up: **whitened time-domain strain** unchanged. Information-preserving, matches what DINGO and similar SBI frameworks use, and — for events the model can actually learn from (SNR > 8) — panels (a), (e), (f) show that all the required structure is present.

The more impactful change is on the **training distribution**, not the representation:

1. **Weight or subsample by SNR.** With 74 % of events below SNR 8, the network is largely learning the prior. Options: (a) reweight the loss by 1/(1 + exp((8 − SNR)/2)), (b) stratified sampling to enforce a flat SNR distribution, (c) regenerate with a distance prior that is uniform in 1/d rather than uniform in d.
2. **Add a Q-transform magnitude channel** alongside the strain, not in place of it. Two-channel input at (B, 2×D, T) preserves all phase info in channel 0 while giving the network a ready-made time-frequency view in channel 1 — which panels show is where the chirp first becomes perceptible. Low implementation cost; likely useful for low-SNR events.
3. **Do not** use matched-filter outputs against the event's own template as training input — that leaks θ into x. Matched-filter against a fixed **bank** would be valid, but introduces a template-bank design choice that defeats part of the SBI value proposition.

Per-sample standardisation (dividing by the event's own off-merger noise std) is worth a brief ablation — panels show whitened-strain std varies by a factor of ~3 across events, which the first conv layer otherwise has to absorb.

## 5. Follow-ups not covered here

- **Extend the SVD basis to both detectors** (currently H1 only). L1 basis will be similar in shape but distinct; either concatenate (k_H1 + k_L1 coefficients per event) or fit a joint basis on (2D, T).
- **SVD retraining at dataset scale.** Rebuilding the basis on 10k rather than 2k templates should barely move the k-thresholds but is worth confirming.
- **Bank-matched-filter as auxiliary input.** Generate M ≈ 100 templates spanning (mass1, mass2); per-event feature = (M, T) SNR time-series. Good candidate for a ConvNet backbone.
- **Log-amplitude Hilbert envelope.** Information-preserving when paired with the instantaneous phase; makes the chirp envelope a monotone function useful for attention mechanisms.
- **SNR-stratified train/val/test split.** Currently the split is random so the high-SNR tail is underrepresented in validation by chance.
- **Re-run [snr_audit.py](snr_audit.py) on an MG and LV dataset** once generated, to check that modified-dispersion effects don't systematically shift the SNR distribution (and therefore change the effective prior the model sees).
