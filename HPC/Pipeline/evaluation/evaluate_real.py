# evaluate_real.py — run the trained model on real O4 GW events and
# compare posteriors to the GWTC catalogue values where available.
#
# Inputs (relative to HPC/Pipeline/):
#   - <CHECKPOINT>                          trained model
#   - Data/dataset.pt                       synthetic dataset (for param_norm_info)
#   - Real Data/pt/o4_all_events_2s_real.pt 112 pre-whitened real events
#   - gw_events_stats.csv                   catalogue values (ra, dec, mass_1_source,
#                                            mass_2_source, luminosity_distance, chi_eff)
#
# Outputs (plots/):
#   - corner_real_<EVENT>.png   a few named events with catalogue truth lines
#   - real_residuals.png        posterior mean vs catalogue for matched params
#   - real_zscores.png          z-score histograms across matched events
#   - real_widths.png           posterior-width distributions per param
#   - real_logprob.png          log-prob distribution: synthetic test vs real
#   - real_metrics.txt          per-param residual/z-score/coverage summary
#
# Caveats baked into the output:
#   * catalogue rows give source-frame masses; model was trained on source-frame.
#   * spin1z/spin2z aren't in the CSV — only chi_eff. Derived from posterior samples.
#   * inclination, coa_phase, polarization have no catalogue truth.

import csv
import math
import re
import sys
from pathlib import Path

import corner
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

from train_model_cpu import (
    DINGOModel,
    crop_to_merger,
    load_dataset_pt,
    resolve_crop_for_embedding,
)


DEVICE = torch.device('cpu')


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHECKPOINT   = 'dingo_N8k_F4_C128_H64_E20_conv1d_whitened_cpu.pt'
if len(sys.argv) > 1:
    CHECKPOINT = sys.argv[1]
SYN_DATASET  = 'Data/dataset.pt'
REAL_DATA_PT = 'Real Data/pt/o4_all_events_2s_real.pt'
CSV_PATH     = 'gw_events_stats.csv'
_embedding_stem = next(
    (tag for tag in ('simple', 'conv1d', 'lstm') if tag in Path(CHECKPOINT).stem),
    'misc',
)
PLOT_DIR     = Path('plots') / _embedding_stem
PLOT_DIR.mkdir(parents=True, exist_ok=True)
NUM_SAMPLES  = 5000
CORNER_EVENTS = ['GW150914', 'GW230628_231200', 'GW230820_212515', 'GW250114_082203']


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_model_from_checkpoint(ckpt):
    cfg = ckpt['config']
    model = DINGOModel(
        num_detectors=cfg['num_detectors'],
        seq_len=cfg['seq_len'],
        param_dim=cfg['param_dim'],
        context_dim=cfg['context_dim'],
        num_flow_layers=cfg['num_flow_layers'],
        hidden_dim=cfg['hidden_dim'],
        embedding_type=cfg['embedding_type'],
        embedding_dropout=cfg.get('embedding_dropout', 0.1),
        share_detector_weights=cfg.get('share_detector_weights', True),
        lstm_hidden_dim=cfg.get('lstm_hidden_dim', 128),
        lstm_num_layers=cfg.get('lstm_num_layers', 2),
        conv1d_num_filters=tuple(cfg.get('conv1d_num_filters', (64, 128, 256))),
        coupling_type=cfg.get('coupling_type', 'affine'),
        spline_num_bins=cfg.get('spline_num_bins', 8),
        spline_tail_bound=cfg.get('spline_tail_bound', 3.0),
    ).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def denormalize(z_samples, param_names, param_norm_info):
    mean = np.array([param_norm_info[n]['mean'] for n in param_names])
    std = np.array([param_norm_info[n]['std'] for n in param_names])
    safe = np.where(std > 0, std, 1.0)
    return z_samples * safe + mean


def normalize(phys, param_names, param_norm_info):
    mean = np.array([param_norm_info[n]['mean'] for n in param_names])
    std = np.array([param_norm_info[n]['std'] for n in param_names])
    safe = np.where(std > 0, std, 1.0)
    return (phys - mean) / safe


def format_label(name):
    return {
        'mass1': r'$m_1\ [M_\odot]$',
        'mass2': r'$m_2\ [M_\odot]$',
        'chirp_mass': r'$\mathcal{M}\ [M_\odot]$',
        'mass_ratio': r'$q$',
        'spin1z': r'$\chi_{1z}$',
        'spin2z': r'$\chi_{2z}$',
        'distance': r'$d_L\ [\mathrm{Mpc}]$',
        'inclination': r'$\iota$',
        'coa_phase': r'$\phi_c$',
        'ra': r'$\alpha$',
        'dec': r'$\delta$',
        'polarization': r'$\psi$',
        'm_g': r'$m_g$',
        'alpha_lv': r'$\alpha_{\mathrm{LV}}$',
        'A': r'$A_{\mathrm{LV}}$',
    }.get(name, name)


def strip_version(name):
    """GW230601_224134-v1 → GW230601_224134, to match CSV entries."""
    return re.sub(r'-v\d+$', '', name)


def load_catalogue(path):
    """Return dict event_name → dict of floats."""
    table = {}
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            ev = row['event']
            table[ev] = {k: float(v) for k, v in row.items() if k != 'event'}
    return table


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, data, num_samples):
    N = data.shape[0]
    samples_per_event = []
    means = np.zeros((N, data.shape[0]))  # placeholder, fixed after first sample
    with torch.inference_mode():
        for i in range(N):
            z = model.sample_posterior(data[i:i + 1], num_samples=num_samples)
            samples_per_event.append(z.cpu().numpy())
            if (i + 1) % 20 == 0 or (i + 1) == N:
                print(f"  inference: {i + 1}/{N}")
    return samples_per_event


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_corner_real(samples_phys, truth_partial, param_names, title, outpath):
    labels = [format_label(n) for n in param_names]
    s = samples_phys.copy()
    for j in range(s.shape[1]):
        if s[:, j].std() == 0:
            s[:, j] = s[:, j] + np.random.normal(0, 1e-12, size=s.shape[0])
    # corner wants either all truths or none; pass an array with nan for missing.
    truths = np.full(s.shape[1], np.nan)
    for j, n in enumerate(param_names):
        if n in truth_partial and np.isfinite(truth_partial[n]):
            truths[j] = truth_partial[n]
    # corner handles nan by not drawing that line.
    fig = corner.corner(
        s, truths=truths, labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True, title_kwargs={'fontsize': 9},
        label_kwargs={'fontsize': 9},
        truth_color='tab:red',
    )
    fig.suptitle(title, y=1.01)
    fig.savefig(outpath, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {outpath}")


def plot_residuals(matched, outpath, params=('mass1', 'mass2', 'distance', 'ra', 'dec')):
    params = list(params)
    fig, axes = plt.subplots(1, len(params), figsize=(3.8 * len(params), 3.8))
    for ax, p in zip(axes, params):
        truth = np.array([m['truth'][p] for m in matched if p in m['truth']])
        mean = np.array([m['post_mean'][p] for m in matched if p in m['truth']])
        sigma = np.array([m['post_std'][p] for m in matched if p in m['truth']])
        lo, hi = float(min(truth.min(), mean.min())), float(max(truth.max(), mean.max()))
        pad = 0.05 * (hi - lo)
        ax.errorbar(truth, mean, yerr=sigma, fmt='o', alpha=0.6, ms=4,
                    ecolor='grey', elinewidth=0.8, mec='navy', mfc='tab:blue',
                    capsize=2)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], 'k--', lw=1, alpha=0.6)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel(f'catalogue {format_label(p)}')
        ax.set_ylabel(f'posterior mean {format_label(p)}')
        ax.set_title(p, fontsize=10)
        ax.grid(alpha=0.3)
    fig.suptitle('Posterior mean vs catalogue value (error bars: posterior σ)',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  saved {outpath}")


def plot_zscores_real(matched, outpath, params=('mass1', 'mass2', 'distance', 'ra', 'dec')):
    params = list(params)
    fig, axes = plt.subplots(1, len(params), figsize=(3.4 * len(params), 3.4),
                             sharey=True)
    x = np.linspace(-5, 5, 200)
    for ax, p in zip(axes, params):
        zs = []
        for m in matched:
            if p in m['truth'] and m['post_std'][p] > 0:
                zs.append((m['truth'][p] - m['post_mean'][p]) / m['post_std'][p])
        zs = np.array(zs)
        ax.hist(zs, bins=15, density=True, range=(-5, 5), alpha=0.7,
                color='steelblue')
        ax.plot(x, stats.norm.pdf(x), 'r-', lw=1.2, label=r'$\mathcal{N}(0,1)$')
        ax.axvline(0, color='k', lw=0.8, alpha=0.5)
        ax.set_title(f'{p}  (N={len(zs)}, median |z|={np.median(np.abs(zs)):.2f})',
                     fontsize=9)
        ax.set_xlim(-5, 5)
        ax.grid(alpha=0.3)
    fig.suptitle('Z-score of catalogue truth under model posterior', fontsize=11)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  saved {outpath}")


def plot_widths(samples_phys_list, param_names, outpath):
    P = len(param_names)
    stds = np.array([s.std(axis=0) for s in samples_phys_list])  # (N, P)
    # drop degenerate params (zero std across every event because training std=0)
    keep = [j for j in range(P) if stds[:, j].max() > 0]
    n = len(keep)
    ncols = 5
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.4 * nrows))
    axes = np.atleast_2d(axes).flatten()
    for i, j in enumerate(keep):
        ax = axes[i]
        ax.hist(stds[:, j], bins=25, color='tab:orange', alpha=0.8)
        ax.set_title(format_label(param_names[j]), fontsize=10)
        ax.grid(alpha=0.3)
    for ax in axes[n:]:
        ax.axis('off')
    fig.suptitle('Posterior 1σ width distribution across 112 real events',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  saved {outpath}")


def plot_logprob_distribution(real_lp, syn_lp, outpath):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(min(real_lp.min(), syn_lp.min()) - 2,
                       max(real_lp.max(), syn_lp.max()) + 2, 60)
    ax.hist(syn_lp, bins=bins, alpha=0.55, label=f'synthetic test (N={len(syn_lp)})',
            color='tab:blue', density=True)
    ax.hist(real_lp, bins=bins, alpha=0.55, label=f'real events (N={len(real_lp)})',
            color='tab:orange', density=True)
    ax.axvline(syn_lp.mean(), color='tab:blue', ls='--', lw=1,
               label=f'syn mean = {syn_lp.mean():.2f}')
    ax.axvline(real_lp.mean(), color='tab:orange', ls='--', lw=1,
               label=f'real mean = {real_lp.mean():.2f}')
    ax.set_xlabel('log-prob of (partially filled) truth under posterior')
    ax.set_ylabel('density')
    ax.set_title('Per-event log-prob: synthetic test vs real events')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  saved {outpath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    PLOT_DIR.mkdir(exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)

    # --- checkpoint ---
    print(f"Loading checkpoint: {CHECKPOINT}")
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    model = build_model_from_checkpoint(ckpt)
    print(f"  best log-prob @ ep{ckpt.get('best_epoch')}: {ckpt.get('best_log_prob'):.4f}")

    # --- synthetic dataset (for param_norm_info) ---
    crop_hw = resolve_crop_for_embedding(ckpt['config'])
    if crop_hw is not None:
        print(f"Applying merger crop ±{crop_hw} samples (from checkpoint config)")
    param_param = ckpt['config'].get('param_parameterization', 'm1_m2')
    if param_param != 'm1_m2':
        print(f"Parameterization: {param_param}")
    syn = load_dataset_pt(
        SYN_DATASET,
        use_whitened=ckpt['config'].get('whiten', True),
        merger_crop_half_width=crop_hw,
        param_parameterization=param_param,
    )
    param_names = syn['param_names']
    param_norm_info = syn['param_norm_info']

    # --- real data ---
    print(f"Loading real data: {REAL_DATA_PT}")
    real = torch.load(REAL_DATA_PT, weights_only=False)
    X_real_w = real['X_whitened'].float()  # (N_real, 2, 8192)
    X_real_w = crop_to_merger(X_real_w, crop_hw)
    event_names = list(real['metadata']['events'])
    N_real = X_real_w.shape[0]
    print(f"  {N_real} events, data shape {tuple(X_real_w.shape)}")

    # --- catalogue ---
    catalogue = load_catalogue(CSV_PATH)
    # Mc/q-trained models see chirp_mass + mass_ratio instead of mass1 + mass2.
    # The CSV only has source-frame component masses, so derive Mc and q per row.
    if param_param == 'Mc_q':
        csv_to_model = {
            'ra': 'ra', 'dec': 'dec',
            'luminosity_distance': 'distance',
        }
        mass_params = ['chirp_mass', 'mass_ratio']
    else:
        csv_to_model = {
            'ra': 'ra', 'dec': 'dec',
            'mass_1_source': 'mass1', 'mass_2_source': 'mass2',
            'luminosity_distance': 'distance',
        }
        mass_params = ['mass1', 'mass2']
    matched_params = mass_params + ['distance', 'ra', 'dec']

    # --- posterior sampling per event ---
    print(f"\nSampling {NUM_SAMPLES} posteriors per event on {N_real} real events …")
    raw_samples = run_inference(model, X_real_w, NUM_SAMPLES)
    samples_phys = [denormalize(s, param_names, param_norm_info) for s in raw_samples]

    # --- per-event log-prob (catalogue truth where available, posterior mean otherwise) ---
    # For events with catalogue truth we fill {mass1, mass2, distance, ra, dec}
    # from the CSV and the remaining 5 astrophysical params from the posterior mean.
    # Degenerate LV/MG params stay at 0.  This is an imperfect likelihood proxy
    # (partial truth) — matching GWTC on the 5 constrained params is the useful bit.
    logprobs_real = np.zeros(N_real)
    matched = []
    for i, ev in enumerate(event_names):
        cat_name = strip_version(ev)
        cat_row = catalogue.get(cat_name)
        pseudo = samples_phys[i].mean(axis=0).copy()
        truth = {}
        if cat_row is not None:
            for csv_col, model_param in csv_to_model.items():
                v = cat_row[csv_col]
                j = param_names.index(model_param)
                pseudo[j] = v
                truth[model_param] = v
            if param_param == 'Mc_q':
                m1_cat = cat_row['mass_1_source']
                m2_cat = cat_row['mass_2_source']
                m_big, m_small = max(m1_cat, m2_cat), min(m1_cat, m2_cat)
                q_cat = m_small / m_big
                mc_cat = (m1_cat * m2_cat) ** 0.6 / (m1_cat + m2_cat) ** 0.2
                for name, v in (('chirp_mass', mc_cat), ('mass_ratio', q_cat)):
                    j = param_names.index(name)
                    pseudo[j] = v
                    truth[name] = v
        z = normalize(pseudo, param_names, param_norm_info)
        z_t = torch.as_tensor(z[None, :], dtype=torch.float32)
        with torch.inference_mode():
            lp = model(z_t, X_real_w[i:i + 1]).item()
        logprobs_real[i] = lp
        matched.append({
            'event': ev,
            'truth': truth,
            'post_mean': {n: float(samples_phys[i][:, j].mean())
                          for j, n in enumerate(param_names)},
            'post_std':  {n: float(samples_phys[i][:, j].std())
                          for j, n in enumerate(param_names)},
            'samples':   samples_phys[i],
            'logprob':   lp,
        })

    # --- synthetic-test log-prob for comparison ---
    syn_test_data = syn['test_data']
    syn_test_params = syn['test_params']
    with torch.inference_mode():
        syn_lp = model(syn_test_params.to(DEVICE),
                       syn_test_data.to(DEVICE)).cpu().numpy()

    # --- corner plots ---
    name_to_i = {ev: i for i, ev in enumerate(event_names)}
    for ev in CORNER_EVENTS:
        # accept bare name or match any -v* variant
        hit = None
        for cand in event_names:
            if strip_version(cand) == ev or cand == ev:
                hit = cand
                break
        if hit is None:
            print(f"  [corner] {ev} not in real-data file — skipping")
            continue
        i = name_to_i[hit]
        cat = catalogue.get(strip_version(hit), {})
        truth_partial = {csv_to_model[k]: v for k, v in cat.items() if k in csv_to_model}
        if param_param == 'Mc_q' and 'mass_1_source' in cat and 'mass_2_source' in cat:
            m1_cat, m2_cat = cat['mass_1_source'], cat['mass_2_source']
            m_big, m_small = max(m1_cat, m2_cat), min(m1_cat, m2_cat)
            truth_partial['chirp_mass'] = (m1_cat * m2_cat) ** 0.6 / (m1_cat + m2_cat) ** 0.2
            truth_partial['mass_ratio'] = m_small / m_big
        plot_corner_real(
            samples_phys[i], truth_partial, param_names,
            title=f'{hit}  (log-prob {logprobs_real[i]:.2f})',
            outpath=PLOT_DIR / f'corner_real_{hit}.png',
        )

    plot_residuals(matched, PLOT_DIR / 'real_residuals.png', params=matched_params)
    plot_zscores_real(matched, PLOT_DIR / 'real_zscores.png', params=matched_params)
    plot_widths(samples_phys, param_names, PLOT_DIR / 'real_widths.png')
    plot_logprob_distribution(logprobs_real, syn_lp, PLOT_DIR / 'real_logprob.png')

    # --- metrics table ---
    lines = [
        f"Real-data evaluation — {N_real} events, {NUM_SAMPLES} posterior samples each",
        f"Checkpoint: {CHECKPOINT}",
        "",
        f"Mean per-event log-prob:",
        f"  synthetic test : {syn_lp.mean():8.4f}   (std {syn_lp.std():.4f})",
        f"  real events    : {logprobs_real.mean():8.4f}   (std {logprobs_real.std():.4f})",
        "",
        "Per-parameter residuals (catalogue − posterior mean) and z-scores:",
        f"{'param':<10}  {'N':>4}  {'median Δ':>12}  {'median σ_post':>14}  {'median |z|':>11}  {'cov@68%':>8}  {'cov@95%':>8}",
    ]
    for p in matched_params:
        d = np.array([m['truth'][p] - m['post_mean'][p] for m in matched if p in m['truth']])
        sig = np.array([m['post_std'][p] for m in matched if p in m['truth']])
        z = d / np.where(sig > 0, sig, 1.0)
        cov68 = np.mean(np.abs(z) < 1.0)
        cov95 = np.mean(np.abs(z) < 1.96)
        lines.append(
            f"{p:<10}  {len(d):>4}  {np.median(d):12.3f}  {np.median(sig):14.3f}  "
            f"{np.median(np.abs(z)):11.3f}  {cov68*100:7.1f}%  {cov95*100:7.1f}%"
        )
    lines += [
        "",
        "Notes:",
        "  * catalogue chi_eff, inclination, coa_phase, polarization — no direct comparison here.",
        "  * per-event log-prob uses catalogue truth for matched params + posterior mean elsewhere;",
        "    this is a partial-truth proxy, lower than synthetic-test log-prob because truth is imperfect.",
    ]
    text = '\n'.join(lines)
    print('\n' + text)
    (PLOT_DIR / 'real_metrics.txt').write_text(text + '\n')
    print(f"\nWrote {PLOT_DIR / 'real_metrics.txt'}")


if __name__ == '__main__':
    main()
