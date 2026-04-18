# evaluate_model.py
# Load a trained DINGO checkpoint, run posterior inference on the held-out
# split of dataset.pt, and produce report-quality plots + NPE metrics.
#
# Usage:
#     python3.11 evaluate_model.py [checkpoint.pt]
#
# If a checkpoint path is supplied it overrides the default. Outputs are
# written to HPC/Pipeline/plots/<embedding>/ where <embedding> is parsed
# from the checkpoint filename.

import math
import os
import sys
from pathlib import Path

import corner
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

from train_model_cpu import DINGOModel, load_dataset_pt


DEVICE = torch.device('cpu')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_model_from_checkpoint(ckpt):
    """Rebuild a DINGOModel that matches a saved checkpoint's config."""
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
    ).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def denormalize(z_samples, param_names, param_norm_info):
    """Undo the train-split z-score from load_dataset_pt."""
    mean = np.array([param_norm_info[n]['mean'] for n in param_names])
    std = np.array([param_norm_info[n]['std'] for n in param_names])
    safe = np.where(std > 0, std, 1.0)
    return z_samples * safe + mean


def format_label(name):
    return {
        'mass1': r'$m_1\ [M_\odot]$',
        'mass2': r'$m_2\ [M_\odot]$',
        'spin1z': r'$\chi_{1z}$',
        'spin2z': r'$\chi_{2z}$',
        'distance': r'$d_L\ [\mathrm{Mpc}]$',
        'inclination': r'$\iota$',
        'coa_phase': r'$\phi_c$',
        'ra': r'$\alpha$',
        'dec': r'$\delta$',
        'polarization': r'$\psi$',
        'm_g': r'$m_g\ [\mathrm{kg}]$',
        'alpha_lv': r'$\alpha_{\mathrm{LV}}$',
        'A': r'$A_{\mathrm{LV}}$',
    }.get(name, name)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, data, truths_physical, param_names, param_norm_info,
                  num_samples=5000, event_batch=16):
    """Sample posterior for each event and collect summary statistics.

    Returns a dict with shape-(N, P) arrays for posterior mean/std, z-score,
    truth quantile within the marginal posterior, and a shape-(N, S, P) list
    of posterior samples (in physical units).
    """
    N = data.shape[0]
    P = len(param_names)
    means = np.zeros((N, P))
    stds = np.zeros((N, P))
    zscores = np.zeros((N, P))
    quantiles = np.full((N, P), 0.5)
    samples_all = []

    std_arr = np.array([param_norm_info[n]['std'] for n in param_names])
    degenerate = std_arr == 0.0

    with torch.inference_mode():
        for i in range(0, N, event_batch):
            chunk = data[i:i + event_batch]
            truth_chunk = truths_physical[i:i + event_batch]
            for j in range(chunk.shape[0]):
                z = model.sample_posterior(chunk[j:j + 1], num_samples=num_samples)
                z_np = z.cpu().numpy()
                phys = denormalize(z_np, param_names, param_norm_info)
                samples_all.append(phys)

                mu = phys.mean(axis=0)
                sig = phys.std(axis=0)
                truth = truth_chunk[j]

                means[i + j] = mu
                stds[i + j] = sig

                # z-score: guarded against zero-width posteriors
                safe_sig = np.where(sig > 0, sig, 1.0)
                z_score = (truth - mu) / safe_sig
                z_score[sig == 0] = 0.0
                zscores[i + j] = z_score

                # quantile of truth within marginal posterior samples
                q = (phys < truth[None, :]).mean(axis=0)
                q[degenerate] = 0.5
                quantiles[i + j] = q

            if (i // event_batch) % 5 == 0:
                print(f"  inference: {min(i + event_batch, N)}/{N}")

    return {
        'means': means,
        'stds': stds,
        'zscores': zscores,
        'quantiles': quantiles,
        'samples': samples_all,
        'degenerate': degenerate,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_training_curves(ckpt, outpath):
    train = ckpt.get('losses', [])
    val = ckpt.get('val_losses', [])
    if not train:
        print("  [train_curves] no losses in checkpoint — skipping")
        return
    epochs = np.arange(1, len(train) + 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, train, label='train', lw=2)
    if val:
        vx = np.arange(1, len(val) + 1)
        ax.plot(vx, val, label='val', lw=2)
    be = ckpt.get('best_epoch', 0)
    if be:
        ax.axvline(be, ls='--', color='grey', alpha=0.7, label=f'best @ ep{be}')
    ax.set_xlabel('epoch')
    ax.set_ylabel('log-prob')
    ax.set_title('Training curves')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  saved {outpath}")


def plot_corner(samples, truth, param_names, title, outpath):
    labels = [format_label(n) for n in param_names]
    # corner chokes on zero-variance dims — nudge them with a negligible spread
    # so the KDE doesn't error, without moving the distribution visibly.
    s = samples.copy()
    for j in range(s.shape[1]):
        if s[:, j].std() == 0:
            s[:, j] = s[:, j] + np.random.normal(0, 1e-12, size=s.shape[0])
    fig = corner.corner(
        s, truths=truth, labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True, title_kwargs={'fontsize': 9},
        label_kwargs={'fontsize': 9},
        truth_color='tab:red',
    )
    fig.suptitle(title, y=1.01)
    fig.savefig(outpath, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {outpath}")


def plot_pp(quantiles, param_names, degenerate, outpath):
    N, P = quantiles.shape
    grid = np.linspace(0, 1, 200)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], color='black', lw=1, ls='--', label='ideal')
    cmap = plt.get_cmap('tab20')
    for j, name in enumerate(param_names):
        if degenerate[j]:
            continue
        q_sorted = np.sort(quantiles[:, j])
        empirical = np.searchsorted(q_sorted, grid, side='right') / N
        ax.plot(grid, empirical, color=cmap(j % 20), lw=1.5, label=format_label(name))
    ax.set_xlabel('expected CDF')
    ax.set_ylabel('empirical CDF')
    ax.set_title('P–P plot (truth quantile within marginal posterior)')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right', fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  saved {outpath}")


def plot_coverage(quantiles, param_names, degenerate, outpath, levels=(0.5, 0.68, 0.9, 0.95, 0.99)):
    levels = np.array(levels)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], color='black', lw=1, ls='--', label='ideal')
    cmap = plt.get_cmap('tab20')
    coverage_table = {}
    for j, name in enumerate(param_names):
        if degenerate[j]:
            continue
        lo = 0.5 - levels / 2
        hi = 0.5 + levels / 2
        cov = [((quantiles[:, j] >= l) & (quantiles[:, j] <= h)).mean()
               for l, h in zip(lo, hi)]
        coverage_table[name] = cov
        ax.plot(levels, cov, 'o-', color=cmap(j % 20), label=format_label(name))
    ax.set_xlabel('nominal credible level')
    ax.set_ylabel('empirical coverage')
    ax.set_title('Credible-interval coverage')
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right', fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  saved {outpath}")
    return coverage_table, levels


def plot_residual_zscores(zscores, param_names, degenerate, outpath):
    keep = [j for j in range(len(param_names)) if not degenerate[j]]
    n = len(keep)
    ncols = 4
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows))
    axes = np.atleast_2d(axes).flatten()
    x = np.linspace(-4, 4, 200)
    pdf = stats.norm.pdf(x)
    for ax_idx, j in enumerate(keep):
        ax = axes[ax_idx]
        z = zscores[:, j]
        z = z[np.isfinite(z)]
        ax.hist(z, bins=30, density=True, alpha=0.7, color='steelblue',
                range=(-4, 4))
        ax.plot(x, pdf, color='red', lw=1.2, label=r'$\mathcal{N}(0,1)$')
        ax.set_title(format_label(param_names[j]), fontsize=10)
        ax.set_xlim(-4, 4)
        ax.grid(alpha=0.3)
    for ax in axes[n:]:
        ax.axis('off')
    fig.suptitle(r'Posterior-mean residual z-scores: $(\theta_{\rm true}-\mu)/\sigma$',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  saved {outpath}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(model, eval_params, eval_data, results, param_names,
                    coverage_table, coverage_levels, eval_logprob_mean):
    P = len(param_names)
    lines = []
    lines.append(f"Mean log-prob on eval split: {eval_logprob_mean:.4f}")
    lines.append("")
    lines.append(f"{'param':<14}  {'med σ':>10}  {'med |z|':>8}  "
                 + "  ".join(f'cov@{int(l*100)}%' for l in coverage_levels)
                 + "   KS (p)")
    for j, name in enumerate(param_names):
        med_sig = float(np.median(results['stds'][:, j]))
        med_absz = float(np.median(np.abs(results['zscores'][:, j])))
        if results['degenerate'][j]:
            cov_str = '  '.join('   -   ' for _ in coverage_levels)
            ks_str = '    -'
        else:
            cov_values = coverage_table.get(name, [float('nan')] * len(coverage_levels))
            cov_str = '  '.join(f"{c*100:6.1f}%" for c in cov_values)
            ks = stats.kstest(results['quantiles'][:, j], 'uniform')
            ks_str = f"{ks.statistic:5.3f} ({ks.pvalue:.2g})"
        lines.append(f"{name:<14}  {med_sig:10.4g}  {med_absz:8.3f}  {cov_str}   {ks_str}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    CHECKPOINT = 'dingo_N8k_F4_C128_H64_E20_conv1d_whitened_cpu.pt'
    if len(sys.argv) > 1:
        CHECKPOINT = sys.argv[1]
    DATASET_PATH = 'Data/dataset.pt'
    EVAL_SPLIT = 'test'
    NUM_SAMPLES = 5000
    NUM_CORNERS = 3

    embedding_stem = next(
        (tag for tag in ('simple', 'conv1d', 'lstm') if tag in Path(CHECKPOINT).stem),
        'misc',
    )
    PLOT_DIR = Path('plots') / embedding_stem
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    SEED = 0
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"Loading checkpoint: {CHECKPOINT}")
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    model = build_model_from_checkpoint(ckpt)
    print(f"  Best epoch: {ckpt.get('best_epoch', '?')}, "
          f"best log-prob: {ckpt.get('best_log_prob', float('nan')):.4f}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    print(f"\nLoading dataset: {DATASET_PATH}  ({EVAL_SPLIT} split)")
    crop_hw = ckpt['config'].get('merger_crop_half_width')
    if crop_hw is not None:
        print(f"  Applying merger crop ±{crop_hw} samples (from checkpoint config)")
    ds = load_dataset_pt(
        DATASET_PATH,
        use_whitened=ckpt['config'].get('whiten', True),
        merger_crop_half_width=crop_hw,
    )
    eval_data = ds[f'{EVAL_SPLIT}_data']
    eval_params_z = ds[f'{EVAL_SPLIT}_params']
    param_names = ds['param_names']
    param_norm_info = ds['param_norm_info']
    print(f"  {EVAL_SPLIT}: N={len(eval_params_z)}, data shape={tuple(eval_data.shape)}")

    truths_physical = denormalize(eval_params_z.cpu().numpy(), param_names, param_norm_info)

    # ---- Mean log-prob on eval split ----
    with torch.inference_mode():
        eval_logprob = model(eval_params_z.to(DEVICE), eval_data.to(DEVICE))
        eval_logprob_mean = eval_logprob.mean().item()
    print(f"\nMean log-prob on {EVAL_SPLIT}: {eval_logprob_mean:.4f}")

    # ---- Posterior sampling for every event ----
    print(f"\nSampling {NUM_SAMPLES} posterior draws for each of {len(eval_data)} events …")
    results = run_inference(
        model, eval_data, truths_physical, param_names, param_norm_info,
        num_samples=NUM_SAMPLES, event_batch=16,
    )

    # ---- Pick best / median / worst events by per-event log-prob ----
    per_event_lp = eval_logprob.cpu().numpy()
    order = np.argsort(per_event_lp)
    worst_idx = int(order[0])
    best_idx = int(order[-1])
    median_idx = int(order[len(order) // 2])
    picks = [('best', best_idx), ('median', median_idx), ('worst', worst_idx)][:NUM_CORNERS]

    # ---- Plots ----
    plot_training_curves(ckpt, PLOT_DIR / 'training_curves.png')

    for tag, idx in picks:
        title = f'{tag.capitalize()} event (idx {idx}, log-prob {per_event_lp[idx]:.2f})'
        plot_corner(results['samples'][idx], truths_physical[idx], param_names,
                    title, PLOT_DIR / f'corner_{tag}_event{idx}.png')

    plot_pp(results['quantiles'], param_names, results['degenerate'],
            PLOT_DIR / 'pp_plot.png')
    coverage_table, levels = plot_coverage(
        results['quantiles'], param_names, results['degenerate'],
        PLOT_DIR / 'coverage.png')
    plot_residual_zscores(results['zscores'], param_names, results['degenerate'],
                          PLOT_DIR / 'residual_zscores.png')

    # ---- Metrics ----
    metrics_text = compute_metrics(
        model, eval_params_z, eval_data, results, param_names,
        coverage_table, levels, eval_logprob_mean,
    )
    print('\n' + metrics_text)
    metrics_path = PLOT_DIR / 'metrics.txt'
    metrics_path.write_text(metrics_text + '\n')
    print(f"\nMetrics written to {metrics_path}")
    print(f"All plots in: {PLOT_DIR.resolve()}")
