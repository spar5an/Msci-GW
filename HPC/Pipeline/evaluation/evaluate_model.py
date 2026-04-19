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

# This script lives at <Pipeline>/evaluation/; the DINGOModel class + dataset
# loader live at <Pipeline>/ML/cpu/train_model.py (the GPU trainer is the same
# file + an appended overrides block). Evaluation is light, so we always pull
# from the CPU copy and choose the device at runtime.
PIPELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_ROOT / 'ML' / 'cpu'))
from train_model import DINGOModel, load_dataset_pt, resolve_crop_for_embedding  # noqa: E402

# Sky-time conversion: every synthetic waveform is generated at a single
# reference GPS time (T_REF_DEFAULT = 1126259462.4, GW150914), so the model
# learns RA in that training frame. For a held-out event at a different GPS
# time you would rotate the model's RA forward by the sidereal angle swept
# between t_ref and t_event. The synthetic test split is at t_ref so the
# rotation is an identity — we still route predictions and truths through
# `ra_from_reference_time` so the frame convention is explicit here and the
# code path matches evaluate_real.py (where the conversion actually moves).
sys.path.insert(0, str(PIPELINE_ROOT / 'Real Data'))
from sky_time_conversion import ra_from_reference_time, T_REF_DEFAULT  # noqa: E402


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_model_from_checkpoint(ckpt):
    """Rebuild a DINGOModel that matches a saved checkpoint's config."""
    cfg = ckpt['config']
    # New checkpoints save data_shape explicitly (tuple of the per-detector
    # trailing shape). Older ones only have seq_len — fall back to (seq_len,).
    data_shape = cfg.get('data_shape')
    if data_shape is None:
        data_shape = (cfg['seq_len'],)
    data_shape = tuple(data_shape)
    model = DINGOModel(
        num_detectors=cfg['num_detectors'],
        data_shape=data_shape,
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
    """Undo the train-split z-score from load_dataset_pt."""
    mean = np.array([param_norm_info[n]['mean'] for n in param_names])
    std = np.array([param_norm_info[n]['std'] for n in param_names])
    safe = np.where(std > 0, std, 1.0)
    return z_samples * safe + mean


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


def _violin_x_extent(verts, y_level):
    """x-range where a violin body crosses y_level (for drawing marker lines)."""
    above = verts[:, 1] >= y_level
    crossings = []
    for k in range(len(verts) - 1):
        if above[k] != above[k + 1]:
            y0, y1 = verts[k, 1], verts[k + 1, 1]
            t = (y_level - y0) / (y1 - y0) if y1 != y0 else 0.5
            crossings.append(verts[k, 0] + t * (verts[k + 1, 0] - verts[k, 0]))
    if not crossings:
        return None, None
    return min(crossings), max(crossings)


def plot_zscore_violins(picks, samples_list, truths, param_names, degenerate, outpath):
    """Per-parameter z-score violins for the picked events.

    Each row is one event; each violin shows the marginal posterior samples
    rescaled to (x − μ) / σ on the parameter axis. The red dashed line marks
    the true value's z-score under that posterior. Degenerate params are
    skipped (would divide by zero).
    """
    keep = [j for j in range(len(param_names)) if not degenerate[j]]
    if not keep:
        return
    palette = ['#FFB3BA', '#BAFFC9', '#BAE1FF', '#FFFFBA',
               '#E8BAFF', '#FFD9BA', '#C4F0C5', '#FFC8DD']
    labels = [format_label(param_names[j]) for j in keep]

    fig, axes = plt.subplots(len(picks), 1,
                             figsize=(max(1.1 * len(keep), 10), 3.8 * len(picks)),
                             squeeze=False)
    for row, (tag, idx) in enumerate(picks):
        ax = axes[row, 0]
        phys = samples_list[idx]
        truth = truths[idx]
        for pos, j in enumerate(keep):
            col = phys[:, j]
            col = col[np.isfinite(col)]
            if len(col) == 0:
                continue
            mu, sig = col.mean(), col.std()
            if sig < 1e-30:
                continue
            z = (col - mu) / sig
            parts = ax.violinplot(z, positions=[pos], widths=0.7,
                                  showmeans=False, showmedians=False,
                                  showextrema=False)
            for pc in parts['bodies']:
                pc.set_facecolor(palette[pos % len(palette)])
                pc.set_edgecolor('black')
                pc.set_linewidth(0.8)
                pc.set_alpha(0.85)
            verts = parts['bodies'][0].get_paths()[0].vertices
            for y_level, style, width in ((0.0, 'solid', 1.8),
                                          (+1.0, 'dashed', 1.2),
                                          (-1.0, 'dashed', 1.2)):
                xlo, xhi = _violin_x_extent(verts, y_level)
                if xlo is not None:
                    ax.hlines(y_level, xlo, xhi, colors='black',
                              linewidths=width, linestyles=style)
            if np.isfinite(truth[j]):
                z_true = (truth[j] - mu) / sig
                xlo, xhi = _violin_x_extent(verts, z_true)
                if xlo is None:
                    xlo, xhi = pos - 0.35, pos + 0.35
                ax.hlines(z_true, xlo, xhi, colors='red',
                          linewidths=1.5, linestyles='dashed')
        ax.axhline(0.0, color='black', lw=0.5, alpha=0.25, zorder=0)
        ax.axhline(+1.0, color='black', lw=0.5, ls='dashed', alpha=0.25, zorder=0)
        ax.axhline(-1.0, color='black', lw=0.5, ls='dashed', alpha=0.25, zorder=0)
        ax.set_xticks(range(len(keep)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel('(θ − μ) / σ', fontsize=10)
        ax.set_title(f'{tag.capitalize()} event (idx {idx})', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
    # Legend once on the top axes.
    axes[0, 0].plot([], [], color='black', lw=1.8, label='mean (z=0)')
    axes[0, 0].plot([], [], color='black', lw=1.2, ls='dashed', label='±1σ')
    axes[0, 0].plot([], [], color='red', lw=1.5, ls='dashed', label='truth')
    axes[0, 0].legend(fontsize=9, loc='upper right')
    fig.suptitle('Posterior z-score violins', fontsize=12)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  saved {outpath}")


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
    DATASET_PATH = str(PIPELINE_ROOT / 'Data' / 'dataset.pt')
    EVAL_SPLIT = 'test'
    NUM_SAMPLES = 5000
    NUM_CORNERS = 3

    # Longer tags first so 'qtransform_conv2d' wins over 'conv1d' etc.
    embedding_stem = next(
        (tag for tag in ('qtransform_conv2d', 'svd_mlp',
                         'simple', 'conv1d', 'lstm')
         if tag in Path(CHECKPOINT).stem),
        'misc',
    )
    PLOT_DIR = PIPELINE_ROOT / 'plots' / embedding_stem
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
    cfg = ckpt['config']
    crop_hw = resolve_crop_for_embedding(cfg)
    if crop_hw is not None:
        print(f"  Applying merger crop ±{crop_hw} samples (from checkpoint config)")
    param_param = cfg.get('param_parameterization', 'm1_m2')
    if param_param != 'm1_m2':
        print(f"  Parameterization: {param_param}")
    input_repr = cfg.get('input_representation', 'strain')
    if input_repr != 'strain':
        print(f"  Representation:   {input_repr}")
    ds = load_dataset_pt(
        DATASET_PATH,
        use_whitened=cfg.get('whiten', True),
        merger_crop_half_width=crop_hw,
        param_parameterization=param_param,
        input_representation=input_repr,
        svd_k=cfg.get('svd_k'),
        svd_basis_path=cfg.get('svd_basis_path'),
        qtransform_logfsteps=cfg.get('qtransform_logfsteps', 50),
        qtransform_delta_t_out=cfg.get('qtransform_delta_t_out', 0.002),
    )
    eval_data = ds[f'{EVAL_SPLIT}_data']
    eval_params_z = ds[f'{EVAL_SPLIT}_params']
    param_names = ds['param_names']
    param_norm_info = ds['param_norm_info']
    print(f"  {EVAL_SPLIT}: N={len(eval_params_z)}, data shape={tuple(eval_data.shape)}")

    truths_physical = denormalize(eval_params_z.cpu().numpy(), param_names, param_norm_info)

    # Sky-time convention. The synthetic dataset is generated at the single
    # reference GPS time T_REF_DEFAULT (GW150914), so both the model's RA
    # prediction and the stored truth RA are in the training frame. We route
    # both through `ra_from_reference_time(_, T_REF_DEFAULT)`, which is the
    # identity at t_event == t_ref but keeps the frame convention explicit
    # and guarantees [0, 2π) wrap-around on both sides of the comparison.
    ra_idx = param_names.index('ra') if 'ra' in param_names else None
    if ra_idx is not None:
        truths_physical[:, ra_idx] = ra_from_reference_time(
            truths_physical[:, ra_idx], T_REF_DEFAULT,
        )

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

    # Match the truth-side RA rotation on the posterior samples too. For the
    # synthetic test split this is an identity (all data at T_REF_DEFAULT),
    # but applying the same transform symmetrically keeps the stored samples
    # in the exact frame used for the corner plots, and documents what the
    # evaluator expects of any future non-t_ref dataset.
    if ra_idx is not None:
        for arr in results['samples']:
            arr[:, ra_idx] = ra_from_reference_time(
                arr[:, ra_idx], T_REF_DEFAULT,
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
    plot_zscore_violins(picks, results['samples'], truths_physical, param_names,
                        results['degenerate'], PLOT_DIR / 'zscore_violins.png')

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
