# plot_results.py
# Script 3 of 3: Load the saved model and produce all inference plots.
# Run generate_data.py and train_model.py first.
#
# Inputs:
#   <model_name>.pt  — checkpoint written by train_model.py
#   prepared_data.pt — test tensors written by generate_data.py
# Outputs:
#   Plots/run_<timestamp>/ directory containing PNG plot files

import torch
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import norm as scipy_norm
import os
from datetime import datetime

# Reuse model class definitions and DEVICE from train_model
from train_model import DINGOModel, DEVICE

# Reuse physics/parameter utilities from generate_data
import generate_data
from generate_data import (
    denormalize_params,
    split_samples_into_symmetric_and_component,
    split_vector_into_symmetric_and_component,
    convert_source_to_detector_frame_masses,
    estimate_redshift_from_luminosity_distance,
    transform_component_dict_to_reparameterized,
    convert_samples_to_eval_space,
    convert_vector_to_eval_space,
)

print("Libraries imported successfully")
print(f"PyTorch version: {torch.__version__}")
print(f"Device: {DEVICE}")


# ==============================================================================
# INFERENCE FUNCTION
# ==============================================================================

def infer_with_dingo(model, observed_data, num_samples=5000, param_norm_info=None,
                     param_names=None, denormalize=True):
    """
    Perform inference using the DINGO-style model.

    Args:
        model:           trained DINGOModel
        observed_data:   observed waveform [data_dim] as numpy array
        num_samples:     number of posterior samples to draw
        param_norm_info: dict with normalization info; if provided and denormalize=True,
                         output is in physical parameter space
        param_names:     list of parameter names matching param_norm_info keys
        denormalize:     whether to convert samples from normalized to physical space

    Returns:
        samples:    posterior samples [num_samples, param_dim]
        statistics: dict with mean/median/std/q05/q95, or None for multi-dim
    """
    model.eval()

    if isinstance(observed_data, np.ndarray):
        data_tensor = torch.FloatTensor(observed_data)
    else:
        data_tensor = observed_data

    if data_tensor.dim() == 1:
        data_tensor = data_tensor.unsqueeze(0)

    data_tensor = data_tensor.to(DEVICE)

    with torch.no_grad():
        try:
            samples = model.sample_posterior(data_tensor, num_samples=num_samples)
            samples_np = samples.cpu().numpy()

            if np.isnan(samples_np).any():
                nan_count = np.isnan(samples_np).sum()
                print(f"  ⚠ Warning: NaN values detected in samples ({nan_count}/{samples_np.size} values)")
                samples_np = np.nan_to_num(samples_np, nan=0.0)

            samples = samples_np
        except Exception as e:
            print(f"  ⚠ Warning: Inference failed: {e}")
            samples = np.zeros((num_samples, data_tensor.shape[1] if len(data_tensor.shape) > 1 else 1))

    if denormalize and param_norm_info is not None and param_names is not None:
        for j, param_name in enumerate(param_names):
            if param_name in param_norm_info:
                info = param_norm_info[param_name]
                samples[:, j] = samples[:, j] * info['std'] + info['mean']

    if samples.shape[1] == 1:
        samples = samples.flatten()
        samples_clean = samples[~np.isnan(samples)]
        if len(samples_clean) > 0:
            statistics = {
                'mean': np.mean(samples_clean),
                'median': np.median(samples_clean),
                'std': np.std(samples_clean),
                'q05': np.percentile(samples_clean, 5),
                'q95': np.percentile(samples_clean, 95),
            }
        else:
            statistics = {'mean': 0.0, 'median': 0.0, 'std': 0.0, 'q05': 0.0, 'q95': 0.0}
    else:
        statistics = None

    return samples, statistics


# ==============================================================================
# PLOT HELPER FUNCTIONS
# ==============================================================================

def format_param_label(param_name):
    if param_name in ('mass1', 'mass2'):
        return f'{param_name} ($M_\\odot$)'
    if param_name == 'chirp_mass':
        return r'$\mathcal{M}$ ($M_\odot$)'
    if param_name == 'chi_eff':
        return r'$\chi_{\mathrm{eff}}$'
    if param_name == 'chi_a':
        return r'$\chi_a$'
    if param_name == 'q':
        return r'$q$'
    if param_name == 'distance':
        return 'distance (Mpc)'
    if param_name == 'coa_phase':
        return r'$\phi_c$ (rad)'
    if param_name == 'lambda_g':
        return r'$\lambda_g$ (m)'
    return f'{param_name}'


def format_legend(param_name, value):
    if not np.isfinite(value):
        return 'nan'
    if param_name == 'lambda_g':
        return f'{value:.3g}'
    if param_name in ('q', 'chi_eff', 'chi_a', 'spin1z', 'spin2z'):
        return f'{value:.3f}'
    if param_name == 'coa_phase':
        return f'{value:.3f}'
    return f'{value:.2f}'


def plot_posterior_row(axes_row, samples, true_vals, param_names_list, color, row_title_prefix):
    """Plot a single row of posterior histograms."""
    for pidx, pname in enumerate(param_names_list):
        ax = axes_row[pidx]
        ps = samples[:, pidx]
        ps_clean = ps[~np.isnan(ps)]
        tv = true_vals[pidx]
        tv_label = format_legend(pname, tv)
        if len(ps_clean) > 0:
            ax.hist(ps_clean, bins=50, alpha=0.7, color=color, edgecolor='black', density=True)
            if np.isfinite(tv):
                ax.axvline(tv, color='red', linestyle='--', linewidth=2, label=f'True: {tv_label}')
            pmean = np.mean(ps_clean)
            ax.axvline(pmean, color='orange', linestyle='-', linewidth=2, label='Inferred')
            pstd = np.std(ps_clean)
            ax.axvline(pmean + pstd, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label='±1σ')
            ax.axvline(pmean - pstd, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
            plabel = format_param_label(pname)
            ax.set_xlabel(f'{plabel} Value', fontsize=9)
            ax.set_ylabel('Density', fontsize=9)
            ax.set_title(f'{plabel}\n{row_title_prefix}', fontsize=10, fontweight='bold')
            ax.legend(fontsize=7, loc='upper right')
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=10, color='red', fontweight='bold')
            ax.set_xticks([]); ax.set_yticks([])


def fill_empty_row(axes_row, num_cols, msg='Real GW data\nnot available'):
    """Gray out a row of axes with a centered message."""
    for c in range(num_cols):
        axes_row[c].text(0.5, 0.5, msg, ha='center', va='center',
                         transform=axes_row[c].transAxes, fontsize=10, color='red', fontweight='bold')
        axes_row[c].set_xticks([]); axes_row[c].set_yticks([])


def _contour_levels(z_grid, credible_fracs=(0.5, 0.9)):
    """Return density thresholds that enclose credible_fracs of the probability."""
    z_sorted = np.sort(z_grid.ravel())[::-1]
    cumsum = np.cumsum(z_sorted)
    cumsum /= cumsum[-1]
    levels = []
    for frac in sorted(credible_fracs):
        idx = np.searchsorted(cumsum, frac)
        idx = min(idx, len(z_sorted) - 1)
        levels.append(z_sorted[idx])
    return sorted(levels)


def make_corner_plot(samples_dict, corner_param_names, true_values_dict=None,
                     title='', filename='corner.png'):
    """
    Create a corner plot.

    Parameters
    ----------
    samples_dict : dict
        {label: (samples_array [N, n_params], colour_string)}
    corner_param_names : list[str]
    true_values_dict : dict or None
        {label: array-like of true values}
    """
    n = len(corner_param_names)
    fig, axes = plt.subplots(n, n, figsize=(3.2 * n, 3.2 * n))
    if n == 1:
        axes = np.array([[axes]])

    for i in range(n):
        for j in range(i + 1, n):
            axes[i, j].set_visible(False)

    for label, (samples, color) in samples_dict.items():
        for i in range(n):
            ax = axes[i, i]
            data = samples[:, i]
            data = data[~np.isnan(data)]
            if len(data) < 20:
                continue
            try:
                kde1d = stats.gaussian_kde(data)
                xmin, xmax = np.percentile(data, [0.5, 99.5])
                xs = np.linspace(xmin, xmax, 300)
                ys = kde1d(xs)
                ax.plot(xs, ys, color=color, lw=1.6, label=label)
                ax.fill_between(xs, ys, alpha=0.15, color=color)
            except Exception:
                ax.hist(data, bins=50, density=True, alpha=0.4, color=color, label=label)

            for j in range(i):
                ax2 = axes[i, j]
                xdata = samples[:, j]
                ydata = samples[:, i]
                mask = ~(np.isnan(xdata) | np.isnan(ydata))
                xdata = xdata[mask]
                ydata = ydata[mask]
                if len(xdata) < 50:
                    continue
                try:
                    kde2d = stats.gaussian_kde(np.vstack([xdata, ydata]))
                    xmin, xmax = np.percentile(xdata, [0.5, 99.5])
                    ymin, ymax = np.percentile(ydata, [0.5, 99.5])
                    xx = np.linspace(xmin, xmax, 80)
                    yy = np.linspace(ymin, ymax, 80)
                    XX, YY = np.meshgrid(xx, yy)
                    ZZ = kde2d(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
                    levels = _contour_levels(ZZ, (0.5, 0.9))
                    ax2.contour(XX, YY, ZZ, levels=levels, colors=[color], linewidths=1.2, alpha=0.85)
                    ax2.contourf(XX, YY, ZZ, levels=[levels[0], levels[1], ZZ.max()],
                                 colors=[color, color], alpha=0.25)
                except Exception as e:
                    print(f"  [corner] 2D KDE failed for ({corner_param_names[j]}, {corner_param_names[i]}): {e}")
                    try:
                        ax2.hist2d(xdata, ydata, bins=40, cmap='Greens', alpha=0.6)
                    except Exception:
                        pass

    if true_values_dict is not None:
        for label, tv_arr in true_values_dict.items():
            for i in range(n):
                axes[i, i].axvline(tv_arr[i], color='black', ls='--', lw=1.0, alpha=0.7)
                for j in range(i):
                    axes[i, j].axvline(tv_arr[j], color='black', ls='--', lw=0.8, alpha=0.5)
                    axes[i, j].axhline(tv_arr[i], color='black', ls='--', lw=0.8, alpha=0.5)

    for i in range(n):
        axes[n - 1, i].set_xlabel(format_param_label(corner_param_names[i]), fontsize=11)
        if i > 0:
            axes[i, 0].set_ylabel(format_param_label(corner_param_names[i]), fontsize=11)
        if i < n - 1:
            axes[i, i].set_xticklabels([])
        for j in range(i):
            if j > 0:
                axes[i, j].set_yticklabels([])
            if i < n - 1:
                axes[i, j].set_xticklabels([])
        axes[i, i].set_yticks([])
        axes[i, i].tick_params(labelsize=8)
        for j in range(i):
            axes[i, j].tick_params(labelsize=8)

    handles, labels_leg = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels_leg, loc='upper right', fontsize=10, frameon=True, framealpha=0.9)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    try:
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  ✓ Corner plot saved to: {filename}")
    except Exception as e:
        print(f"  ✗ Failed to save corner plot: {e}")
        import traceback; traceback.print_exc()
    plt.close(fig)
    return filename


def _violin_x_extent_v2(verts, y_level):
    """Return (x_min, x_max) of a violin polygon at y_level via linear interpolation."""
    above = verts[:, 1] >= y_level
    crossings_x = []
    for k in range(len(verts) - 1):
        if above[k] != above[k + 1]:
            y0, y1 = verts[k, 1], verts[k + 1, 1]
            t = (y_level - y0) / (y1 - y0) if y1 != y0 else 0.5
            crossings_x.append(verts[k, 0] + t * (verts[k + 1, 0] - verts[k, 0]))
    if not crossings_x:
        return None, None
    return min(crossings_x), max(crossings_x)


def _draw_violins_zscore(ax, samples, true_vals, param_names_list, palette, title):
    """Draw all parameter violins on one axes using z-score normalisation."""
    n_params = len(param_names_list)

    PASTEL_PALETTE = [
        '#FFB3BA', '#BAFFC9', '#BAE1FF', '#FFFFBA',
        '#E8BAFF', '#FFD9BA', '#C4F0C5', '#FFC8DD',
    ]

    for pidx, pname in enumerate(param_names_list):
        col_data = samples[:, pidx]
        col_clean = col_data[~np.isnan(col_data)]
        if len(col_clean) == 0:
            continue

        p_mean = np.mean(col_clean)
        p_std = np.std(col_clean)
        if p_std < 1e-30:
            continue

        col_z = (col_clean - p_mean) / p_std

        colour = palette[pidx % len(palette)]
        parts = ax.violinplot(col_z, positions=[pidx], showmeans=False,
                              showmedians=False, showextrema=False, widths=0.7)
        for pc in parts['bodies']:
            pc.set_facecolor(colour)
            pc.set_edgecolor('black')
            pc.set_linewidth(0.8)
            pc.set_alpha(0.85)

        body_path = parts['bodies'][0].get_paths()[0]
        verts = body_path.vertices

        xlo, xhi = _violin_x_extent_v2(verts, 0.0)
        if xlo is not None:
            ax.hlines(0.0, xlo, xhi, colors='black', linewidths=1.8, linestyles='solid')

        for sigma_z in [-1.0, 1.0]:
            slo, shi = _violin_x_extent_v2(verts, sigma_z)
            if slo is not None:
                ax.hlines(sigma_z, slo, shi, colors='black', linewidths=1.2, linestyles='dashed')

        true_val = true_vals[pidx] if pidx < len(true_vals) else np.nan
        if np.isfinite(true_val):
            true_z = (true_val - p_mean) / p_std
            tlo, thi = _violin_x_extent_v2(verts, true_z)
            if tlo is not None:
                ax.hlines(true_z, tlo, thi, colors='red', linewidths=1.5, linestyles='dashed')
            else:
                ax.hlines(true_z, pidx - 0.35, pidx + 0.35,
                          colors='red', linewidths=1.5, linestyles='dashed')

    ax.axhline(0.0, color='black', linewidth=0.5, linestyle='solid', alpha=0.25, zorder=0)
    ax.axhline(1.0, color='black', linewidth=0.5, linestyle='dashed', alpha=0.25, zorder=0)
    ax.axhline(-1.0, color='black', linewidth=0.5, linestyle='dashed', alpha=0.25, zorder=0)

    ax.plot([], [], color='black', linestyle='solid', linewidth=1.8, label='Mean (= 0)')
    ax.plot([], [], color='black', linestyle='dashed', linewidth=1.2, label='±1σ (= ±1)')
    ax.plot([], [], color='red', linestyle='dashed', linewidth=1.5, label='True value')

    ax.set_xticks(range(n_params))
    ax.set_xticklabels([format_param_label(p) for p in param_names_list], fontsize=10)
    ax.set_ylabel('Deviation from posterior mean (σ)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.3)


def _plot_coverage_subplot(ax, param_coverages, param_list, gaussian_cov, sigma_thresh, title, n_samples_label):
    """Helper to populate one sigma-coverage subplot."""
    colors_sub = plt.cm.tab10(np.linspace(0, 1, max(len(param_list), 1)))
    for idx, param in enumerate(param_list):
        if param in param_coverages:
            ax.plot(sigma_thresh, param_coverages[param] * 100, label=param,
                    color=colors_sub[idx], alpha=0.8)
    ax.plot(sigma_thresh, gaussian_cov * 100, label='Ideal Gaussian',
            color='grey', linewidth=1.5, linestyle='--')
    for s_val in [1, 2, 3]:
        ax.axvline(x=s_val, color='grey', linewidth=0.5, linestyle=':')
    ax.set_xlabel('Confidence level (σ)', fontsize=11)
    ax.set_ylabel('Fraction within CI (%)', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8, loc='lower right')
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    if len(param_coverages) == 0:
        ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes, fontsize=24,
                ha='center', va='center', color='grey', alpha=0.5)


# ==============================================================================
# MAIN: Load Model + Inference + Plots
# ==============================================================================

if __name__ == '__main__':
    # -----------------------------------------------------------------------
    # CONFIGURATION
    # -----------------------------------------------------------------------
    DENORMALIZE_PARAMETERS = True

    # These must match the values used in generate_data.py and train_model.py
    # (they are also read from the checkpoint, but needed for MODEL_SAVE_PATH)
    PARAM_SET        = 'symmetric'
    MODIFIED_GRAVITY = True
    MODEL_PARAMS     = ['chirp_mass', 'q', 'chi_eff', 'chi_a', 'lambda_g']

    # Model architecture — must match train_model.py
    CONTEXT_DIM    = 512
    NUM_FLOW_LAYERS = 5
    HIDDEN_DIM     = 128
    EMBEDDING_TYPE = 'simple'
    NUM_EPOCHS     = 200

    # Paths
    PREPARED_DATA_PATH = 'prepared_data.pt'

    # Set this once to switch both the "_first2.csv" and "_all.csv" real-data inputs.
    REAL_DATA_CSV_PREFIX = 'test7_o4_whitened'

    def get_real_data_csv_path(csv_variant):
        base_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
        return os.path.join(base_dir, f'{REAL_DATA_CSV_PREFIX}_{csv_variant}.csv')

    # -----------------------------------------------------------------------
    # LOAD PREPARED DATA
    # -----------------------------------------------------------------------
    print(f"\nLoading prepared data from: {PREPARED_DATA_PATH}")
    if not os.path.exists(PREPARED_DATA_PATH):
        raise FileNotFoundError(
            f"'{PREPARED_DATA_PATH}' not found. Run generate_data.py first."
        )

    prepared = torch.load(PREPARED_DATA_PATH, map_location='cpu')
    pycbc_data        = prepared['train_data']
    pycbc_test_data   = prepared['test_data']
    pycbc_test_params = prepared['test_params']
    param_norm_info   = prepared['param_norm_info']
    model_param_names = prepared['param_names']
    GPS_TIME_DELAY    = prepared['gps_time_delay']

    saved_config = prepared.get('config', {})
    ADD_NOISE        = saved_config.get('add_noise', True)
    WHITEN           = saved_config.get('whiten', True)
    EXTRA_PARAM_NAMES = saved_config.get('extra_param_names', [])
    USE_REPARAMETERIZED_TARGETS = saved_config.get('use_reparameterized_targets', True)
    NUM_TRAINING_SAMPLES = saved_config.get('num_training_samples', len(pycbc_data))

    # Propagate EXTRA_PARAM_NAMES into generate_data module so transform functions work correctly
    generate_data.EXTRA_PARAM_NAMES = EXTRA_PARAM_NAMES

    PARAM_DIM = len(model_param_names)

    # -----------------------------------------------------------------------
    # CONSTRUCT MODEL SAVE PATH AND LOAD CHECKPOINT
    # -----------------------------------------------------------------------
    samples_str = f"{NUM_TRAINING_SAMPLES//1000}k" if NUM_TRAINING_SAMPLES >= 1000 else str(NUM_TRAINING_SAMPLES)
    noise_str   = "noisy" if ADD_NOISE else "clean"
    whiten_str  = "_whitened" if WHITEN else ""
    mg_str      = "_MG" if MODIFIED_GRAVITY else ""
    MODEL_SAVE_PATH = (
        f"dingo_N{samples_str}_F{NUM_FLOW_LAYERS}_C{CONTEXT_DIM}_H{HIDDEN_DIM}"
        f"_E{NUM_EPOCHS}_{EMBEDDING_TYPE}_{noise_str}{whiten_str}_{PARAM_SET}{mg_str}.pt"
    )

    print(f"Loading model checkpoint from: {MODEL_SAVE_PATH}")
    if not os.path.exists(MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"'{MODEL_SAVE_PATH}' not found. Run train_model.py first."
        )

    ckpt = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)

    # Override norm info / param names from checkpoint if present (authoritative source)
    if 'param_norm_info' in ckpt:
        param_norm_info = ckpt['param_norm_info']
    if 'model_param_names' in ckpt:
        model_param_names = ckpt['model_param_names']
    if 'gps_time_delay' in ckpt:
        GPS_TIME_DELAY = ckpt['gps_time_delay']

    ckpt_config = ckpt.get('config', {})
    CONTEXT_DIM    = ckpt_config.get('context_dim', CONTEXT_DIM)
    NUM_FLOW_LAYERS = ckpt_config.get('num_flow_layers', NUM_FLOW_LAYERS)
    HIDDEN_DIM     = ckpt_config.get('hidden_dim', HIDDEN_DIM)
    EMBEDDING_TYPE = ckpt_config.get('embedding_type', EMBEDDING_TYPE)
    NUM_EPOCHS     = ckpt_config.get('num_epochs', NUM_EPOCHS)
    PARAM_DIM      = ckpt_config.get('param_dim', PARAM_DIM)

    model = DINGOModel(
        data_dim=pycbc_data.shape[1],
        param_dim=PARAM_DIM,
        context_dim=CONTEXT_DIM,
        num_flow_layers=NUM_FLOW_LAYERS,
        hidden_dim=HIDDEN_DIM,
        device=DEVICE,
        embedding_type=EMBEDDING_TYPE,
        time_delay_value=GPS_TIME_DELAY
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()

    param_names = list(model_param_names)
    OUTPUT_COMPONENT_DISTRIBUTIONS_FROM_SYMMETRIC = USE_REPARAMETERIZED_TARGETS

    print(f"✓ Model loaded successfully")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Target parameters: {param_names}")
    print(f"  Data dimension: {pycbc_data.shape[1]}")

    # -----------------------------------------------------------------------
    # DERIVE PARAM NAME LISTS FOR PLOTTING
    # -----------------------------------------------------------------------
    _SYMMETRIC_ALL = ['chirp_mass', 'q', 'chi_eff', 'chi_a']
    _BASE_COMP = ['mass1', 'mass2', 'spin1z', 'spin2z']

    SYMMETRIC_PARAM_NAMES = list(MODEL_PARAMS)

    _CAN_CONVERT_TO_COMPONENT = (
        (PARAM_SET == 'symmetric' and {'chirp_mass', 'q', 'chi_eff', 'chi_a'}.issubset(set(MODEL_PARAMS)))
        or (PARAM_SET == 'component' and {'mass1', 'mass2', 'spin1z', 'spin2z'}.issubset(set(MODEL_PARAMS)))
    )

    if _CAN_CONVERT_TO_COMPONENT:
        _EXTRA_IN_MODEL = [p for p in MODEL_PARAMS if p not in _SYMMETRIC_ALL and p not in _BASE_COMP]
        COMPONENT_PARAM_NAMES = _BASE_COMP + _EXTRA_IN_MODEL
    else:
        COMPONENT_PARAM_NAMES = list(MODEL_PARAMS)
        if PARAM_SET == 'symmetric':
            print(f"  Note: MODEL_PARAMS does not include all 4 symmetric params — component conversion disabled")

    NUM_PLOT_COLS = max(len(SYMMETRIC_PARAM_NAMES), len(COMPONENT_PARAM_NAMES))

    # -----------------------------------------------------------------------
    # REAL GW EVENT REFERENCE VALUES
    # -----------------------------------------------------------------------
    GW250114_COMPONENT_TRUE_PARAMS_SOURCE = {
        'mass1': 33.6,
        'mass2': 32.2,
        'spin1z': 0.0,
        'spin2z': 0.0,
        'coa_phase': 0.0,
        'distance': 410.0,
    }
    GW250114_REDSHIFT_OVERRIDE = None

    m1_det, m2_det, GW250114_EFFECTIVE_Z = convert_source_to_detector_frame_masses(
        GW250114_COMPONENT_TRUE_PARAMS_SOURCE['mass1'],
        GW250114_COMPONENT_TRUE_PARAMS_SOURCE['mass2'],
        redshift=GW250114_REDSHIFT_OVERRIDE,
        distance_mpc=GW250114_COMPONENT_TRUE_PARAMS_SOURCE.get('distance', None)
    )

    GW250114_COMPONENT_TRUE_PARAMS_FULL = dict(GW250114_COMPONENT_TRUE_PARAMS_SOURCE)
    GW250114_COMPONENT_TRUE_PARAMS_FULL['mass1'] = m1_det
    GW250114_COMPONENT_TRUE_PARAMS_FULL['mass2'] = m2_det
    print(
        f"GW250114 mass-frame conversion: source->detector with z={GW250114_EFFECTIVE_Z:.4f} | "
        f"m1: {GW250114_COMPONENT_TRUE_PARAMS_SOURCE['mass1']:.3f}->{m1_det:.3f}, "
        f"m2: {GW250114_COMPONENT_TRUE_PARAMS_SOURCE['mass2']:.3f}->{m2_det:.3f}"
    )
    GW250114_SYMMETRIC_TRUE_PARAMS = transform_component_dict_to_reparameterized(GW250114_COMPONENT_TRUE_PARAMS_FULL)
    GW250114_COMPONENT_TRUE_PARAMS = {
        k: v for k, v in GW250114_COMPONENT_TRUE_PARAMS_FULL.items()
        if k in ('mass1', 'mass2', 'spin1z', 'spin2z')
    }

    # -----------------------------------------------------------------------
    # OUTPUT DIRECTORY
    # -----------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_plots_dir = os.path.join("Plots", f"run_{timestamp}")
    os.makedirs(run_plots_dir, exist_ok=True)
    print(f"\nSaving plots to: {run_plots_dir}")

    # -----------------------------------------------------------------------
    # INFERENCE ON 1000 TEST SAMPLES
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TESTING PYCBC PARAMETER INFERENCE - 1000 SAMPLES")
    print("=" * 80)

    num_test_samples = min(1000, len(pycbc_test_data))
    test_indices = list(range(num_test_samples))

    _, eval_param_names = convert_samples_to_eval_space(
        np.zeros((1, len(param_names)), dtype=np.float32),
        param_names,
        output_component_distributions=OUTPUT_COMPONENT_DISTRIBUTIONS_FROM_SYMMETRIC
    )

    mean_errors = {param: [] for param in eval_param_names}
    mean_differences = {param: [] for param in eval_param_names}
    sigma_deviations = {param: [] for param in eval_param_names}

    print(f"\nInferring posteriors for {num_test_samples} test samples...")
    if DENORMALIZE_PARAMETERS:
        print("Denormalizing to physical parameter space...")
    else:
        print("Computing errors in NORMALIZED space...")

    for i, test_idx in enumerate(test_indices):
        if i % 100 == 0:
            print(f"  Processing sample {i+1}/{num_test_samples}")

        observed_data = pycbc_test_data[test_idx].numpy()
        true_params_normalized = pycbc_test_params[test_idx].numpy()

        if DENORMALIZE_PARAMETERS:
            true_params = denormalize_params(true_params_normalized, param_norm_info, param_names)
        else:
            true_params = true_params_normalized

        posterior_samples, _ = infer_with_dingo(
            model, observed_data, num_samples=10000,
            param_norm_info=param_norm_info if DENORMALIZE_PARAMETERS else None,
            param_names=param_names if DENORMALIZE_PARAMETERS else None,
            denormalize=DENORMALIZE_PARAMETERS
        )

        posterior_eval, _ = convert_samples_to_eval_space(
            posterior_samples, param_names,
            output_component_distributions=OUTPUT_COMPONENT_DISTRIBUTIONS_FROM_SYMMETRIC
        )
        true_eval, _ = convert_vector_to_eval_space(
            true_params, param_names,
            output_component_distributions=OUTPUT_COMPONENT_DISTRIBUTIONS_FROM_SYMMETRIC
        )

        for param_idx in range(len(eval_param_names)):
            param_samples = posterior_eval[:, param_idx]
            true_val = true_eval[param_idx]
            inferred_mean = np.mean(param_samples)
            error = inferred_mean - true_val
            abs_error = abs(error)

            mean_errors[eval_param_names[param_idx]].append(abs_error)
            mean_differences[eval_param_names[param_idx]].append(error)

            posterior_std = np.std(param_samples)
            z_score = abs_error / posterior_std if posterior_std > 0 else float('inf')
            sigma_deviations[eval_param_names[param_idx]].append(z_score)

    print(f"\n✓ Completed inference on {num_test_samples} samples")

    if DENORMALIZE_PARAMETERS:
        print("\nParameter Inference Summary (1000 samples in PHYSICAL SPACE):")
    else:
        print("\nParameter Inference Summary (1000 samples in NORMALIZED SPACE):")

    for param in eval_param_names:
        errors = mean_errors[param]
        print(f"\n{param}:")
        print(f"  Mean absolute error: {np.mean(errors):.4f}")
        print(f"  Std dev of errors:   {np.std(errors):.4f}")
        print(f"  Min error:           {np.min(errors):.4f}")
        print(f"  Max error:           {np.max(errors):.4f}")
        print(f"  Median error:        {np.median(errors):.4f}")

    # -----------------------------------------------------------------------
    # GENERATE POSTERIORS FOR SELECTED SAMPLES
    # -----------------------------------------------------------------------
    sample_indices = [0, 250, 500, 750, 900]
    sample_posteriors = {}
    print(f"\nGenerating posterior samples for visualization (samples {sample_indices})...")

    for sample_idx in sample_indices:
        observed_data = pycbc_test_data[sample_idx].numpy()
        posterior_samples, _ = infer_with_dingo(
            model, observed_data, num_samples=10000,
            param_norm_info=param_norm_info if DENORMALIZE_PARAMETERS else None,
            param_names=param_names if DENORMALIZE_PARAMETERS else None,
            denormalize=DENORMALIZE_PARAMETERS
        )
        sym_samples, sym_names, comp_samples, comp_names = split_samples_into_symmetric_and_component(
            posterior_samples, param_names
        )
        sample_posteriors[sample_idx] = {
            'symmetric': sym_samples,
            'component': comp_samples,
        }
        print(f"  ✓ Generated posteriors for sample {sample_idx}")

    # -----------------------------------------------------------------------
    # REAL GW EVENT INFERENCE
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    real_gw_row = None
    try:
        import pandas as pd
        csv_path = get_real_data_csv_path('first2')
        print(f"Testing against real GW event from {os.path.basename(csv_path)} (second event)...")
        gw_csv = pd.read_csv(csv_path)

        event_2_data = gw_csv[gw_csv['event_rank'] == 2].copy()
        if len(event_2_data) == 0:
            print("  ⚠ Second event not found in CSV, skipping real GW event row")
        else:
            event_name = event_2_data['event_name'].iloc[0]
            event_gps = event_2_data['gps'].iloc[0]

            h1_data = event_2_data[event_2_data['detector'] == 'H1'].sort_values('t_seconds')
            l1_data = event_2_data[event_2_data['detector'] == 'L1'].sort_values('t_seconds')

            if len(h1_data) == 0 or len(l1_data) == 0:
                print(f"  ⚠ Missing detector data for event {event_name}, skipping real GW event row")
            else:
                h1_strain_full = h1_data['whitened_strain'].values.astype(np.float32)
                l1_strain_full = l1_data['whitened_strain'].values.astype(np.float32)
                h1_times = h1_data['t_seconds'].values
                l1_times = l1_data['t_seconds'].values

                expected_length = pycbc_data.shape[1] // 2
                print(f"  Model expects {expected_length} samples per detector ({expected_length/4096:.2f}s at 4096 Hz)")
                print(f"  Real GW data has {len(h1_strain_full)} H1 samples, {len(l1_strain_full)} L1 samples")

                h1_t0_idx = np.argmin(np.abs(h1_times))
                l1_t0_idx = np.argmin(np.abs(l1_times))
                buffer_samples = int(0.1 * 4096)

                h1_end_idx = min(h1_t0_idx + buffer_samples, len(h1_strain_full))
                l1_end_idx = min(l1_t0_idx + buffer_samples, len(l1_strain_full))
                h1_start_idx = max(0, h1_end_idx - expected_length)
                l1_start_idx = max(0, l1_end_idx - expected_length)

                h1_strain = h1_strain_full[h1_start_idx:h1_start_idx + expected_length]
                l1_strain = l1_strain_full[l1_start_idx:l1_start_idx + expected_length]

                if len(h1_strain) < expected_length:
                    h1_strain = np.pad(h1_strain, (expected_length - len(h1_strain), 0), mode='constant')
                if len(l1_strain) < expected_length:
                    l1_strain = np.pad(l1_strain, (expected_length - len(l1_strain), 0), mode='constant')

                print(f"  Cropped to {len(h1_strain)} samples per detector")

                real_gw_concat = np.concatenate([h1_strain, l1_strain])
                concat_mean = np.mean(real_gw_concat)
                concat_std = np.std(real_gw_concat) + 1e-8
                real_gw_concat = (real_gw_concat - concat_mean) / concat_std

                print(f"  Running inference on {event_name} (GPS {event_gps:.1f})...")
                real_gw_posterior, _ = infer_with_dingo(
                    model, real_gw_concat, num_samples=10000,
                    param_norm_info=param_norm_info if DENORMALIZE_PARAMETERS else None,
                    param_names=param_names if DENORMALIZE_PARAMETERS else None,
                    denormalize=DENORMALIZE_PARAMETERS
                )

                real_gw_symmetric, _, real_gw_component, _ = split_samples_into_symmetric_and_component(
                    real_gw_posterior, param_names
                )

                real_gw_row = {
                    'event_name': event_name,
                    'gps': event_gps,
                    'h1_strain': h1_strain,
                    'l1_strain': l1_strain,
                    'posterior_samples': real_gw_component,
                    'symmetric_samples': real_gw_symmetric
                }
                print(f"  ✓ Inference complete for {event_name}")

    except Exception as e:
        print(f"  ⚠ Error loading/processing real GW event: {e}")
        import traceback
        traceback.print_exc()

    # -----------------------------------------------------------------------
    # SAVE PARAMETER SUMMARY TEXT FILE
    # -----------------------------------------------------------------------
    param_summary_path = os.path.join(run_plots_dir, f"Parameter_Labels_{timestamp}.txt")
    try:
        parameter_lines = [
            "DINGO RUN PARAMETER SUMMARY",
            "=" * 80,
            f"Timestamp: {timestamp}",
            f"Run plots directory: {run_plots_dir}",
            "",
            "[Training Parameters]",
            f"NUM_TRAINING_SAMPLES: {NUM_TRAINING_SAMPLES}",
            f"NUM_EPOCHS: {NUM_EPOCHS}",
            f"DENORMALIZE_PARAMETERS: {DENORMALIZE_PARAMETERS}",
            f"USE_REPARAMETERIZED_TARGETS: {USE_REPARAMETERIZED_TARGETS}",
            f"PARAM_SET: {PARAM_SET}",
            f"MODEL_PARAMS: {MODEL_PARAMS}",
            f"model_param_names (effective targets): {model_param_names}",
            f"EXTRA_PARAM_NAMES: {EXTRA_PARAM_NAMES}",
            "",
            "[Model Architecture]",
            f"MODEL_SAVE_PATH: {MODEL_SAVE_PATH}",
            f"EMBEDDING_TYPE: {EMBEDDING_TYPE}",
            f"CONTEXT_DIM: {CONTEXT_DIM}",
            f"NUM_FLOW_LAYERS: {NUM_FLOW_LAYERS}",
            f"HIDDEN_DIM: {HIDDEN_DIM}",
            f"PARAM_DIM: {PARAM_DIM}",
            f"Input waveform dimension (H1+L1 concatenated): {pycbc_data.shape[1]}",
            f"Total model parameters: {sum(p.numel() for p in model.parameters())}",
            f"SYMMETRIC_PARAM_NAMES: {SYMMETRIC_PARAM_NAMES}",
            f"COMPONENT_PARAM_NAMES: {COMPONENT_PARAM_NAMES}",
            "",
            "[Waveform Generation Parameters]",
            f"MODIFIED_GRAVITY: {MODIFIED_GRAVITY}",
            f"ADD_NOISE: {ADD_NOISE}",
            f"WHITEN: {WHITEN}",
            f"GPS_TIME_DELAY (s): {GPS_TIME_DELAY}",
        ]
        with open(param_summary_path, 'w') as f:
            f.write("\n".join(parameter_lines) + "\n")
        print(f"Saved parameter summary: {param_summary_path}")
    except Exception as e:
        print(f"⚠ Failed to save parameter summary: {e}")

    # -----------------------------------------------------------------------
    # PLOT 1: Main inference grid (error histograms + posterior rows)
    # -----------------------------------------------------------------------
    if PARAM_SET == 'component' or not _CAN_CONVERT_TO_COMPONENT:
        num_plot_rows = 1 + len(sample_indices) + 1
    else:
        num_plot_rows = 1 + (2 * len(sample_indices)) + 1

    fig = plt.figure(figsize=(NUM_PLOT_COLS * 5, max(28, 3.8 * num_plot_rows)))
    gs = gridspec.GridSpec(num_plot_rows, NUM_PLOT_COLS, figure=fig, hspace=0.45, wspace=0.35)
    axes = [[fig.add_subplot(gs[row, col]) for col in range(NUM_PLOT_COLS)] for row in range(num_plot_rows)]
    axes = np.array(axes)

    total_params = sum(p.numel() for p in model.parameters())
    model_info_text = (
        f"Model Architecture:\n"
        f"  Embedding: {model.embedding_type.upper()}\n"
        f"  Flow Layers: {len(model.flow.layers)}\n"
        f"  Input Dim (H1+L1): {pycbc_data.shape[1]}\n"
        f"  Context Dim: {model.flow.context_dim}\n"
        f"  Hidden Dim: {model.flow.layers[0].hidden_dim}\n"
        f"  Total Parameters: {total_params:,}\n"
        f"  Epochs: {NUM_EPOCHS}\n"
        f"  Two-Detector Mode: H1+L1 concatenated"
    )
    fig.text(0.50, 0.99, model_info_text, transform=fig.transFigure,
             fontsize=8, verticalalignment='top', horizontalalignment='center',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Row 0: inference error histograms
    for param_idx, param in enumerate(eval_param_names):
        ax = axes[0, param_idx]
        diffs = mean_differences[param]
        diffs_clean = np.array([d for d in diffs if not np.isnan(d)])
        if len(diffs_clean) > 0:
            ax.hist(diffs_clean, bins=50, alpha=0.7, color='steelblue', edgecolor='black', density=False)
            ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Perfect Inference (0)')
            ax.set_xlabel('Mean Difference (Inferred - True)', fontsize=9)
            ax.set_ylabel('Frequency', fontsize=9)
            param_label = format_param_label(param)
            ax.set_title(f'{param_label} Inference Errors\n({num_test_samples} test samples)', fontsize=10, fontweight='bold')
            ax.legend(fontsize=8, loc='upper right')
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'No valid data\n(all NaN)', ha='center', va='center',
                    transform=ax.transAxes, fontsize=10, color='red', fontweight='bold')
            ax.set_xticks([]); ax.set_yticks([])

    # Rows 1+: posterior rows per sample
    for row_idx, sample_idx in enumerate(sample_indices):
        posterior_sym  = sample_posteriors[sample_idx]['symmetric']
        posterior_comp = sample_posteriors[sample_idx]['component']
        true_params_normalized = pycbc_test_params[sample_idx].numpy()

        if DENORMALIZE_PARAMETERS:
            true_params = denormalize_params(true_params_normalized, param_norm_info, param_names)
        else:
            true_params = true_params_normalized

        true_sym, _, true_comp, _ = split_vector_into_symmetric_and_component(true_params, param_names)

        if PARAM_SET == 'symmetric' and _CAN_CONVERT_TO_COMPONENT:
            row_sym = 1 + (2 * row_idx)
            row_comp = row_sym + 1
            plot_posterior_row(axes[row_sym], posterior_sym, true_sym,
                               SYMMETRIC_PARAM_NAMES, 'darkgreen',
                               f'Symmetric Posterior (Sample {sample_idx})')
        else:
            row_comp = 1 + row_idx

        if _CAN_CONVERT_TO_COMPONENT:
            comp_origin = 'Direct' if PARAM_SET == 'component' else 'Derived'
            plot_posterior_row(axes[row_comp], posterior_comp, true_comp,
                               COMPONENT_PARAM_NAMES, 'darkgreen',
                               f'{comp_origin} Component (Sample {sample_idx})')
        else:
            plot_posterior_row(axes[row_comp], posterior_sym, true_sym,
                               SYMMETRIC_PARAM_NAMES, 'darkgreen',
                               f'Model Output (Sample {sample_idx})')

    # Final row: real GW event
    if PARAM_SET == 'component' or not _CAN_CONVERT_TO_COMPONENT:
        real_event_row = 1 + len(sample_indices)
    else:
        real_event_row = 1 + (2 * len(sample_indices))

    if _CAN_CONVERT_TO_COMPONENT:
        _gw_names = COMPONENT_PARAM_NAMES
        _gw_key   = 'posterior_samples'
        _gw_true  = GW250114_COMPONENT_TRUE_PARAMS
    else:
        _gw_names = SYMMETRIC_PARAM_NAMES
        _gw_key   = 'symmetric_samples'
        _gw_true  = GW250114_SYMMETRIC_TRUE_PARAMS

    if real_gw_row is not None:
        gw_true_arr = np.array([_gw_true.get(p, np.nan) for p in _gw_names], dtype=np.float32)
        plot_posterior_row(axes[real_event_row], real_gw_row[_gw_key], gw_true_arr,
                           _gw_names, 'purple', real_gw_row['event_name'])
    else:
        fill_empty_row(axes[real_event_row], NUM_PLOT_COLS)

    plot_filename = os.path.join(run_plots_dir, f"PyCBC_Parameter_Inference_TwoDetector_{timestamp}.png")
    try:
        fig.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"\n✓ Plot 1 saved to: {plot_filename}")
    except Exception as e:
        print(f"\n✗ Failed to save plot 1: {e}")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # PLOT 2: Best, Average, and Real GW Event
    # -----------------------------------------------------------------------
    PASTEL_GREEN  = '#77dd77'
    PASTEL_RED    = '#ff6961'
    PASTEL_PURPLE = '#b39ddb'

    std_per_param = {}
    for param in eval_param_names:
        std_per_param[param] = np.std(mean_differences[param])

    num_eval_params = len(eval_param_names)
    scores = np.zeros(num_test_samples)
    for i in range(num_test_samples):
        for param in eval_param_names:
            s = std_per_param[param]
            if s > 0:
                scores[i] += abs(mean_differences[param][i]) / s

    best_sample_idx = int(np.argmin(scores))
    avg_target = float(num_eval_params)
    avg_sample_idx = int(np.argmin(np.abs(scores - avg_target)))

    print(f"\n  Best sample index: {best_sample_idx} (score={scores[best_sample_idx]:.4f})")
    print(f"  Average sample index: {avg_sample_idx} (score={scores[avg_sample_idx]:.4f})")

    special_posteriors = {}
    for label, sidx in [('best', best_sample_idx), ('average', avg_sample_idx)]:
        obs = pycbc_test_data[sidx].numpy()
        post, _ = infer_with_dingo(
            model, obs, num_samples=10000,
            param_norm_info=param_norm_info if DENORMALIZE_PARAMETERS else None,
            param_names=param_names if DENORMALIZE_PARAMETERS else None,
            denormalize=DENORMALIZE_PARAMETERS
        )
        sym_s, _, comp_s, _ = split_samples_into_symmetric_and_component(post, param_names)

        true_norm = pycbc_test_params[sidx].numpy()
        if DENORMALIZE_PARAMETERS:
            true_p = denormalize_params(true_norm, param_norm_info, param_names)
        else:
            true_p = true_norm
        true_sym, _, true_comp, _ = split_vector_into_symmetric_and_component(true_p, param_names)

        special_posteriors[label] = {
            'sym_samples': sym_s, 'comp_samples': comp_s,
            'true_sym': true_sym, 'true_comp': true_comp, 'index': sidx,
        }
        print(f"  ✓ Posteriors generated for {label} sample (index {sidx})")

    if PARAM_SET == 'component' or not _CAN_CONVERT_TO_COMPONENT:
        fig2_num_rows = 3
    else:
        fig2_num_rows = 6

    fig2 = plt.figure(figsize=(NUM_PLOT_COLS * 5, 3.8 * fig2_num_rows))
    gs2 = gridspec.GridSpec(fig2_num_rows, NUM_PLOT_COLS, figure=fig2, hspace=0.55, wspace=0.35)
    axes2 = [[fig2.add_subplot(gs2[r, c]) for c in range(NUM_PLOT_COLS)] for r in range(fig2_num_rows)]
    axes2 = np.array(axes2)

    best = special_posteriors['best']
    avg  = special_posteriors['average']

    if PARAM_SET == 'component' or not _CAN_CONVERT_TO_COMPONENT:
        _fig2_names = COMPONENT_PARAM_NAMES if PARAM_SET == 'component' else SYMMETRIC_PARAM_NAMES
        _fig2_best_samples = best['comp_samples'] if PARAM_SET == 'component' else best['sym_samples']
        _fig2_best_true    = best['true_comp']    if PARAM_SET == 'component' else best['true_sym']
        _fig2_avg_samples  = avg['comp_samples']  if PARAM_SET == 'component' else avg['sym_samples']
        _fig2_avg_true     = avg['true_comp']     if PARAM_SET == 'component' else avg['true_sym']
        _fig2_gw_true_dict = GW250114_COMPONENT_TRUE_PARAMS if PARAM_SET == 'component' else GW250114_SYMMETRIC_TRUE_PARAMS
        _fig2_suffix = 'Direct Component' if PARAM_SET == 'component' else 'Symmetric'
        _fig2_gw_key = 'posterior_samples' if PARAM_SET == 'component' else 'symmetric_samples'

        plot_posterior_row(axes2[0], _fig2_best_samples, _fig2_best_true,
                           _fig2_names, PASTEL_GREEN, f'Best Sample (idx {best["index"]}) – {_fig2_suffix}')
        plot_posterior_row(axes2[1], _fig2_avg_samples, _fig2_avg_true,
                           _fig2_names, PASTEL_RED, f'Average Sample (idx {avg["index"]}) – {_fig2_suffix}')
        if real_gw_row is not None:
            _gw_true2 = np.array([_fig2_gw_true_dict.get(p, np.nan) for p in _fig2_names], dtype=np.float32)
            plot_posterior_row(axes2[2], real_gw_row[_fig2_gw_key], _gw_true2,
                               _fig2_names, PASTEL_PURPLE, f'{real_gw_row["event_name"]} – {_fig2_suffix}')
        else:
            fill_empty_row(axes2[2], NUM_PLOT_COLS)
    else:
        plot_posterior_row(axes2[0], best['sym_samples'], best['true_sym'],
                           SYMMETRIC_PARAM_NAMES, PASTEL_GREEN, f'Best Sample (idx {best["index"]}) – Symmetric')
        plot_posterior_row(axes2[1], best['comp_samples'], best['true_comp'],
                           COMPONENT_PARAM_NAMES, PASTEL_GREEN, f'Best Sample (idx {best["index"]}) – Derived Component')
        plot_posterior_row(axes2[2], avg['sym_samples'], avg['true_sym'],
                           SYMMETRIC_PARAM_NAMES, PASTEL_RED, f'Average Sample (idx {avg["index"]}) – Symmetric')
        plot_posterior_row(axes2[3], avg['comp_samples'], avg['true_comp'],
                           COMPONENT_PARAM_NAMES, PASTEL_RED, f'Average Sample (idx {avg["index"]}) – Derived Component')
        if real_gw_row is not None:
            gw_sym_true = np.array([GW250114_SYMMETRIC_TRUE_PARAMS.get(p, np.nan) for p in SYMMETRIC_PARAM_NAMES], dtype=np.float32)
            gw_comp_true = np.array([GW250114_COMPONENT_TRUE_PARAMS.get(p, np.nan) for p in COMPONENT_PARAM_NAMES], dtype=np.float32)
            plot_posterior_row(axes2[4], real_gw_row['symmetric_samples'], gw_sym_true,
                               SYMMETRIC_PARAM_NAMES, PASTEL_PURPLE, f'{real_gw_row["event_name"]} – Symmetric')
            plot_posterior_row(axes2[5], real_gw_row['posterior_samples'], gw_comp_true,
                               COMPONENT_PARAM_NAMES, PASTEL_PURPLE, f'{real_gw_row["event_name"]} – Derived Component')
        else:
            fill_empty_row(axes2[4], NUM_PLOT_COLS)
            fill_empty_row(axes2[5], NUM_PLOT_COLS)

    plot_filename2 = os.path.join(run_plots_dir, f"PyCBC_BestAvgReal_{timestamp}.png")
    try:
        fig2.savefig(plot_filename2, dpi=150, bbox_inches='tight')
        print(f"\n✓ Plot 2 saved to: {plot_filename2}")
    except Exception as e:
        print(f"\n✗ Failed to save plot 2: {e}")
    plt.close(fig2)

    # -----------------------------------------------------------------------
    # PLOT 3: All real whitened events summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("GENERATING PLOT 3: All real whitened events summary")
    print("=" * 80)

    all_real_plot_filename = os.path.join(run_plots_dir, f"PyCBC_AllRealWhitened_{PARAM_SET}_{timestamp}.png")

    try:
        import pandas as pd

        all_csv_path = get_real_data_csv_path('all')
        all_gw_csv = pd.read_csv(all_csv_path)
        grouped = all_gw_csv.groupby(['event_rank', 'event_name', 'gps'], sort=True)

        if PARAM_SET == 'component' or not _CAN_CONVERT_TO_COMPONENT:
            all_real_param_names = COMPONENT_PARAM_NAMES if PARAM_SET == 'component' else SYMMETRIC_PARAM_NAMES
            all_real_key = 'posterior_samples' if PARAM_SET == 'component' else 'symmetric_samples'
            all_real_title_suffix = 'Component' if PARAM_SET == 'component' else 'Symmetric'
        else:
            all_real_param_names = COMPONENT_PARAM_NAMES
            all_real_key = 'posterior_samples'
            all_real_title_suffix = 'Derived Component'

        expected_length = pycbc_data.shape[1] // 2
        all_real_results = []
        all_real_num_samples = 5000

        for (event_rank, event_name, event_gps), g in grouped:
            h1_data = g[g['detector'] == 'H1'].sort_values('t_seconds')
            l1_data = g[g['detector'] == 'L1'].sort_values('t_seconds')

            if len(h1_data) == 0 or len(l1_data) == 0:
                print(f"  ⚠ Skipping {event_name}: missing H1/L1 detector data")
                continue

            h1_strain_full = h1_data['whitened_strain'].values.astype(np.float32)
            l1_strain_full = l1_data['whitened_strain'].values.astype(np.float32)
            h1_times = h1_data['t_seconds'].values
            l1_times = l1_data['t_seconds'].values

            h1_t0_idx = np.argmin(np.abs(h1_times))
            l1_t0_idx = np.argmin(np.abs(l1_times))
            buffer_samples = int(0.1 * 4096)

            h1_end_idx = min(h1_t0_idx + buffer_samples, len(h1_strain_full))
            l1_end_idx = min(l1_t0_idx + buffer_samples, len(l1_strain_full))
            h1_start_idx = max(0, h1_end_idx - expected_length)
            l1_start_idx = max(0, l1_end_idx - expected_length)

            h1_strain = h1_strain_full[h1_start_idx:h1_start_idx + expected_length]
            l1_strain = l1_strain_full[l1_start_idx:l1_start_idx + expected_length]

            if len(h1_strain) < expected_length:
                h1_strain = np.pad(h1_strain, (expected_length - len(h1_strain), 0), mode='constant')
            if len(l1_strain) < expected_length:
                l1_strain = np.pad(l1_strain, (expected_length - len(l1_strain), 0), mode='constant')

            real_concat = np.concatenate([h1_strain, l1_strain])
            real_mean = np.mean(real_concat)
            real_std = np.std(real_concat) + 1e-8
            real_concat = (real_concat - real_mean) / real_std

            try:
                posterior, _ = infer_with_dingo(
                    model, real_concat, num_samples=all_real_num_samples,
                    param_norm_info=param_norm_info if DENORMALIZE_PARAMETERS else None,
                    param_names=param_names if DENORMALIZE_PARAMETERS else None,
                    denormalize=DENORMALIZE_PARAMETERS
                )
                sym_s, _, comp_s, _ = split_samples_into_symmetric_and_component(posterior, param_names)
                all_real_results.append({
                    'event_rank': int(event_rank),
                    'event_name': str(event_name),
                    'posterior_samples': comp_s,
                    'symmetric_samples': sym_s,
                })
                print(f"  ✓ Inference complete: {event_name} (rank {int(event_rank)})")
            except Exception as infer_err:
                print(f"  ⚠ Inference failed for {event_name}: {infer_err}")

        if len(all_real_results) == 0:
            print("  ⚠ No valid all-event inferences; skipping plot 3")
        else:
            num_params_all = len(all_real_param_names)

            stats_candidates = [
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gw_events_stats.csv'),
                'gw_events_stats.csv',
            ]
            stats_path = next((p for p in stats_candidates if os.path.exists(p)), None)
            actual_by_event = {}
            if stats_path is not None:
                try:
                    stats_df = pd.read_csv(stats_path)
                    for _, row in stats_df.iterrows():
                        event_name_key = str(row.get('event', ''))
                        if not event_name_key:
                            continue
                        m1_src = row.get('mass_1_source', np.nan)
                        m2_src = row.get('mass_2_source', np.nan)
                        z_val = row.get('redshift', np.nan)
                        d_l_mpc = row.get('luminosity_distance', np.nan)
                        m1_d = np.nan; m2_d = np.nan
                        if np.isfinite(m1_src) and np.isfinite(m2_src):
                            try:
                                z_in = float(z_val) if np.isfinite(z_val) else None
                                d_in = float(d_l_mpc) if np.isfinite(d_l_mpc) else 410.0
                                m1_d, m2_d, _ = convert_source_to_detector_frame_masses(
                                    m1_src, m2_src, redshift=z_in, distance_mpc=d_in
                                )
                            except Exception:
                                pass
                        s1_val = np.nan; s2_val = np.nan
                        for c in ('spin1z', 'spin_1z', 'chi1z'):
                            if c in row and np.isfinite(row[c]):
                                s1_val = float(row[c]); break
                        for c in ('spin2z', 'spin_2z', 'chi2z'):
                            if c in row and np.isfinite(row[c]):
                                s2_val = float(row[c]); break
                        actual_by_event[event_name_key] = {
                            'mass1': m1_d, 'mass2': m2_d, 'spin1z': s1_val, 'spin2z': s2_val,
                        }
                except Exception as stats_err:
                    print(f"  ⚠ Could not load gw_events_stats.csv: {stats_err}")
            else:
                print("  ⚠ gw_events_stats.csv not found; actual-value overlay disabled")

            fig_width = max(24.0, 0.42 * len(all_real_results) + 12.0)
            fig_height = max(3.5 * num_params_all, 7.0)
            fig_all, axes_all = plt.subplots(
                num_params_all, 1, figsize=(fig_width, fig_height),
                squeeze=False, sharex=True,
            )
            axes_flat_all = axes_all.flatten()
            event_labels = [f"{r['event_rank']}: {r['event_name']}" for r in all_real_results]
            x = np.arange(len(all_real_results))

            for p_idx, pname in enumerate(all_real_param_names):
                ax = axes_flat_all[p_idx]
                med = []; q05 = []; q95 = []
                for r in all_real_results:
                    arr = r[all_real_key][:, p_idx]
                    med.append(np.median(arr))
                    q05.append(np.percentile(arr, 5))
                    q95.append(np.percentile(arr, 95))
                med = np.array(med); q05 = np.array(q05); q95 = np.array(q95)
                yerr = np.vstack([med - q05, q95 - med])
                ax.errorbar(x, med, yerr=yerr, fmt='o', capsize=3, markersize=4,
                            linewidth=1.2, color='tab:blue', ecolor='tab:blue')
                actual_vals = np.array([
                    actual_by_event.get(r['event_name'], {}).get(pname, np.nan)
                    for r in all_real_results
                ], dtype=np.float64)
                valid_actual = np.isfinite(actual_vals)
                if np.any(valid_actual):
                    ax.scatter(x[valid_actual], actual_vals[valid_actual],
                               marker='D', s=26, color='tab:orange', edgecolors='black',
                               linewidths=0.35, label='Actual value', zorder=3)
                elif pname in ('spin1z', 'spin2z'):
                    ax.text(0.01, 0.93, 'Actual spin values unavailable in gw_events_stats.csv',
                            transform=ax.transAxes, fontsize=8, color='tab:orange', ha='left', va='top')
                ax.set_title(format_param_label(pname), fontsize=10, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(event_labels, rotation=65, ha='right', fontsize=8)
                ax.tick_params(axis='y', labelsize=8)
                ax.grid(True, alpha=0.25)
                ax.legend(loc='best', fontsize=8)

            fig_all.suptitle(
                f'All Real Whitened Events: Posterior Median +/- 90% CI + Actual Values ({all_real_title_suffix} params)',
                fontsize=13, fontweight='bold',
            )
            fig_all.tight_layout(rect=[0, 0.03, 1, 0.965])
            fig_all.savefig(all_real_plot_filename, dpi=150, bbox_inches='tight')
            plt.close(fig_all)
            print(f"  ✓ Plot 3 saved to: {all_real_plot_filename}")

    except Exception as e:
        print(f"  ⚠ Failed to generate plot 3: {e}")
        import traceback
        traceback.print_exc()

    # -----------------------------------------------------------------------
    # PLOT 4: Corner plot (2D posteriors)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("GENERATING PLOT 4: Corner plot (2D posteriors)")
    print("=" * 80)

    if PARAM_SET == 'component' and _CAN_CONVERT_TO_COMPONENT:
        CORNER_PARAM_NAMES = list(COMPONENT_PARAM_NAMES)
        corner_samples_dict = {f'Best (idx {best["index"]})': (best['comp_samples'], PASTEL_GREEN)}
        corner_true_dict    = {f'Best (idx {best["index"]})': best['true_comp']}
    else:
        CORNER_PARAM_NAMES = list(SYMMETRIC_PARAM_NAMES)
        corner_samples_dict = {f'Best (idx {best["index"]})': (best['sym_samples'], PASTEL_GREEN)}
        corner_true_dict    = {f'Best (idx {best["index"]})': best['true_sym']}

    corner_origin = 'direct' if PARAM_SET == 'component' else PARAM_SET
    plot_filename3 = os.path.join(run_plots_dir, f"PyCBC_CornerPlot_{PARAM_SET}_{timestamp}.png")
    make_corner_plot(
        corner_samples_dict, CORNER_PARAM_NAMES,
        true_values_dict=corner_true_dict,
        title=f'Best Sample – 2D Posterior Corner Plot ({corner_origin} parameters, {NUM_EPOCHS} epochs)',
        filename=plot_filename3
    )

    # -----------------------------------------------------------------------
    # PLOT 5: Sigma coverage plot
    # -----------------------------------------------------------------------
    sigma_thresholds = np.linspace(0, 5, 500)
    gaussian_coverage = np.array([2 * scipy_norm.cdf(s) - 1 for s in sigma_thresholds])

    gr_params = [p for p in eval_param_names if p != 'lambda_g']
    mg_params = [p for p in eval_param_names if p == 'lambda_g']

    sim_coverage_gr = {}
    for param in gr_params:
        z_scores = np.array(sigma_deviations[param])
        sim_coverage_gr[param] = np.array([np.mean(z_scores <= s) for s in sigma_thresholds])

    sim_coverage_mg = {}
    for param in mg_params:
        z_scores = np.array(sigma_deviations[param])
        sim_coverage_mg[param] = np.array([np.mean(z_scores <= s) for s in sigma_thresholds])

    real_sigma_deviations = {}
    real_coverage_gr = {}
    real_n_events = 0

    try:
        real_catalog_sym = {}
        stats_candidates_5 = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gw_events_stats.csv'),
            'gw_events_stats.csv',
        ]
        stats_path_5 = next((p for p in stats_candidates_5 if os.path.exists(p)), None)
        if stats_path_5 is not None:
            import pandas as pd
            stats_df_5 = pd.read_csv(stats_path_5)
            for _, row in stats_df_5.iterrows():
                ev_name = str(row.get('event', ''))
                if not ev_name:
                    continue
                m1_src = row.get('mass_1_source', np.nan)
                m2_src = row.get('mass_2_source', np.nan)
                z_val = row.get('redshift', np.nan)
                d_l = row.get('luminosity_distance', np.nan)
                chi_eff_cat = row.get('chi_eff', np.nan)
                if np.isfinite(m1_src) and np.isfinite(m2_src):
                    z_in = float(z_val) if np.isfinite(z_val) else None
                    d_in = float(d_l) if np.isfinite(d_l) else 410.0
                    try:
                        m1_det_c, m2_det_c, z_used = convert_source_to_detector_frame_masses(
                            m1_src, m2_src, redshift=z_in, distance_mpc=d_in
                        )
                        q_cat = min(m1_det_c, m2_det_c) / max(m1_det_c, m2_det_c) if max(m1_det_c, m2_det_c) > 0 else np.nan
                        mc_det_c = (m1_det_c * m2_det_c) ** 0.6 / (m1_det_c + m2_det_c) ** 0.2 if (m1_det_c + m2_det_c) > 0 else np.nan
                        real_catalog_sym[ev_name] = {
                            'chirp_mass': mc_det_c, 'q': q_cat,
                            'chi_eff': float(chi_eff_cat) if np.isfinite(chi_eff_cat) else np.nan,
                            'mass1': m1_det_c, 'mass2': m2_det_c,
                        }
                    except Exception:
                        pass

        if 'all_real_results' in locals():
            for res in all_real_results:
                ev = res['event_name']
                if ev not in real_catalog_sym:
                    continue
                cat = real_catalog_sym[ev]
                sym_post = res.get('symmetric_samples')
                comp_post = res.get('posterior_samples')
                for param in gr_params:
                    true_val = cat.get(param, np.nan)
                    if not np.isfinite(true_val):
                        continue
                    if param in ['chirp_mass', 'q', 'chi_eff', 'chi_a']:
                        sym_names = ['chirp_mass', 'q', 'chi_eff', 'chi_a'] + [p for p in EXTRA_PARAM_NAMES if p != 'lambda_g']
                        if param in sym_names and sym_post is not None and sym_post.shape[1] > sym_names.index(param):
                            ps = sym_post[:, sym_names.index(param)]
                        else:
                            continue
                    elif param in ['mass1', 'mass2', 'spin1z', 'spin2z']:
                        comp_names = ['mass1', 'mass2', 'spin1z', 'spin2z'] + [p for p in EXTRA_PARAM_NAMES if p != 'lambda_g']
                        if param in comp_names and comp_post is not None and comp_post.shape[1] > comp_names.index(param):
                            ps = comp_post[:, comp_names.index(param)]
                        else:
                            continue
                    else:
                        continue
                    ps_clean = ps[np.isfinite(ps)]
                    if len(ps_clean) < 10:
                        continue
                    post_mean = np.mean(ps_clean)
                    post_std = np.std(ps_clean)
                    if post_std > 0:
                        z = abs(post_mean - true_val) / post_std
                        real_sigma_deviations.setdefault(param, []).append(z)

            for param in gr_params:
                if param in real_sigma_deviations and len(real_sigma_deviations[param]) > 0:
                    z_arr = np.array(real_sigma_deviations[param])
                    real_coverage_gr[param] = np.array([np.mean(z_arr <= s) for s in sigma_thresholds])
            real_n_events = max((len(v) for v in real_sigma_deviations.values()), default=0)

    except Exception as real_cov_err:
        print(f"  ⚠ Could not compute real-event sigma coverage: {real_cov_err}")

    fig5, axes5 = plt.subplots(2, 2, figsize=(16, 12))
    _plot_coverage_subplot(axes5[0, 0], sim_coverage_gr, gr_params, gaussian_coverage, sigma_thresholds,
                           f'GR Parameters – Simulated ({num_test_samples} samples)', num_test_samples)
    _plot_coverage_subplot(axes5[0, 1], sim_coverage_mg, mg_params, gaussian_coverage, sigma_thresholds,
                           f'Modified GR – Simulated ({num_test_samples} samples)', num_test_samples)
    _plot_coverage_subplot(axes5[1, 0], real_coverage_gr, gr_params, gaussian_coverage, sigma_thresholds,
                           f'GR Parameters – Real Events ({real_n_events} events)', real_n_events)
    _plot_coverage_subplot(axes5[1, 1], {}, mg_params, gaussian_coverage, sigma_thresholds,
                           f'Modified GR – Real Events (no true λ_g)', 0)

    fig5.suptitle(f'Sigma Coverage – {NUM_EPOCHS} Epochs', fontsize=14, fontweight='bold')
    fig5.tight_layout(rect=[0, 0, 1, 0.96])
    plot_filename5 = os.path.join(run_plots_dir, f"PyCBC_SigmaCoverage_{PARAM_SET}_{timestamp}.png")
    fig5.savefig(plot_filename5, dpi=150, bbox_inches='tight')
    plt.close(fig5)
    print(f"\n✓ Plot 5 (sigma coverage) saved to: {plot_filename5}")

    # Print sigma coverage statistics
    print(f"\n{'='*65}")
    print(f"SIGMA COVERAGE STATISTICS – SIMULATED ({num_test_samples} test samples)")
    print(f"{'='*65}")
    print(f"{'Parameter':<15} {'1σ':>8} {'2σ':>8} {'3σ':>8} {'Median z':>10}")
    print(f"{'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    for param in eval_param_names:
        z_arr = np.array(sigma_deviations[param])
        c1 = np.mean(z_arr <= 1) * 100
        c2 = np.mean(z_arr <= 2) * 100
        c3 = np.mean(z_arr <= 3) * 100
        med_z = np.median(z_arr)
        print(f"{param:<15} {c1:>7.1f}% {c2:>7.1f}% {c3:>7.1f}% {med_z:>10.3f}")
    print(f"\nIdeal Gaussian:  {68.3:>7.1f}% {95.4:>7.1f}% {99.7:>7.1f}%")
    print(f"{'='*65}")

    # -----------------------------------------------------------------------
    # PLOT 6: Violin plot of posterior distributions
    # -----------------------------------------------------------------------
    print("\nGenerating violin plot of posterior distributions (z-score normalised)...")

    PASTEL_PALETTE = [
        '#FFB3BA', '#BAFFC9', '#BAE1FF', '#FFFFBA',
        '#E8BAFF', '#FFD9BA', '#C4F0C5', '#FFC8DD',
    ]

    best_post = special_posteriors['best']
    violin_names = list(SYMMETRIC_PARAM_NAMES)
    violin_n_params = len(violin_names)

    fig_violin, (ax_sim, ax_real) = plt.subplots(
        2, 1, figsize=(max(3.5 * violin_n_params, 12), 12), constrained_layout=True
    )

    _draw_violins_zscore(ax_sim,
                         best_post['sym_samples'], best_post['true_sym'],
                         violin_names, PASTEL_PALETTE,
                         f'Simulated – Best Sample (idx {best_post["index"]})')

    if real_gw_row is not None:
        gw_sym_true = np.array([GW250114_SYMMETRIC_TRUE_PARAMS.get(p, np.nan)
                                for p in violin_names], dtype=np.float32)
        _draw_violins_zscore(ax_real,
                             real_gw_row['symmetric_samples'], gw_sym_true,
                             violin_names, PASTEL_PALETTE,
                             f'Real Event – {real_gw_row["event_name"]}')
    else:
        ax_real.text(0.5, 0.5, 'Real GW data not available', ha='center', va='center',
                     transform=ax_real.transAxes, fontsize=14, color='gray', fontweight='bold')
        ax_real.set_title('Real Event', fontsize=12, fontweight='bold')
        ax_real.set_xticks(range(violin_n_params))
        ax_real.set_xticklabels([format_param_label(p) for p in violin_names], fontsize=10)

    fig_violin.suptitle(f'Posterior Violin Plot – {NUM_EPOCHS} Epochs', fontsize=14, fontweight='bold')
    plot_filename_violin = os.path.join(run_plots_dir, f"PyCBC_ViolinPlot_{PARAM_SET}_{timestamp}.png")
    fig_violin.savefig(plot_filename_violin, dpi=150, bbox_inches='tight')
    plt.close(fig_violin)
    print(f"✓ Plot 6 (violin) saved to: {plot_filename_violin}")

    print(f"\n{'='*80}")
    print(f"All plots saved to: {run_plots_dir}")
    print(f"{'='*80}")
