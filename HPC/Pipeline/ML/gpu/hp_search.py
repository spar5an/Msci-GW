"""hp_search.py — grid hyper-parameter search over the DINGO training config.

Edit `SEARCH_SPACE` below to list candidate values per hyperparameter. The
script takes the cartesian product of those lists, passes each combination
through `run_training(config)` (from the sibling `train_model.py`), and
appends one CSV row per trial. Whether that trainer is the CPU or GPU
variant is determined by which copy of this folder you are running in —
ML/cpu/ imports the CPU trainer, ML/gpu/ imports the GPU trainer.

Usage:

    python hp_search.py                       # full grid
    python hp_search.py --max-runs 4          # only first 4 combinations
    python hp_search.py --dry-run             # list combinations, do not train
    python hp_search.py --tag smoke           # suffix on results CSV + ckpts

Every trial is wrapped in try/except — a failed run logs its traceback
to the CSV and the sweep continues.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
import time
import traceback
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[2]


# ── SEARCH SPACE ─────────────────────────────────────────────────────────────
# Each key is a DEFAULT_CONFIG key; each value is a list of candidate settings.
# Drop a list to a single value to pin that axis. Combinations are the
# cartesian product across all keys. Tuples (e.g. conv1d_num_filters) must
# be wrapped in an outer list as a single element — see example below.

SEARCH_SPACE: dict[str, list] = {
    # Input representation. 'strain' pairs with simple/conv1d/lstm; 'svd'
    # pairs with svd_mlp; 'qtransform' pairs with qtransform_conv2d.
    # prune_for_embedding drops the impossible pairings below.
    'input_representation': ['strain', 'svd', 'qtransform'],
    'embedding_type':   ['simple', 'conv1d', 'lstm',
                         'svd_mlp', 'qtransform_conv2d'],
    # Flow coupling: 'affine' (fast, less expressive) vs 'spline' (DINGO-style,
    # rational-quadratic neural spline — much tighter posteriors).
    'coupling_type':    ['affine', 'spline'],
    'context_dim':      [64, 128],
    'hidden_dim':       [64, 128],
    'num_flow_layers':  [4, 6],
    # Parameter-space reparameterisation. 'Mc_q' swaps mass1/mass2 for
    # chirp_mass and mass_ratio — usually converges faster and hits tighter
    # posteriors on the mass axis.
    'param_parameterization': ['m1_m2', 'Mc_q'],
    # Distance prior. 'volume' importance-weights the loss by d^2 so the
    # flow learns posteriors under a uniform-in-comoving-volume prior.
    'distance_prior':   ['uniform', 'volume'],
    # Per-embedding crop overrides. Pruned so each embedding only picks
    # from its own list — lstm gets the longer window, conv1d/simple the
    # tight window around merger. Base merger_crop_half_width in
    # COMMON_CONFIG stays the fallback for any embedding whose override
    # is left at None.
    'simple_crop_half_width': [100, 500],
    'conv1d_crop_half_width': [100, 500],
    'lstm_crop_half_width':   [500],
    # Embedding-internal knobs — kept to singleton lists here so they do
    # not explode the grid; widen if you have the budget.
    'lstm_hidden_dim':  [128],
    'lstm_num_layers':  [2],
    'conv1d_num_filters': [(64, 128, 256)],
    # SVD representation knobs. Only used when input_representation='svd'.
    'svd_k':            [150, 300],
    # Q-transform representation knobs. Only used when
    # input_representation='qtransform'.
    'qtransform_logfsteps':   [50],
    'qtransform_delta_t_out': [0.002],
    # Optimiser knobs.
    'learning_rate':    [1e-4],
    'batch_size':       [256],
}

# Allowed (representation, embedding) pairings. Any combo that doesn't appear
# here is dropped in prune_for_embedding.
_REPR_EMBEDDINGS: dict[str, set[str]] = {
    'strain':     {'simple', 'conv1d', 'lstm'},
    'svd':        {'svd_mlp'},
    'qtransform': {'qtransform_conv2d'},
}

# Settings shared by every run (overrides DEFAULT_CONFIG once per sweep).
COMMON_CONFIG: dict = {
    'dataset_path':           str(PIPELINE_ROOT / 'Data' / 'dataset.pt'),
    'merger_crop_half_width': 500,          # fallback when a per-embedding override is None
    'num_epochs':              20,
    'seed':                    0,
    # GPU-only knobs; safely ignored by the CPU trainer.
    'use_amp':                 True,
    'compile_model':           False,
    'drop_last_batch':         False,
    'val_chunk_size':          256,
}

# Where to write the rolling CSV (overwritten on start unless --resume-csv).
RESULTS_CSV_DEFAULT = str(PIPELINE_ROOT / 'hp_search_results.csv')


# ── HELPERS ──────────────────────────────────────────────────────────────────

def iter_combinations(space: dict) -> list[dict]:
    """Cartesian product of a dict-of-lists into a list of flat dicts."""
    keys = list(space.keys())
    values = [space[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def prune_for_embedding(cfg: dict) -> dict | None:
    """Remove HP knobs that the chosen embedding, coupling or representation
    does not use. Returns None when the (representation, embedding) pair is
    impossible — caller filters those out.

    Without this, picking `embedding_type='simple'` with different
    `lstm_hidden_dim` values would log distinct rows for identical trainings
    — and similarly for coupling-specific spline knobs and representation-
    specific SVD / Q-transform knobs.
    """
    pruned = dict(cfg)
    et = pruned.get('embedding_type')
    rep = pruned.get('input_representation', 'strain')

    # Drop impossible (representation, embedding) pairings outright.
    if et not in _REPR_EMBEDDINGS.get(rep, set()):
        return None

    # Embedding-specific knobs.
    if et != 'lstm':
        pruned.pop('lstm_hidden_dim', None)
        pruned.pop('lstm_num_layers', None)
        pruned.pop('lstm_crop_half_width', None)
    if et != 'conv1d':
        pruned.pop('conv1d_num_filters', None)
        pruned.pop('conv1d_crop_half_width', None)
    if et != 'simple':
        pruned.pop('simple_crop_half_width', None)

    # Coupling-specific knobs.
    if pruned.get('coupling_type') != 'spline':
        pruned.pop('spline_num_bins', None)
        pruned.pop('spline_tail_bound', None)

    # Representation-specific knobs. Crop half-widths only matter for raw
    # strain — SVD / Q-transform live in their own shape so the time-domain
    # crop is irrelevant to the learned representation. svd_k / qtransform_*
    # are obviously only consulted by their own pipeline.
    if rep != 'svd':
        pruned.pop('svd_k', None)
        pruned.pop('svd_basis_path', None)
    if rep != 'qtransform':
        pruned.pop('qtransform_logfsteps', None)
        pruned.pop('qtransform_delta_t_out', None)
        pruned.pop('qtransform_frange', None)
        pruned.pop('qtransform_qrange', None)
    if rep != 'strain':
        pruned.pop('simple_crop_half_width', None)
        pruned.pop('conv1d_crop_half_width', None)
        pruned.pop('lstm_crop_half_width', None)

    return pruned


def dedupe(combos: list[dict]) -> list[dict]:
    """Collapse combinations that become identical after pruning."""
    seen, out = set(), []
    for c in combos:
        key = tuple(sorted(c.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def build_csv_columns(all_configs: list[dict]) -> list[str]:
    """Every config key that appears in any combination, plus result columns."""
    keys = set()
    for c in all_configs:
        keys.update(c.keys())
    # Stable ordering: the SEARCH_SPACE key order comes first, then any others.
    ordered_cfg_keys = [k for k in SEARCH_SPACE if k in keys]
    ordered_cfg_keys += sorted(k for k in keys if k not in SEARCH_SPACE)
    return ['run_id', 'status', *ordered_cfg_keys,
            'best_log_prob', 'best_epoch', 'epochs_completed',
            'num_params', 'elapsed_sec', 'checkpoint_path',
            'val_loss_history', 'log_file', 'error']


def format_value(v):
    """Make tuples / lists CSV-safe without losing the structure."""
    if isinstance(v, (list, tuple)):
        return '|'.join(str(x) for x in v)
    return v


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--max-runs', type=int, default=None,
                        help='Cap on combinations (useful for smoke tests).')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print the planned combinations and exit.')
    parser.add_argument('--results-csv', default=RESULTS_CSV_DEFAULT)
    parser.add_argument('--tag', default='hp',
                        help='Prefix for per-run checkpoint_tag (hp000, hp001, ...).')
    parser.add_argument('--skip', type=int, default=0,
                        help='Skip the first N combinations (resume a sweep).')
    args = parser.parse_args()

    # Sibling trainer — CPU in ML/cpu/, GPU in ML/gpu/. The dispatch choice
    # is implicit in which folder this script lives in.
    from train_model import run_training

    pruned = [prune_for_embedding(c) for c in iter_combinations(SEARCH_SPACE)]
    combos = dedupe([c for c in pruned if c is not None])
    print(f"Planned combinations: {len(combos)} (after pruning)")

    if args.skip:
        combos = combos[args.skip:]
    if args.max_runs is not None:
        combos = combos[:args.max_runs]
    print(f"Running: {len(combos)}  (tag={args.tag})")

    if args.dry_run:
        for i, c in enumerate(combos):
            print(f"  [{i:03d}] {c}")
        return

    columns = build_csv_columns(combos)
    results_path = Path(args.results_csv)

    # Write header once. Appending rows below so a crashed sweep still
    # leaves its completed rows on disk.
    fresh = not results_path.exists()
    if fresh:
        with results_path.open('w', newline='') as f:
            csv.writer(f).writerow(columns)

    logs_dir = PIPELINE_ROOT / 'logs'
    logs_dir.mkdir(exist_ok=True)

    sweep_start = time.time()
    for i, combo in enumerate(combos):
        run_id = f"{args.tag}{args.skip + i:03d}"
        log_file = str(logs_dir / f"{run_id}.log")
        cfg = {**COMMON_CONFIG, **combo,
               'checkpoint_tag': run_id,
               'log_file':       log_file}

        print("\n" + "=" * 72)
        print(f"[{i + 1}/{len(combos)}] run_id={run_id}  log={log_file}")
        print(f"  {combo}")
        print("=" * 72)

        row = {c: '' for c in columns}
        row.update({'run_id': run_id,
                    **{k: format_value(v) for k, v in combo.items()},
                    'log_file': log_file})

        try:
            summary = run_training(cfg)
            val_losses = _load_val_losses(summary['checkpoint_path'])
            row.update({
                'status':           'ok',
                'best_log_prob':    summary['best_log_prob'],
                'best_epoch':       summary['best_epoch'],
                'epochs_completed': summary['epochs_completed'],
                'num_params':       summary['num_params'],
                'elapsed_sec':      f"{summary['elapsed_sec']:.1f}",
                'checkpoint_path':  summary['checkpoint_path'],
                'val_loss_history': '|'.join(f"{v:.4f}" for v in val_losses),
            })
        except KeyboardInterrupt:
            print("\nInterrupted by user — writing partial results and exiting.")
            row.update({'status': 'interrupted'})
            _append_row(results_path, columns, row)
            raise
        except Exception as e:
            row.update({'status': 'failed', 'error': f"{type(e).__name__}: {e}"})
            traceback.print_exc()

        _append_row(results_path, columns, row)

    total = time.time() - sweep_start
    print(f"\nSweep finished in {total / 60:.1f} min → {results_path}")


def _append_row(path: Path, columns: list[str], row: dict):
    with path.open('a', newline='') as f:
        w = csv.writer(f)
        w.writerow([row.get(c, '') for c in columns])


def _load_val_losses(ckpt_path):
    """Pull val_losses out of a checkpoint. Empty list on any failure."""
    try:
        import torch
        ckpt = torch.load(ckpt_path, weights_only=False, map_location='cpu')
        return list(ckpt.get('val_losses', []))
    except Exception:
        return []


if __name__ == '__main__':
    # Make sibling modules importable when run from anywhere.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
