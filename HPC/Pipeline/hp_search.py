"""hp_search.py — grid hyper-parameter search over the DINGO training config.

Edit `SEARCH_SPACE` below to list candidate values per hyperparameter. The
script takes the cartesian product of those lists, passes each combination
through `run_training(config)` (from `train_model_cpu` or `train_model_gpu`),
and appends one CSV row per trial.

Usage:

    python hp_search.py                       # CPU trainer, full grid
    python hp_search.py --device gpu          # GPU trainer
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


# ── SEARCH SPACE ─────────────────────────────────────────────────────────────
# Each key is a DEFAULT_CONFIG key; each value is a list of candidate settings.
# Drop a list to a single value to pin that axis. Combinations are the
# cartesian product across all keys. Tuples (e.g. conv1d_num_filters) must
# be wrapped in an outer list as a single element — see example below.

SEARCH_SPACE: dict[str, list] = {
    'embedding_type':   ['simple', 'conv1d', 'lstm'],
    'context_dim':      [64, 128],
    'hidden_dim':       [64, 128],
    'num_flow_layers':  [4, 6],
    # Embedding-specific knobs — only read when the matching embedding is picked.
    # Including them here still multiplies the grid, so keep lists short.
    'lstm_hidden_dim':  [128],
    'lstm_num_layers':  [2],
    'conv1d_num_filters': [(64, 128, 256)],
    # Optimiser knobs (optional to vary)
    'learning_rate':    [1e-4],
    'batch_size':       [32],
}

# Settings shared by every run (overrides DEFAULT_CONFIG once per sweep).
COMMON_CONFIG: dict = {
    'dataset_path':           'Data/dataset.pt',
    'merger_crop_half_width': 500,
    'num_epochs':             20,
    'seed':                   0,
}

# Where to write the rolling CSV (overwritten on start unless --resume-csv).
RESULTS_CSV_DEFAULT = 'hp_search_results.csv'


# ── HELPERS ──────────────────────────────────────────────────────────────────

def iter_combinations(space: dict) -> list[dict]:
    """Cartesian product of a dict-of-lists into a list of flat dicts."""
    keys = list(space.keys())
    values = [space[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def prune_for_embedding(cfg: dict) -> dict:
    """Remove HP knobs that the chosen embedding does not use.

    Without this, picking `embedding_type='simple'` with different
    `lstm_hidden_dim` values would log distinct rows for identical trainings.
    """
    pruned = dict(cfg)
    et = pruned.get('embedding_type')
    if et != 'lstm':
        pruned.pop('lstm_hidden_dim', None)
        pruned.pop('lstm_num_layers', None)
    if et != 'conv1d':
        pruned.pop('conv1d_num_filters', None)
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
            'num_params', 'elapsed_sec', 'checkpoint_path', 'error']


def format_value(v):
    """Make tuples / lists CSV-safe without losing the structure."""
    if isinstance(v, (list, tuple)):
        return '|'.join(str(x) for x in v)
    return v


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='cpu',
                        help='Which trainer module to call.')
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

    # Import the trainer lazily so the CPU sweep does not need torch.cuda.
    if args.device == 'gpu':
        from train_model_gpu import run_training
    else:
        from train_model_cpu import run_training

    combos = dedupe([prune_for_embedding(c) for c in iter_combinations(SEARCH_SPACE)])
    print(f"Planned combinations: {len(combos)} (after pruning)")

    if args.skip:
        combos = combos[args.skip:]
    if args.max_runs is not None:
        combos = combos[:args.max_runs]
    print(f"Running: {len(combos)}  (device={args.device}, tag={args.tag})")

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

    sweep_start = time.time()
    for i, combo in enumerate(combos):
        run_id = f"{args.tag}{args.skip + i:03d}"
        cfg = {**COMMON_CONFIG, **combo, 'checkpoint_tag': run_id}

        print("\n" + "=" * 72)
        print(f"[{i + 1}/{len(combos)}] run_id={run_id}  {combo}")
        print("=" * 72)

        row = {c: '' for c in columns}
        row.update({'run_id': run_id, **{k: format_value(v) for k, v in combo.items()}})

        try:
            summary = run_training(cfg)
            row.update({
                'status':           'ok',
                'best_log_prob':    summary['best_log_prob'],
                'best_epoch':       summary['best_epoch'],
                'epochs_completed': summary['epochs_completed'],
                'num_params':       summary['num_params'],
                'elapsed_sec':      f"{summary['elapsed_sec']:.1f}",
                'checkpoint_path':  summary['checkpoint_path'],
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


if __name__ == '__main__':
    # Make sibling modules importable when run from anywhere.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
