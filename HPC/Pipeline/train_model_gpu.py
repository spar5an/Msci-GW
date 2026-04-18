# train_model_gpu.py
# GPU-optimised training entry point for the DINGO model.
#
# This is a copy of the training logic from `train_model_cpu.py`, tuned for
# CUDA:
#
#   * Auto-detect device (cuda → cpu fallback).
#   * Seed CUDA alongside CPU / NumPy so multi-GPU runs are reproducible.
#   * Upload the whole (small) dataset to the GPU once, then index into it
#     for each batch — no pin_memory / DataLoader indirection needed.
#   * In-loop `.to(device, non_blocking=True)` kept as a safety net for the
#     rare path where the dataset was not uploaded.
#   * Validation forward pass is chunked to avoid the 1 000-event one-shot
#     OOM you hit with the conv1d / lstm embeddings on a smaller GPU.
#   * Optional `torch.compile(model, mode='reduce-overhead')` via the
#     `compile_model` config flag. Turns itself off automatically on CPU or
#     if the compile call raises.
#
# The public API mirrors `train_model_cpu.run_training(config)` so
# `hp_search.py` can swap the two without caring which device backs it.

from __future__ import annotations

import copy
import math
import os
import time

import numpy as np
import torch

# Re-use everything the CPU module defines — model classes, dataset loader,
# crop helper, save-path builder, default config — so the two files never
# drift on model internals.
from train_model_cpu import (
    DEFAULT_CONFIG,
    DINGOModel,
    build_checkpoint_path,
    crop_to_merger,  # noqa: F401 (exposed for downstream imports)
    load_dataset_pt,
)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[train_model_gpu] PyTorch {torch.__version__}  device={DEVICE}")
if DEVICE.type == 'cuda':
    print(f"[train_model_gpu] CUDA device: {torch.cuda.get_device_name(0)}  "
          f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")


# GPU-friendly defaults. Anything unset falls back to the CPU module's
# DEFAULT_CONFIG, so only the keys we override live here.
GPU_DEFAULT_CONFIG = {
    **DEFAULT_CONFIG,
    'batch_size':     128,    # GPU amortises kernel launch — go wider
    'val_chunk_size': 128,    # chunk val forward pass to avoid OOM
    'compile_model':  False,  # torch.compile — opt-in per run
    'device_tag':     'gpu',
}


def _seed_everything_gpu(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _chunked_val_logprob(model, val_params, val_data, chunk_size):
    """Mean log-prob over the val split, computed in chunks to bound memory."""
    totals, counts = 0.0, 0
    model.eval()
    with torch.inference_mode():
        for i in range(0, val_data.shape[0], chunk_size):
            p = val_params[i:i + chunk_size]
            d = val_data[i:i + chunk_size]
            lp = model(p, d)
            if torch.isnan(lp).any() or torch.isinf(lp).any():
                continue
            totals += float(lp.sum().item())
            counts += int(lp.shape[0])
    return totals / counts if counts > 0 else float('nan')


def train_dingo_model_gpu(model, train_params, train_data,
                          *, device,
                          num_epochs=20, batch_size=128, lr=1e-4,
                          weight_decay=1e-4,
                          val_params=None, val_data=None,
                          val_chunk_size=128,
                          extra_patience_after_scheduler=3,
                          checkpoint_path=None,
                          checkpoint_extras=None,
                          optimizer_state_dict=None,
                          scheduler_state_dict=None,
                          start_epoch=0,
                          best_log_prob_init=None,
                          best_state_init=None,
                          best_epoch_init=0,
                          bad_epochs_init=0):
    """GPU-optimised training loop. Public contract matches the CPU version
    (returns the same 8-tuple), so the rest of the pipeline is unchanged."""
    if torch.isnan(train_params).any() or torch.isnan(train_data).any():
        raise ValueError("Input data contains NaN values")

    # Upload the full dataset to the device once. For this project the
    # whitened tensor is ~64 MB — well within any modern GPU — so we avoid
    # per-batch H2D copies entirely.
    train_params = train_params.to(device, non_blocking=True)
    train_data   = train_data.to(device, non_blocking=True)
    if val_params is not None:
        val_params = val_params.to(device, non_blocking=True)
        val_data   = val_data.to(device, non_blocking=True)

    print(f"\nData stats:")
    print(f"  train_params: min={train_params.min():.4f}, max={train_params.max():.4f}, "
          f"mean={train_params.mean():.4f}, std={train_params.std():.4f}")
    print(f"  train_data:   min={train_data.min():.4f}, max={train_data.max():.4f}, "
          f"mean={train_data.mean():.4f}, std={train_data.std():.4f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, min_lr=lr * 0.01,
    )
    patience = scheduler.patience + extra_patience_after_scheduler

    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    if scheduler_state_dict is not None:
        scheduler.load_state_dict(scheduler_state_dict)

    num_samples = len(train_params)
    has_val = val_params is not None and val_data is not None

    print(f"\nTraining for up to {num_epochs} epochs  (device={device})")
    print(f"  Early-stop patience: {patience}  "
          f"(scheduler.patience={scheduler.patience} + extra={extra_patience_after_scheduler})")
    print(f"  Samples: {num_samples}  Batch: {batch_size}  "
          f"Val chunk: {val_chunk_size}")
    print(f"  lr={lr}  weight_decay={weight_decay}")
    if checkpoint_path is not None:
        print(f"  Persisting best model on each improvement: {checkpoint_path}")
    if start_epoch > 0:
        print(f"  Resuming from epoch {start_epoch}")
    print()

    losses, val_losses = [], []

    best_log_prob = -float('inf') if best_log_prob_init is None else best_log_prob_init
    best_state = (copy.deepcopy(best_state_init) if best_state_init is not None
                  else copy.deepcopy(model.state_dict()))
    best_epoch = best_epoch_init
    bad_epochs = bad_epochs_init

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_log_prob = 0.0
        num_batches = 0
        batch_log_probs = []

        indices = torch.randperm(num_samples, device=device)

        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:min(i + batch_size, num_samples)]
            batch_params = train_params[batch_indices]
            batch_data = train_data[batch_indices]
            if batch_params.shape[0] < 2:
                continue

            optimizer.zero_grad(set_to_none=True)

            log_prob = model(batch_params, batch_data)
            loss = -log_prob.mean()

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  NaN/Inf loss at epoch {epoch+1}, batch {i//batch_size + 1} — skipped")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            lp = -loss.item()
            epoch_log_prob += lp
            batch_log_probs.append(lp)
            num_batches += 1

        avg_log_prob = epoch_log_prob / num_batches if num_batches > 0 else float('nan')
        losses.append(avg_log_prob)

        avg_val_log_prob = float('nan')
        if has_val:
            avg_val_log_prob = _chunked_val_logprob(
                model, val_params, val_data, val_chunk_size
            )
            val_losses.append(avg_val_log_prob)

        current_metric = (avg_val_log_prob
                          if has_val and not math.isnan(avg_val_log_prob)
                          else avg_log_prob)

        if not math.isnan(current_metric):
            scheduler.step(current_metric)

        if not math.isnan(current_metric) and current_metric > best_log_prob:
            best_log_prob = current_metric
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            bad_epochs = 0
            marker = " *"
            if checkpoint_path is not None:
                ckpt = {
                    'model_state_dict':     best_state,
                    'best_state':           best_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'losses':               list(losses),
                    'val_losses':           list(val_losses),
                    'best_log_prob':        best_log_prob,
                    'best_epoch':           best_epoch,
                    'bad_epochs':           bad_epochs,
                    'epochs_completed':     epoch + 1,
                }
                if checkpoint_extras:
                    ckpt.update(checkpoint_extras)
                torch.save(ckpt, checkpoint_path)
        else:
            bad_epochs += 1
            marker = ""

        current_lr = optimizer.param_groups[0]['lr']
        batch_std = float(np.std(batch_log_probs)) if batch_log_probs else float('nan')
        val_str = f", Val: {avg_val_log_prob:7.4f}" if has_val else ""
        print(f"Epoch {epoch+1:3d}/{num_epochs}, Train: {avg_log_prob:7.4f}{val_str}, "
              f"Best: {best_log_prob:7.4f} @ ep{best_epoch}, Std: {batch_std:6.4f}, "
              f"LR: {current_lr:.2e}{marker}")

        if bad_epochs >= patience:
            print(f"\nEarly stop: no improvement for {patience} epochs "
                  f"(scheduler.patience={scheduler.patience} + {extra_patience_after_scheduler}; "
                  f"best was epoch {best_epoch}).")
            break

    model.load_state_dict(best_state)
    print(f"\nTraining complete. Best log-prob: {best_log_prob:.4f} (epoch {best_epoch}).")
    return losses, val_losses, best_state, best_log_prob, best_epoch, optimizer, scheduler, bad_epochs


def run_training(config=None):
    """GPU-side counterpart to `train_model_cpu.run_training`. Same contract."""
    cfg = {**GPU_DEFAULT_CONFIG, **(config or {})}
    dev = DEVICE
    _seed_everything_gpu(cfg['seed'])

    print(f"\nLoading dataset from: {cfg['dataset_path']}")
    if cfg['merger_crop_half_width'] is not None:
        print(f"  Cropping strain to ±{cfg['merger_crop_half_width']} samples around merger "
              f"(new T = {2 * cfg['merger_crop_half_width']})")
    ds = load_dataset_pt(cfg['dataset_path'],
                         use_whitened=cfg['use_whitened'],
                         merger_crop_half_width=cfg['merger_crop_half_width'])

    train_data,   train_params = ds['train_data'],   ds['train_params']
    val_data,     val_params   = ds['val_data'],     ds['val_params']
    param_norm_info = ds['param_norm_info']
    param_names     = ds['param_names']
    metadata        = ds['metadata']

    param_dim = len(param_names)
    _, num_detectors, seq_len = train_data.shape
    num_training_samples = len(train_params)
    add_noise = metadata.get('add_noise', True)

    print(f"  Train: {len(train_params)}  Val: {len(val_params)}  "
          f"Shape: (N, {num_detectors}, {seq_len})")

    save_path = build_checkpoint_path(cfg, num_training_samples, add_noise)
    print(f"\nModel will be saved as: {save_path}")

    print("\nModel:")
    print(f"  PARAM_DIM={param_dim}  NUM_DETECTORS={num_detectors}  SEQ_LEN={seq_len}")
    print(f"  CONTEXT_DIM={cfg['context_dim']}  NUM_FLOW_LAYERS={cfg['num_flow_layers']}  "
          f"HIDDEN_DIM={cfg['hidden_dim']}  EMBEDDING={cfg['embedding_type']}")

    model = DINGOModel(
        num_detectors=num_detectors,
        seq_len=seq_len,
        param_dim=param_dim,
        context_dim=cfg['context_dim'],
        num_flow_layers=cfg['num_flow_layers'],
        hidden_dim=cfg['hidden_dim'],
        embedding_type=cfg['embedding_type'],
        embedding_dropout=cfg['embedding_dropout'],
        share_detector_weights=cfg['share_detector_weights'],
        lstm_hidden_dim=cfg['lstm_hidden_dim'],
        lstm_num_layers=cfg['lstm_num_layers'],
        conv1d_num_filters=cfg['conv1d_num_filters'],
    ).to(dev)

    if cfg.get('compile_model') and dev.type == 'cuda':
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print("  torch.compile: enabled (mode=reduce-overhead)")
        except Exception as e:
            print(f"  torch.compile: disabled ({type(e).__name__}: {e})")

    num_params = sum(p.numel() for p in model.parameters())

    # Optional resume
    optimizer_state_dict = None
    scheduler_state_dict = None
    start_epoch = 0
    best_log_prob_init = None
    best_state_init = None
    best_epoch_init = 0
    bad_epochs_init = 0
    if cfg['resume_from'] is not None:
        print(f"\nResuming from checkpoint: {cfg['resume_from']}")
        ckpt = torch.load(cfg['resume_from'], weights_only=False, map_location=dev)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer_state_dict = ckpt.get('optimizer_state_dict')
        scheduler_state_dict = ckpt.get('scheduler_state_dict')
        start_epoch = ckpt.get('epochs_completed', 0)
        best_log_prob_init = ckpt.get('best_log_prob')
        best_state_init = ckpt.get('best_state', ckpt.get('model_state_dict'))
        best_epoch_init = ckpt.get('best_epoch', 0)
        bad_epochs_init = ckpt.get('bad_epochs', 0)
        print(f"  Resumed @ epoch {start_epoch}, best_log_prob={best_log_prob_init}")

    checkpoint_extras = {
        'param_norm_info':   param_norm_info,
        'model_param_names': param_names,
        'config': {
            'num_detectors':          num_detectors,
            'seq_len':                seq_len,
            'param_dim':              param_dim,
            'context_dim':            cfg['context_dim'],
            'num_flow_layers':        cfg['num_flow_layers'],
            'hidden_dim':             cfg['hidden_dim'],
            'embedding_type':         cfg['embedding_type'],
            'embedding_dropout':      cfg['embedding_dropout'],
            'share_detector_weights': cfg['share_detector_weights'],
            'lstm_hidden_dim':        cfg['lstm_hidden_dim'],
            'lstm_num_layers':        cfg['lstm_num_layers'],
            'conv1d_num_filters':     list(cfg['conv1d_num_filters']),
            'num_epochs':             cfg['num_epochs'],
            'batch_size':             cfg['batch_size'],
            'learning_rate':          cfg['learning_rate'],
            'weight_decay':           cfg['weight_decay'],
            'num_training_samples':   num_training_samples,
            'add_noise':              add_noise,
            'whiten':                 cfg['use_whitened'],
            'merger_crop_half_width': cfg['merger_crop_half_width'],
            'compile_model':          bool(cfg.get('compile_model', False)),
            'seed':                   cfg['seed'],
        },
    }

    t0 = time.time()
    (losses, val_losses, best_state, best_log_prob, best_epoch,
     optimizer, scheduler, bad_epochs) = train_dingo_model_gpu(
        model, train_params, train_data,
        device=dev,
        num_epochs=cfg['num_epochs'], batch_size=cfg['batch_size'],
        lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'],
        val_params=val_params, val_data=val_data,
        val_chunk_size=cfg['val_chunk_size'],
        extra_patience_after_scheduler=cfg['extra_patience_after_scheduler'],
        checkpoint_path=save_path,
        checkpoint_extras=checkpoint_extras,
        optimizer_state_dict=optimizer_state_dict,
        scheduler_state_dict=scheduler_state_dict,
        start_epoch=start_epoch,
        best_log_prob_init=best_log_prob_init,
        best_state_init=best_state_init,
        best_epoch_init=best_epoch_init,
        bad_epochs_init=bad_epochs_init,
    )
    elapsed = time.time() - t0

    print(f"\nBest model (epoch {best_epoch}) saved to: {save_path}")
    print(f"Total parameters: {num_params:,}   Wall-clock: {elapsed:.1f}s")

    return {
        'best_log_prob':    float(best_log_prob),
        'best_epoch':       int(best_epoch),
        'epochs_completed': start_epoch + len(losses),
        'num_params':       int(num_params),
        'elapsed_sec':      float(elapsed),
        'checkpoint_path':  save_path,
        'config':           cfg,
    }


if __name__ == '__main__':
    run_training({
        'dataset_path':           'Data/dataset.pt',
        'merger_crop_half_width': 500,
        'embedding_type':         'lstm',
        'context_dim':            128,
        'num_flow_layers':        4,
        'hidden_dim':             64,
        'num_epochs':             20,
        'batch_size':             128,
        'learning_rate':          3e-4,
        'compile_model':          False,
    })
