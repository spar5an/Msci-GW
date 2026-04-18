# train_model_cpu.py
# CPU-only training for the DINGO model.
# Loads dataset.pt produced by generate_dataset.py (no GPU code, no AMP).
#
# Input:  dataset.pt  (written by Data Generation/generate_dataset.py)
# Output: <model_name>_cpu.pt

import copy
import math
import os

import numpy as np
import torch
import torch.nn as nn

print(f"PyTorch version: {torch.__version__}")

DEVICE = torch.device('cpu')
print(f"Device: {DEVICE}")

pi = np.pi
LOG_2PI = math.log(2.0 * math.pi)


# ==============================================================================
# NEURAL NETWORK CLASSES
# ==============================================================================

class AffineCouplingLayer(nn.Module):
    """Affine coupling with LayerNorm (batch-size-independent)."""
    def __init__(self, dim, context_dim, hidden_dim=128, mask_type='half'):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.register_buffer('mask', torch.zeros(dim))
        if mask_type in ['half', 'even']:
            self.mask[::2] = 1
        elif mask_type == 'odd':
            self.mask[1::2] = 1

        self.scale_net = nn.Sequential(
            nn.Linear(dim + context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
            nn.Tanh()
        )

        self.translation_net = nn.Sequential(
            nn.Linear(dim + context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x, context, reverse=False):
        masked_x = x * self.mask
        scale_input = torch.cat([masked_x, context], dim=1)
        translation_input = torch.cat([masked_x, context], dim=1)

        s = self.scale_net(scale_input)
        t = self.translation_net(translation_input)

        s = s * (1 - self.mask)
        t = t * (1 - self.mask)

        if not reverse:
            y = x * torch.exp(s) + t
            log_det = s.sum(dim=1)
        else:
            y = (x - t) * torch.exp(-s)
            log_det = -s.sum(dim=1)

        return y, log_det


class NormalizingFlow(nn.Module):
    def __init__(self, param_dim=1, context_dim=64, num_layers=6, hidden_dim=128):
        super().__init__()
        self.param_dim = param_dim
        self.context_dim = context_dim

        self.layers = nn.ModuleList([
            AffineCouplingLayer(
                dim=param_dim,
                context_dim=context_dim,
                hidden_dim=hidden_dim,
                mask_type='even' if i % 2 == 0 else 'odd'
            )
            for i in range(num_layers)
        ])

        self.register_buffer('base_mean', torch.zeros(param_dim))
        self.register_buffer('base_std', torch.ones(param_dim))

    def forward(self, params, context):
        z = params
        log_det_sum = 0

        for layer in self.layers:
            z, log_det = layer(z, context, reverse=False)
            log_det_sum += log_det

        # log N(z | base_mean, base_std), summed over dims
        log_prob_base = -0.5 * (
            LOG_2PI + 2.0 * torch.log(self.base_std)
            + ((z - self.base_mean) / self.base_std) ** 2
        ).sum(dim=1)

        return log_prob_base + log_det_sum

    def sample(self, context, num_samples=1):
        batch_size = context.shape[0]
        context_repeated = context.repeat_interleave(num_samples, dim=0)
        z = torch.randn(batch_size * num_samples, self.param_dim, device=context.device)

        for layer in reversed(self.layers):
            z, _ = layer(z, context_repeated, reverse=True)

        return z


# ---- Embedding networks ------------------------------------------------------
# All three consume (N, num_detectors, seq_len) tensors — the raw shape emitted
# by generate_dataset.py — so the two-detector channel axis is preserved.


class SimpleEmbeddingNetwork(nn.Module):
    """
    MLP embedding that respects the detector channel axis.

    Applies a per-channel MLP to each detector, then concatenates the
    per-channel embeddings and projects to `context_dim`. When
    `share_detector_weights=True` the per-channel MLP weights are shared
    across detectors (cheaper, permutation-equivariant between detectors).
    When False each detector gets its own independent MLP — slightly more
    parameters but lets the model learn detector-specific features (useful
    when H1/L1 noise or antenna response differ materially).
    """
    def __init__(self, num_detectors, seq_len, context_dim=64, hidden_dim=128,
                 dropout=0.1, share_detector_weights=True):
        super().__init__()
        self.num_detectors = num_detectors
        self.seq_len = seq_len
        self.share_detector_weights = share_detector_weights
        h = hidden_dim * 2
        self.per_channel_out_dim = h

        def make_per_channel():
            return nn.Sequential(
                nn.LayerNorm(seq_len),
                nn.Linear(seq_len, h),
                nn.ReLU(),
                nn.Linear(h, h),
                nn.ReLU(),
                nn.Linear(h, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        if share_detector_weights:
            self.per_channel = make_per_channel()
        else:
            self.per_channel = nn.ModuleList([make_per_channel() for _ in range(num_detectors)])

        self.merge = nn.Sequential(
            nn.Linear(num_detectors * h, h),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h, context_dim),
        )

    def forward(self, data):
        # data: (N, D, T)
        N, D, T = data.shape
        if self.share_detector_weights:
            per = self.per_channel(data.reshape(N * D, T))  # (N*D, h)
            merged_input = per.reshape(N, D * per.shape[-1])
        else:
            per_channel_outs = [net(data[:, d, :]) for d, net in enumerate(self.per_channel)]
            merged_input = torch.cat(per_channel_outs, dim=1)  # (N, D*h)
        return self.merge(merged_input)


class Conv1DEmbeddingNetwork(nn.Module):
    """1D CNN over the detector channels (in_channels = num_detectors)."""
    def __init__(self, num_detectors, seq_len, context_dim=512, num_filters=None):
        super().__init__()
        if num_filters is None:
            num_filters = [64, 128, 256]
        self.num_detectors = num_detectors
        self.seq_len = seq_len
        self.context_dim = context_dim

        self.conv1 = nn.Sequential(
            nn.Conv1d(num_detectors, num_filters[0], kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(num_filters[0]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(num_filters[0], num_filters[1], kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(num_filters[1]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(num_filters[1], num_filters[2], kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(num_filters[2]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(num_filters[2], 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, context_dim),
            nn.LayerNorm(context_dim)
        )

    def forward(self, data):
        # data: (N, D, T)
        x = self.conv1(data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class LSTMEmbeddingNetwork(nn.Module):
    """BiLSTM with per-timestep detector-vector input."""
    def __init__(self, num_detectors, seq_len, context_dim=512, hidden_dim=256, num_layers=2):
        super().__init__()
        self.num_detectors = num_detectors
        self.seq_len = seq_len
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Sequential(
            nn.Linear(num_detectors, 32),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0
        )

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, context_dim),
            nn.LayerNorm(context_dim)
        )

    def forward(self, data):
        # data: (N, D, T) -> (N, T, D)
        x = data.transpose(1, 2)
        x = self.input_proj(x)
        _, (h_n, _) = self.lstm(x)
        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        final_state = torch.cat([h_forward, h_backward], dim=1)
        return self.output_proj(final_state)


class DINGOModel(nn.Module):
    """observed_data -> EmbeddingNet -> context -> NormalizingFlow -> log p(params | data)"""
    def __init__(self, num_detectors, seq_len, param_dim=1, context_dim=64,
                 num_flow_layers=6, hidden_dim=128, embedding_type='simple',
                 embedding_dropout=0.1, share_detector_weights=True):
        super().__init__()

        self.embedding_type = embedding_type
        self.num_detectors = num_detectors
        self.seq_len = seq_len

        if embedding_type == 'lstm':
            self.embedding_net = LSTMEmbeddingNetwork(
                num_detectors=num_detectors, seq_len=seq_len, context_dim=context_dim,
                hidden_dim=256, num_layers=2
            )
        elif embedding_type == 'conv1d':
            self.embedding_net = Conv1DEmbeddingNetwork(
                num_detectors=num_detectors, seq_len=seq_len, context_dim=context_dim,
                num_filters=[64, 128, 256]
            )
        elif embedding_type == 'simple':
            self.embedding_net = SimpleEmbeddingNetwork(
                num_detectors=num_detectors, seq_len=seq_len, context_dim=context_dim,
                hidden_dim=hidden_dim, dropout=embedding_dropout,
                share_detector_weights=share_detector_weights,
            )
        else:
            raise ValueError(f"Unknown embedding_type: {embedding_type}")

        self.flow = NormalizingFlow(
            param_dim=param_dim,
            context_dim=context_dim,
            num_layers=num_flow_layers,
            hidden_dim=hidden_dim
        )

    def forward(self, params, data):
        context = self.embedding_net(data)
        return self.flow(params, context)

    def sample_posterior(self, data, num_samples=1000):
        self.eval()
        with torch.no_grad():
            context = self.embedding_net(data)
            samples = self.flow.sample(context, num_samples=num_samples)
        return samples


# ==============================================================================
# DATASET LOADING (bridges generate_dataset.py -> training)
# ==============================================================================

def load_dataset_pt(path, use_whitened=True):
    """Load dataset.pt from generate_dataset.py, split it, and z-score the params.

    Keeps the detector channel axis: data tensors have shape (N, num_detectors, T).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' not found. Run HPC/Pipeline/Data Generation/generate_dataset.py "
            f"and copy/move the produced dataset.pt next to this script."
        )

    raw = torch.load(path, weights_only=False)

    X_key = 'X_whitened' if use_whitened else 'X'
    X = raw[X_key].float()      # (N, num_detectors, T)
    y = raw['y'].float()        # (N, P)
    metadata = raw['metadata']
    param_names = list(metadata['parameter_names'])

    train_idx = torch.as_tensor(raw['train_indices'], dtype=torch.long)
    val_idx   = torch.as_tensor(raw['val_indices'],   dtype=torch.long)
    test_idx  = torch.as_tensor(raw['test_indices'],  dtype=torch.long)

    train_X, val_X, test_X = X[train_idx], X[val_idx], X[test_idx]
    train_y_raw, val_y_raw, test_y_raw = y[train_idx], y[val_idx], y[test_idx]

    # z-score using TRAIN split only (no val/test leakage).
    means = train_y_raw.mean(dim=0)
    stds  = train_y_raw.std(dim=0)
    safe_stds = torch.where(stds > 0, stds, torch.ones_like(stds))

    def znorm(t):
        out = (t - means) / safe_stds
        out[:, stds == 0] = 0.0
        return out

    train_y = znorm(train_y_raw)
    val_y   = znorm(val_y_raw)
    test_y  = znorm(test_y_raw)

    param_norm_info = {
        name: {
            'mean':   float(means[j]),
            'std':    float(stds[j]),
            'min':    float(y[:, j].min()),
            'max':    float(y[:, j].max()),
            'method': 'zscore',
        }
        for j, name in enumerate(param_names)
    }

    return {
        'train_data':   train_X, 'train_params': train_y,
        'val_data':     val_X,   'val_params':   val_y,
        'test_data':    test_X,  'test_params':  test_y,
        'param_norm_info': param_norm_info,
        'param_names': param_names,
        'metadata':    metadata,
    }


# ==============================================================================
# TRAINING
# ==============================================================================

def train_dingo_model(model, train_params, train_data,
                      num_epochs=20, batch_size=32, lr=1e-4,
                      weight_decay=1e-4,
                      val_params=None, val_data=None,
                      patience=10,
                      optimizer_state_dict=None,
                      scheduler_state_dict=None,
                      start_epoch=0,
                      best_log_prob_init=None,
                      best_state_init=None,
                      best_epoch_init=0,
                      bad_epochs_init=0):
    """Train with early stopping + best-val checkpointing.

    Uses AdamW + ReduceLROnPlateau (keyed on val log-prob, falls back to
    train log-prob when no val set is provided). Pass the *_state_dict and
    *_init arguments to resume training from a saved checkpoint.

    Returns (losses, val_losses, best_state_dict, best_log_prob, best_epoch,
             optimizer, scheduler).
    """
    if torch.isnan(train_params).any() or torch.isnan(train_data).any():
        raise ValueError("Input data contains NaN values")

    print(f"\nData stats:")
    print(f"  train_params: min={train_params.min():.4f}, max={train_params.max():.4f}, "
          f"mean={train_params.mean():.4f}, std={train_params.std():.4f}")
    print(f"  train_data:   min={train_data.min():.4f}, max={train_data.max():.4f}, "
          f"mean={train_data.mean():.4f}, std={train_data.std():.4f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, min_lr=lr * 0.01
    )

    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    if scheduler_state_dict is not None:
        scheduler.load_state_dict(scheduler_state_dict)

    num_samples = len(train_params)
    has_val = val_params is not None and val_data is not None

    print(f"\nTraining for up to {num_epochs} epochs  (early-stop patience={patience})")
    print(f"  Samples: {num_samples}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Weight decay: {weight_decay}")
    if start_epoch > 0:
        print(f"  Resuming from epoch {start_epoch}")
    print()

    losses = []
    val_losses = []

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

        indices = torch.randperm(num_samples)

        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:min(i + batch_size, num_samples)]
            batch_params = train_params[batch_indices]
            batch_data = train_data[batch_indices]

            if batch_params.shape[0] < 2:
                continue

            optimizer.zero_grad()

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
            model.eval()
            with torch.inference_mode():
                vlog_prob = model(val_params, val_data)
                if not (torch.isnan(vlog_prob).any() or torch.isinf(vlog_prob).any()):
                    avg_val_log_prob = vlog_prob.mean().item()
            val_losses.append(avg_val_log_prob)

        # Track best model on validation log-prob (fallback to train).
        current_metric = avg_val_log_prob if has_val and not math.isnan(avg_val_log_prob) else avg_log_prob

        if not math.isnan(current_metric):
            scheduler.step(current_metric)

        if not math.isnan(current_metric) and current_metric > best_log_prob:
            best_log_prob = current_metric
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            bad_epochs = 0
            marker = " *"
        else:
            bad_epochs += 1
            marker = ""

        current_lr = optimizer.param_groups[0]['lr']
        batch_std = float(np.std(batch_log_probs)) if batch_log_probs else float('nan')
        val_str = f", Val: {avg_val_log_prob:7.4f}" if has_val else ""
        print(f"Epoch {epoch+1:3d}/{num_epochs}, Train: {avg_log_prob:7.4f}{val_str}, "
              f"Best: {best_log_prob:7.4f} @ ep{best_epoch}, Std: {batch_std:6.4f}, LR: {current_lr:.2e}{marker}")

        if bad_epochs >= patience:
            print(f"\nEarly stop: no improvement for {patience} epochs (best was epoch {best_epoch}).")
            break

    # Restore best weights so the caller sees the best model.
    model.load_state_dict(best_state)
    print(f"\nTraining complete. Best log-prob: {best_log_prob:.4f} (epoch {best_epoch}).")
    return losses, val_losses, best_state, best_log_prob, best_epoch, optimizer, scheduler, bad_epochs


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == '__main__':
    # ----- Reproducibility -----
    SEED = 0
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ----- Config -----
    DATASET_PATH = 'Data/dataset.pt'    # produced by Data Generation/generate_dataset.py
    USE_WHITENED = True

    # Model (CPU-friendly defaults)
    CONTEXT_DIM     = 128
    NUM_FLOW_LAYERS = 4
    HIDDEN_DIM      = 64
    EMBEDDING_TYPE  = 'simple'     # 'simple' | 'conv1d' | 'lstm'

    # Training (CPU-friendly defaults)
    NUM_EPOCHS    = 20
    BATCH_SIZE    = 32
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY  = 1e-4
    PATIENCE      = 10

    # Set RESUME_FROM to a checkpoint path to continue training from it.
    RESUME_FROM = None

    # ----- Load data -----
    print(f"\nLoading dataset from: {DATASET_PATH}")
    ds = load_dataset_pt(DATASET_PATH, use_whitened=USE_WHITENED)

    train_data,   train_params = ds['train_data'],   ds['train_params']
    val_data,     val_params   = ds['val_data'],     ds['val_params']
    test_data,    test_params  = ds['test_data'],    ds['test_params']
    param_norm_info = ds['param_norm_info']
    param_names     = ds['param_names']
    metadata        = ds['metadata']

    PARAM_DIM = len(param_names)
    _, NUM_DETECTORS, SEQ_LEN = train_data.shape

    print(f"  Train:          {len(train_params)}")
    print(f"  Validation:     {len(val_params)}")
    print(f"  Test:           {len(test_params)}")
    print(f"  Data shape:     (N, {NUM_DETECTORS}, {SEQ_LEN})  ({'whitened' if USE_WHITENED else 'raw'})")
    print(f"  Params:         {param_names}")

    # ----- Model save path -----
    num_training_samples = len(train_params)
    samples_str = f"{num_training_samples//1000}k" if num_training_samples >= 1000 else str(num_training_samples)
    add_noise = metadata.get('add_noise', True)
    # Signal descriptor: whitening flattens the coloured noise, so "noisy"
    # alongside "whitened" reads as a contradiction. Emit just one tag.
    if USE_WHITENED:
        signal_tag = "whitened"
    else:
        signal_tag = "noisy" if add_noise else "clean"
    MODEL_SAVE_PATH = (
        f"dingo_N{samples_str}_F{NUM_FLOW_LAYERS}_C{CONTEXT_DIM}_H{HIDDEN_DIM}"
        f"_E{NUM_EPOCHS}_{EMBEDDING_TYPE}_{signal_tag}_cpu.pt"
    )
    print(f"\nModel will be saved as: {MODEL_SAVE_PATH}")

    # ----- Build model -----
    print("\nModel:")
    print(f"  PARAM_DIM={PARAM_DIM}  NUM_DETECTORS={NUM_DETECTORS}  SEQ_LEN={SEQ_LEN}")
    print(f"  CONTEXT_DIM={CONTEXT_DIM}  NUM_FLOW_LAYERS={NUM_FLOW_LAYERS}  HIDDEN_DIM={HIDDEN_DIM}  EMBEDDING={EMBEDDING_TYPE}")

    model = DINGOModel(
        num_detectors=NUM_DETECTORS,
        seq_len=SEQ_LEN,
        param_dim=PARAM_DIM,
        context_dim=CONTEXT_DIM,
        num_flow_layers=NUM_FLOW_LAYERS,
        hidden_dim=HIDDEN_DIM,
        embedding_type=EMBEDDING_TYPE,
    ).to(DEVICE)

    # ----- Optional: resume from checkpoint -----
    optimizer_state_dict = None
    scheduler_state_dict = None
    start_epoch = 0
    best_log_prob_init = None
    best_state_init = None
    best_epoch_init = 0
    bad_epochs_init = 0
    if RESUME_FROM is not None:
        print(f"\nResuming from checkpoint: {RESUME_FROM}")
        ckpt = torch.load(RESUME_FROM, weights_only=False, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer_state_dict = ckpt.get('optimizer_state_dict')
        scheduler_state_dict = ckpt.get('scheduler_state_dict')
        start_epoch = ckpt.get('epochs_completed', 0)
        best_log_prob_init = ckpt.get('best_log_prob')
        best_state_init = ckpt.get('best_state', ckpt.get('model_state_dict'))
        best_epoch_init = ckpt.get('best_epoch', 0)
        bad_epochs_init = ckpt.get('bad_epochs', 0)
        print(f"  Resumed @ epoch {start_epoch}, best_log_prob={best_log_prob_init}")

    # ----- Train -----
    (losses, val_losses, best_state, best_log_prob, best_epoch,
     optimizer, scheduler, bad_epochs) = train_dingo_model(
        model, train_params, train_data,
        num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        val_params=val_params, val_data=val_data,
        patience=PATIENCE,
        optimizer_state_dict=optimizer_state_dict,
        scheduler_state_dict=scheduler_state_dict,
        start_epoch=start_epoch,
        best_log_prob_init=best_log_prob_init,
        best_state_init=best_state_init,
        best_epoch_init=best_epoch_init,
        bad_epochs_init=bad_epochs_init,
    )

    epochs_completed = start_epoch + len(losses)

    # ----- Save best checkpoint -----
    torch.save({
        'model_state_dict': best_state,
        'best_state': best_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'losses': losses,
        'val_losses': val_losses,
        'best_log_prob': best_log_prob,
        'best_epoch': best_epoch,
        'bad_epochs': bad_epochs,
        'epochs_completed': epochs_completed,
        'param_norm_info': param_norm_info,
        'model_param_names': param_names,
        'config': {
            'num_detectors': NUM_DETECTORS,
            'seq_len': SEQ_LEN,
            'param_dim': PARAM_DIM,
            'context_dim': CONTEXT_DIM,
            'num_flow_layers': NUM_FLOW_LAYERS,
            'hidden_dim': HIDDEN_DIM,
            'embedding_type': EMBEDDING_TYPE,
            'num_epochs': NUM_EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
            'num_training_samples': num_training_samples,
            'add_noise': add_noise,
            'whiten': USE_WHITENED,
            'seed': SEED,
        },
    }, MODEL_SAVE_PATH)

    print(f"\nSaved best model (epoch {best_epoch}) to: {MODEL_SAVE_PATH}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
