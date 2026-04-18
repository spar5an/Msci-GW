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
    """1D CNN over detector strains concatenated along the time axis.

    Input (N, D, T) is reshaped to (N, 1, D*T) so convolutions operate on one
    long single-channel sequence (e.g. 2 * 8192 = 16384 samples) rather than
    receiving detectors as separate input channels.
    """
    def __init__(self, num_detectors, seq_len, context_dim=128, num_filters=None,
                 dropout=0.1):
        super().__init__()
        if num_filters is None:
            num_filters = [64, 128, 256]
        self.num_detectors = num_detectors
        self.seq_len = seq_len
        self.context_dim = context_dim

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=15, stride=2, padding=7),
                nn.BatchNorm1d(out_c),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
            )

        in_channels = [1] + list(num_filters[:-1])
        self.conv_stack = nn.Sequential(
            *[block(ic, oc) for ic, oc in zip(in_channels, num_filters)]
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(num_filters[-1], 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, context_dim),
            nn.LayerNorm(context_dim),
        )

    def forward(self, data):
        # data: (N, D, T) -> (N, 1, D*T)
        N, D, T = data.shape
        x = data.reshape(N, 1, D * T)
        x = self.conv_stack(x)
        x = self.global_pool(x).view(N, -1)
        return self.fc(x)


class LSTMEmbeddingNetwork(nn.Module):
    """Per-detector 1-D conv down-sampler → shared BiLSTM → merge.

    Each detector's whitened strain (N, 1, T) passes through a stride-2 conv
    stack that shrinks 8192 → ~1024 steps at `conv_channels[-1]` features.
    Both detectors are then fed through THE SAME BiLSTM (shared weights).
    The two detectors' final hidden states are concatenated and projected
    to `context_dim`.
    """
    def __init__(self, num_detectors, seq_len, context_dim=128, hidden_dim=128,
                 num_layers=2, conv_channels=(16, 32, 64), dropout=0.1):
        super().__init__()
        self.num_detectors = num_detectors
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        layers, in_c = [], 1
        for out_c in conv_channels:
            layers += [
                nn.Conv1d(in_c, out_c, kernel_size=15, stride=2, padding=7),
                nn.BatchNorm1d(out_c),
                nn.ReLU(),
            ]
            in_c = out_c
        self.conv_front = nn.Sequential(*layers)
        self._feat_dim = conv_channels[-1]

        self.lstm = nn.LSTM(
            input_size=self._feat_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        merge_in = num_detectors * hidden_dim * 2
        self.output_proj = nn.Sequential(
            nn.Linear(merge_in, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, context_dim),
            nn.LayerNorm(context_dim),
        )

    def forward(self, data):
        # data: (N, D, T)
        N, D, T = data.shape
        x = data.reshape(N * D, 1, T)
        x = self.conv_front(x)
        x = x.transpose(1, 2)
        _, (h_n, _) = self.lstm(x)
        h_fwd = h_n[-2]
        h_bwd = h_n[-1]
        h = torch.cat([h_fwd, h_bwd], dim=1)
        h = h.reshape(N, D * h.shape[-1])
        return self.output_proj(h)


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
                hidden_dim=128, num_layers=2, dropout=embedding_dropout,
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

def crop_to_merger(data, half_width):
    """Centre-crop a (..., T) strain tensor to ±half_width samples around T//2.

    The data-gen pipeline fixes the merger at sample index T//2, so this
    returns `data[..., T//2 - half_width : T//2 + half_width]`. Pass
    `half_width=None` to leave the tensor unchanged.
    """
    if half_width is None:
        return data
    T = data.shape[-1]
    c = T // 2
    lo, hi = c - half_width, c + half_width
    if lo < 0 or hi > T:
        raise ValueError(
            f"crop half_width={half_width} out of bounds for T={T} "
            f"(would need samples [{lo}:{hi}])"
        )
    return data[..., lo:hi].contiguous()


def load_dataset_pt(path, use_whitened=True, merger_crop_half_width=None):
    """Load dataset.pt from generate_dataset.py, split it, and z-score the params.

    Keeps the detector channel axis: data tensors have shape (N, num_detectors, T).
    If `merger_crop_half_width` is an int, strain is centre-cropped to
    ±half_width samples around the merger (which the data-gen pipeline
    places at T//2), giving a new time axis of 2*half_width.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' not found. Run HPC/Pipeline/Data Generation/generate_dataset.py "
            f"and copy/move the produced dataset.pt next to this script."
        )

    raw = torch.load(path, weights_only=False)

    X_key = 'X_whitened' if use_whitened else 'X'
    X = raw[X_key].float()      # (N, num_detectors, T)
    X = crop_to_merger(X, merger_crop_half_width)
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
    """Train with early stopping + mid-run best-checkpoint persistence.

    Uses AdamW + ReduceLROnPlateau (keyed on val log-prob, falls back to
    train log-prob when no val set is provided).

    Early-stop rule: training halts once validation has failed to improve for
    `scheduler.patience + extra_patience_after_scheduler` epochs in a row —
    i.e. three epochs past the point at which the LR scheduler gave up. With
    defaults (scheduler patience 3, extra 3) that is six bad epochs.

    If `checkpoint_path` is provided, a full checkpoint is written to disk
    every time validation improves, so a cancelled run still leaves the best
    model on disk. `checkpoint_extras` is a dict whose entries are merged
    into each saved checkpoint (e.g. config, param_norm_info).

    Pass the *_state_dict and *_init arguments to resume training.

    Returns (losses, val_losses, best_state_dict, best_log_prob, best_epoch,
             optimizer, scheduler, bad_epochs).
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
    patience = scheduler.patience + extra_patience_after_scheduler

    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    if scheduler_state_dict is not None:
        scheduler.load_state_dict(scheduler_state_dict)

    num_samples = len(train_params)
    has_val = val_params is not None and val_data is not None

    print(f"\nTraining for up to {num_epochs} epochs")
    print(f"  Early-stop patience: {patience}  "
          f"(scheduler.patience={scheduler.patience} + extra={extra_patience_after_scheduler})")
    print(f"  Samples: {num_samples}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Weight decay: {weight_decay}")
    if checkpoint_path is not None:
        print(f"  Persisting best model to disk on each improvement: {checkpoint_path}")
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
            if checkpoint_path is not None:
                ckpt = {
                    'model_state_dict': best_state,
                    'best_state': best_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'losses': list(losses),
                    'val_losses': list(val_losses),
                    'best_log_prob': best_log_prob,
                    'best_epoch': best_epoch,
                    'bad_epochs': bad_epochs,
                    'epochs_completed': epoch + 1,
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
              f"Best: {best_log_prob:7.4f} @ ep{best_epoch}, Std: {batch_std:6.4f}, LR: {current_lr:.2e}{marker}")

        if bad_epochs >= patience:
            print(f"\nEarly stop: no improvement for {patience} epochs "
                  f"(scheduler.patience={scheduler.patience} + {extra_patience_after_scheduler}; "
                  f"best was epoch {best_epoch}).")
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

    # Crop whitened strain to ±N samples around the merger (index T//2).
    # Set to None to keep the full 8192-sample window; set to an int (e.g. 500)
    # to train on a physics-relevant window of length 2*N. Applied to train/val/
    # test and to real events at eval time.
    MERGER_CROP_HALF_WIDTH = 500

    # Model (CPU-friendly defaults)
    CONTEXT_DIM     = 128
    NUM_FLOW_LAYERS = 4
    HIDDEN_DIM      = 64
    EMBEDDING_TYPE  = 'lstm'       # 'simple' | 'conv1d' | 'lstm'

    # Training (CPU-friendly defaults)
    NUM_EPOCHS    = 20
    BATCH_SIZE    = 32
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY  = 1e-4
    # Early-stop: halt `EXTRA_PATIENCE_AFTER_SCHEDULER` epochs after the LR
    # scheduler (patience=3) has already given up — so default total patience
    # is scheduler.patience + 3 = 6 bad epochs in a row.
    EXTRA_PATIENCE_AFTER_SCHEDULER = 3

    # Set RESUME_FROM to a checkpoint path to continue training from it.
    RESUME_FROM = None

    # ----- Load data -----
    print(f"\nLoading dataset from: {DATASET_PATH}")
    if MERGER_CROP_HALF_WIDTH is not None:
        print(f"  Cropping strain to ±{MERGER_CROP_HALF_WIDTH} samples around merger "
              f"(new T = {2 * MERGER_CROP_HALF_WIDTH})")
    ds = load_dataset_pt(DATASET_PATH, use_whitened=USE_WHITENED,
                         merger_crop_half_width=MERGER_CROP_HALF_WIDTH)

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
    crop_tag = f"_crop{MERGER_CROP_HALF_WIDTH}" if MERGER_CROP_HALF_WIDTH is not None else ""
    MODEL_SAVE_PATH = (
        f"dingo_N{samples_str}_F{NUM_FLOW_LAYERS}_C{CONTEXT_DIM}_H{HIDDEN_DIM}"
        f"_E{NUM_EPOCHS}_{EMBEDDING_TYPE}{crop_tag}_{signal_tag}_cpu.pt"
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

    # Metadata merged into every checkpoint written during training. The
    # loop persists the best model on each improvement, so a killed run
    # still leaves the best state so far on disk.
    checkpoint_extras = {
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
            'merger_crop_half_width': MERGER_CROP_HALF_WIDTH,
            'seed': SEED,
        },
    }

    # ----- Train -----
    (losses, val_losses, best_state, best_log_prob, best_epoch,
     optimizer, scheduler, bad_epochs) = train_dingo_model(
        model, train_params, train_data,
        num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        val_params=val_params, val_data=val_data,
        extra_patience_after_scheduler=EXTRA_PATIENCE_AFTER_SCHEDULER,
        checkpoint_path=MODEL_SAVE_PATH,
        checkpoint_extras=checkpoint_extras,
        optimizer_state_dict=optimizer_state_dict,
        scheduler_state_dict=scheduler_state_dict,
        start_epoch=start_epoch,
        best_log_prob_init=best_log_prob_init,
        best_state_init=best_state_init,
        best_epoch_init=best_epoch_init,
        bad_epochs_init=bad_epochs_init,
    )

    print(f"\nBest model (epoch {best_epoch}) saved to: {MODEL_SAVE_PATH}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
