# train_model.py
# Script 2 of 3: Load the prepared dataset and train the DINGO model.
# Run generate_data.py first to produce prepared_data.pt, then run this script.
#
# Input:  prepared_data.pt  (written by generate_data.py)
# Output: <model_name>.pt   (checkpoint with state dict, config, and norm info)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gc
import psutil
import os

print("Libraries imported successfully")
print(f"PyTorch version: {torch.__version__}")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

pi = np.pi


# ==============================================================================
# MEMORY UTILITIES
# ==============================================================================

def get_memory_usage():
    """Get current memory usage for CPU and GPU."""
    process = psutil.Process(os.getpid())
    cpu_mem_gb = process.memory_info().rss / (1024**3)
    gpu_mem_gb = 0
    if torch.cuda.is_available():
        gpu_mem_gb = torch.cuda.memory_allocated() / (1024**3)
    return cpu_mem_gb, gpu_mem_gb


def print_memory_status():
    """Print current memory usage."""
    cpu_mem, gpu_mem = get_memory_usage()
    print(f"\n{'='*60}")
    print("MEMORY STATUS:")
    print(f"  CPU Memory: {cpu_mem:.2f} GB")
    if torch.cuda.is_available():
        print(f"  GPU Memory: {gpu_mem:.2f} GB")
    print(f"{'='*60}\n")


def clear_memory(verbose=True):
    """Clear Python memory cache, garbage collection, and PyTorch cache."""
    if verbose:
        print("\nClearing memory...")
        cpu_before, gpu_before = get_memory_usage()
        print(f"  Before: CPU={cpu_before:.2f}GB", end="")
        if torch.cuda.is_available():
            print(f", GPU={gpu_before:.2f}GB", end="")
        print()

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if verbose:
        cpu_after, gpu_after = get_memory_usage()
        print(f"  After:  CPU={cpu_after:.2f}GB", end="")
        if torch.cuda.is_available():
            print(f", GPU={gpu_after:.2f}GB", end="")
        print()
        cpu_freed = cpu_before - cpu_after
        print(f"  Freed: {cpu_freed:.2f} GB")
        if torch.cuda.is_available():
            gpu_freed = gpu_before - gpu_after
            print(f"  GPU Freed: {gpu_freed:.2f} GB")


# ==============================================================================
# NEURAL NETWORK CLASSES
# ==============================================================================

class AffineCouplingLayer(nn.Module):
    """
    Affine coupling layer for normalizing flows.
    Splits input, transforms one half conditioned on the other:
    x2_new = x2 * exp(s(x1, context)) + t(x1, context)
    """
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
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
            nn.Tanh()
        )

        self.translation_net = nn.Sequential(
            nn.Linear(dim + context_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
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
    """Normalizing flow: stack of coupling layers."""
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

        log_prob_base = -0.5 * (torch.log(2 * np.pi * self.base_std**2) +
                                 ((z - self.base_mean) / self.base_std)**2)
        log_prob_base = log_prob_base.sum(dim=1)

        log_prob = log_prob_base + log_det_sum
        return log_prob

    def sample(self, context, num_samples=1):
        batch_size = context.shape[0]
        context_repeated = context.repeat_interleave(num_samples, dim=0)
        z = torch.randn(batch_size * num_samples, self.param_dim, device=context.device)

        for layer in reversed(self.layers):
            z, _ = layer(z, context_repeated, reverse=True)

        return z


class EmbeddingNetwork(nn.Module):
    """Simple fully-connected embedding network."""
    def __init__(self, data_dim=100, context_dim=64, hidden_dim=128):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, context_dim)
        )

    def forward(self, data):
        return self.network(data)


class Conv1DEmbeddingNetwork(nn.Module):
    """1D CNN embedding network for sequential waveform data."""
    def __init__(self, data_dim=5868, context_dim=512, num_filters=None):
        super().__init__()
        if num_filters is None:
            num_filters = [64, 128, 256]
        self.data_dim = data_dim
        self.context_dim = context_dim

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, num_filters[0], kernel_size=15, stride=2, padding=7),
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
        x = data.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        context = self.fc(x)
        return context


class LSTMEmbeddingNetwork(nn.Module):
    """LSTM-based embedding network for waveform data."""
    def __init__(self, data_dim=7241, context_dim=512, hidden_dim=256, num_layers=2):
        super().__init__()
        self.data_dim = data_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Sequential(
            nn.Linear(1, 32),
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
        x = data.unsqueeze(-1)
        x = self.input_proj(x)
        lstm_out, (h_n, c_n) = self.lstm(x)
        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        final_state = torch.cat([h_forward, h_backward], dim=1)
        context = self.output_proj(final_state)
        return context


class DINGOModel(nn.Module):
    """
    Complete DINGO-style neural posterior estimation model.
    observed_data -> EmbeddingNet -> context -> NormalizingFlow -> log p(params | data)
    """
    def __init__(self, data_dim=100, param_dim=1, context_dim=64,
                 num_flow_layers=6, hidden_dim=128, device=None, embedding_type='simple',
                 time_delay_value=0.0):
        super().__init__()

        self.embedding_type = embedding_type
        self.time_delay_value = time_delay_value

        if embedding_type == 'lstm':
            self.embedding_net = LSTMEmbeddingNetwork(
                data_dim=data_dim, context_dim=context_dim, hidden_dim=256, num_layers=2
            )
        elif embedding_type == 'conv1d':
            self.embedding_net = Conv1DEmbeddingNetwork(
                data_dim=data_dim, context_dim=context_dim, num_filters=[64, 128, 256]
            )
        elif embedding_type == 'simple':
            self.embedding_net = nn.Sequential(
                nn.LayerNorm(data_dim),
                nn.Linear(data_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, context_dim)
            )
        else:
            raise ValueError(f"Unknown embedding_type: {embedding_type}. Must be 'simple', 'conv1d', or 'lstm'.")

        flow_context_dim = context_dim + 1  # +1 for time delay

        self.flow = NormalizingFlow(
            param_dim=param_dim,
            context_dim=flow_context_dim,
            num_layers=num_flow_layers,
            hidden_dim=hidden_dim
        )

    def forward(self, params, data):
        context = self.embedding_net(data)
        batch_size = context.shape[0]
        time_delay_tensor = torch.full((batch_size, 1), self.time_delay_value,
                                       dtype=context.dtype, device=context.device)
        context = torch.cat([context, time_delay_tensor], dim=1)
        log_prob = self.flow(params, context)
        return log_prob

    def sample_posterior(self, data, num_samples=1000):
        self.eval()
        with torch.no_grad():
            context = self.embedding_net(data)
            batch_size = context.shape[0]
            time_delay_tensor = torch.full((batch_size, 1), self.time_delay_value,
                                           dtype=context.dtype, device=context.device)
            context = torch.cat([context, time_delay_tensor], dim=1)
            samples = self.flow.sample(context, num_samples=num_samples)
        return samples


# ==============================================================================
# TRAINING FUNCTION
# ==============================================================================

def train_dingo_model_pycbc(model, train_params, train_data,
                             num_epochs=100, batch_size=256, lr=1e-4, use_mixed_precision=True,
                             val_params=None, val_data=None):

    model = model.to(DEVICE)
    train_params = train_params.to(DEVICE)
    train_data = train_data.to(DEVICE)

    if torch.isnan(train_params).any() or torch.isnan(train_data).any():
        print(" ERROR: Input data contains NaN values!")
        print(f"  train_params NaN count: {torch.isnan(train_params).sum().item()}")
        print(f"  train_data NaN count: {torch.isnan(train_data).sum().item()}")
        raise ValueError("Input data contains NaN values")

    print(f"\n✓ Data Statistics:")
    print(f"  train_params: min={train_params.min():.4f}, max={train_params.max():.4f}, mean={train_params.mean():.4f}, std={train_params.std():.4f}")
    print(f"  train_data: min={train_data.min():.4f}, max={train_data.max():.4f}, mean={train_data.mean():.4f}, std={train_data.std():.4f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr * 0.01
    )

    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() and use_mixed_precision else None
    torch.backends.cudnn.benchmark = True

    num_samples = len(train_params)
    print(f"\nTraining PyCBC DINGO model for {num_epochs} epochs...")
    print(f"  Samples: {num_samples}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}\n")

    losses = []
    val_losses = []
    has_val = val_params is not None and val_data is not None
    if has_val:
        val_params = val_params.to(DEVICE)
        val_data = val_data.to(DEVICE)
        print(f"  Validation samples: {len(val_params)}")

    best_loss = -float('inf')
    first_batch_debug = True

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0
        batch_losses = []

        indices = torch.randperm(num_samples, device=DEVICE)

        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:min(i + batch_size, num_samples)]
            batch_params = train_params[batch_indices]
            batch_data = train_data[batch_indices]

            optimizer.zero_grad()

            try:
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        log_prob = model(batch_params, batch_data)

                        if first_batch_debug and epoch == 0:
                            print(f"✓ First batch output check:")
                            print(f"  log_prob shape: {log_prob.shape}")
                            print(f"  log_prob min/max: {log_prob.min():.4f} / {log_prob.max():.4f}")
                            print(f"  log_prob NaN count: {torch.isnan(log_prob).sum().item()}")
                            print(f"  log_prob Inf count: {torch.isinf(log_prob).sum().item()}")
                            first_batch_debug = False

                        loss = -log_prob.mean()

                        if torch.isnan(loss) or torch.isinf(loss):
                            print(f"\n⚠ NaN/Inf detected in loss at epoch {epoch+1}, batch {i//batch_size + 1}")
                            loss = torch.tensor(0.0, device=DEVICE)

                        context = model.embedding_net(batch_data)
                        context_std = context.std(dim=0).mean()
                        reg_loss = 10.0 * torch.clamp(1.5 - context_std, min=0)

                        total_loss = loss + reg_loss

                    scaler.scale(total_loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    log_prob = model(batch_params, batch_data)
                    loss = -log_prob.mean()

                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"\n⚠ NaN/Inf detected in loss at epoch {epoch+1}, batch {i//batch_size + 1}")
                        loss = torch.tensor(0.0, device=DEVICE)

                    context = model.embedding_net(batch_data)
                    context_std = context.std(dim=0).mean()
                    reg_loss = 10.0 * torch.clamp(1.5 - context_std, min=0)

                    total_loss = loss + reg_loss

                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                loss_value = -loss.item() if not torch.isnan(loss) else 0.0
                epoch_loss += loss_value
                batch_losses.append(loss_value)
                num_batches += 1

            except Exception as e:
                print(f"\n Error during training at epoch {epoch+1}, batch {i//batch_size + 1}: {e}")
                import traceback
                traceback.print_exc()
                raise

        avg_log_prob = epoch_loss / num_batches if num_batches > 0 else float('nan')
        losses.append(avg_log_prob)

        with torch.no_grad():
            context_check = model.embedding_net(batch_data)
            context_std_check = context_check.std(dim=0).mean().item()

        avg_val_log_prob = float('nan')
        if has_val:
            model.eval()
            with torch.no_grad():
                val_epoch_loss = 0.0
                val_num_batches = 0
                for vi in range(0, len(val_params), batch_size):
                    vp = val_params[vi:vi + batch_size]
                    vd = val_data[vi:vi + batch_size]
                    try:
                        if scaler is not None:
                            with torch.amp.autocast('cuda'):
                                vlog_prob = model(vp, vd)
                        else:
                            vlog_prob = model(vp, vd)
                        if not torch.isnan(vlog_prob).any() and not torch.isinf(vlog_prob).any():
                            val_epoch_loss += vlog_prob.mean().item()
                            val_num_batches += 1
                    except Exception:
                        pass
            avg_val_log_prob = val_epoch_loss / val_num_batches if val_num_batches > 0 else float('nan')
            val_losses.append(avg_val_log_prob)
            model.train()

        scheduler.step()

        if not np.isnan(avg_log_prob) and avg_log_prob > best_loss:
            best_loss = avg_log_prob

        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            batch_std = np.std(batch_losses) if not all(np.isnan(batch_losses)) else float('nan')
            nan_status = "✓" if not np.isnan(avg_log_prob) else "⚠"
            val_str = f", Val: {avg_val_log_prob:7.4f}" if has_val else ""
            print(f"{nan_status} Epoch {epoch+1:3d}/{num_epochs}, Train: {avg_log_prob:7.4f}{val_str}, Best: {best_loss:7.4f}, Std: {batch_std:6.4f}, Context_std: {context_std_check:6.4f}, LR: {current_lr:.2e}")

    print("\n✓ Training complete!")
    return losses, val_losses


# ==============================================================================
# MAIN: Load Data + Train Model + Save Checkpoint
# ==============================================================================

if __name__ == '__main__':
    # -----------------------------------------------------------------------
    # CONFIGURATION
    # -----------------------------------------------------------------------

    # Path to the prepared dataset produced by generate_data.py
    PREPARED_DATA_PATH = 'prepared_data.pt'

    # Model architecture parameters
    CONTEXT_DIM = 512
    NUM_FLOW_LAYERS = 5
    HIDDEN_DIM = 128
    EMBEDDING_TYPE = 'simple'  # 'simple', 'conv1d', or 'lstm'

    # Training parameters
    NUM_EPOCHS = 200
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    RETRAIN = True  # Set to False to skip training if a saved model already exists

    # -----------------------------------------------------------------------
    # LOAD PREPARED DATA
    # -----------------------------------------------------------------------
    print(f"\nLoading prepared data from: {PREPARED_DATA_PATH}")
    if not os.path.exists(PREPARED_DATA_PATH):
        raise FileNotFoundError(
            f"'{PREPARED_DATA_PATH}' not found. Run generate_data.py first."
        )

    checkpoint_data = torch.load(PREPARED_DATA_PATH, map_location='cpu')
    pycbc_data      = checkpoint_data['train_data']
    pycbc_params    = checkpoint_data['train_params']
    pycbc_val_data  = checkpoint_data['val_data']
    pycbc_val_params = checkpoint_data['val_params']
    pycbc_test_data = checkpoint_data['test_data']
    pycbc_test_params = checkpoint_data['test_params']
    param_norm_info = checkpoint_data['param_norm_info']
    model_param_names = checkpoint_data['param_names']
    GPS_TIME_DELAY  = checkpoint_data['gps_time_delay']

    # Recover config stored by generate_data.py
    saved_config = checkpoint_data.get('config', {})
    ADD_NOISE         = saved_config.get('add_noise', True)
    WHITEN            = saved_config.get('whiten', True)
    MODIFIED_GRAVITY  = saved_config.get('modified_gravity', False)
    PARAM_SET         = saved_config.get('param_set', 'symmetric')
    MODEL_PARAMS      = saved_config.get('model_params', list(model_param_names))
    NUM_TRAINING_SAMPLES = saved_config.get('num_training_samples', len(pycbc_data))

    PARAM_DIM = len(model_param_names)

    print(f"✓ Data loaded:")
    print(f"  Training samples:   {len(pycbc_params)}")
    print(f"  Validation samples: {len(pycbc_val_params)}")
    print(f"  Test samples:       {len(pycbc_test_params)}")
    print(f"  Data dimension:     {pycbc_data.shape[1]} (2 detectors concatenated)")
    print(f"  Target parameters:  {model_param_names}")
    print(f"  GPS time delay:     {GPS_TIME_DELAY*1000:.3f} ms")

    # -----------------------------------------------------------------------
    # MODEL SAVE PATH
    # -----------------------------------------------------------------------
    samples_str = f"{NUM_TRAINING_SAMPLES//1000}k" if NUM_TRAINING_SAMPLES >= 1000 else str(NUM_TRAINING_SAMPLES)
    noise_str = "noisy" if ADD_NOISE else "clean"
    whiten_str = "_whitened" if WHITEN else ""
    mg_str = "_MG" if MODIFIED_GRAVITY else ""
    MODEL_SAVE_PATH = (
        f"dingo_N{samples_str}_F{NUM_FLOW_LAYERS}_C{CONTEXT_DIM}_H{HIDDEN_DIM}"
        f"_E{NUM_EPOCHS}_{EMBEDDING_TYPE}_{noise_str}{whiten_str}_{PARAM_SET}{mg_str}.pt"
    )
    print(f"\nModel will be saved as: {MODEL_SAVE_PATH}")

    # -----------------------------------------------------------------------
    # MODEL SETUP
    # -----------------------------------------------------------------------
    print("\n" + "="*70)
    print("MODEL ARCHITECTURE PARAMETERS")
    print(f"  Parameter Dimension (PARAM_DIM):         {PARAM_DIM}")
    print(f"  Context Dimension (CONTEXT_DIM):         {CONTEXT_DIM}")
    print(f"  Number of Flow Layers (NUM_FLOW_LAYERS): {NUM_FLOW_LAYERS}")
    print(f"  Hidden Dimension (HIDDEN_DIM):           {HIDDEN_DIM}")
    print(f"  Embedding Type (EMBEDDING_TYPE):         {EMBEDDING_TYPE}")
    print(f"  Number of Epochs (NUM_EPOCHS):           {NUM_EPOCHS}")
    print(f"  Batch Size (BATCH_SIZE):                 {BATCH_SIZE}")
    print(f"  Learning Rate (LEARNING_RATE):           {LEARNING_RATE}")
    print("="*70 + "\n")

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

    # -----------------------------------------------------------------------
    # TRAIN OR LOAD
    # -----------------------------------------------------------------------
    if os.path.exists(MODEL_SAVE_PATH) and not RETRAIN:
        print(f"Loading existing model from: {MODEL_SAVE_PATH}")
        try:
            ckpt = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
            model.load_state_dict(ckpt['model_state_dict'])
            model = model.to(DEVICE)
            losses = ckpt.get('losses', [])
            val_losses = ckpt.get('val_losses', [])
            print("✓ Model loaded successfully (skipping training).")
        except Exception as e:
            print(f"  Failed to load model: {e}. Retraining...")
            os.remove(MODEL_SAVE_PATH)
            losses, val_losses = train_dingo_model_pycbc(
                model, pycbc_params, pycbc_data,
                num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE,
                val_params=pycbc_val_params, val_data=pycbc_val_data,
            )
    else:
        if os.path.exists(MODEL_SAVE_PATH):
            print(f"RETRAIN=True — retraining even though {MODEL_SAVE_PATH} exists.")
        else:
            print("No pre-trained model found — training from scratch.")

        losses, val_losses = train_dingo_model_pycbc(
            model, pycbc_params, pycbc_data,
            num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE,
            val_params=pycbc_val_params, val_data=pycbc_val_data,
        )

    # -----------------------------------------------------------------------
    # SAVE CHECKPOINT
    # (includes param_norm_info and model_param_names so plot_results.py
    #  can load everything it needs from this single file)
    # -----------------------------------------------------------------------
    torch.save({
        'model_state_dict': model.state_dict(),
        'losses': losses,
        'val_losses': val_losses,
        'param_norm_info': param_norm_info,
        'model_param_names': model_param_names,
        'gps_time_delay': GPS_TIME_DELAY,
        'config': {
            'data_dim': pycbc_data.shape[1],
            'param_dim': PARAM_DIM,
            'context_dim': CONTEXT_DIM,
            'num_flow_layers': NUM_FLOW_LAYERS,
            'hidden_dim': HIDDEN_DIM,
            'embedding_type': EMBEDDING_TYPE,
            'num_epochs': NUM_EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'num_training_samples': NUM_TRAINING_SAMPLES,
            'add_noise': ADD_NOISE,
            'whiten': WHITEN,
            'modified_gravity': MODIFIED_GRAVITY,
            'param_set': PARAM_SET,
            'model_params': MODEL_PARAMS,
        },
    }, MODEL_SAVE_PATH)
    print(f"\n✓ Model saved to: {MODEL_SAVE_PATH}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Data dimension: {pycbc_data.shape[1]}, Training samples: {len(pycbc_data)}")
    print("\n  Next step: run plot_results.py")
