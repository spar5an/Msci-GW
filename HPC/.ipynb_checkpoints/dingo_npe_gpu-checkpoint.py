#11:24

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg')  # Non-interactive backend for HPC
from scipy import stats
import math
import time
from multiprocessing import Pool
import data_generator_copy as data_generator

# Set random seed for reproducibility
#torch.manual_seed(42)
#np.random.seed(42)

print("Libraries imported successfully")
print(f"PyTorch version: {torch.__version__}")

# GPU/Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

pi = np.pi

import gc
import psutil
import os

def get_memory_usage():
    """
    Get current memory usage for CPU and GPU
    """
    # CPU memory
    process = psutil.Process(os.getpid())
    cpu_mem_gb = process.memory_info().rss / (1024**3)
    
    # GPU memory
    gpu_mem_gb = 0
    if torch.cuda.is_available():
        gpu_mem_gb = torch.cuda.memory_allocated() / (1024**3)
    
    return cpu_mem_gb, gpu_mem_gb

def print_memory_status():
    """
    Print current memory usage
    """
    cpu_mem, gpu_mem = get_memory_usage()
    print(f"\n{'='*60}")
    print("MEMORY STATUS:")
    print(f"  CPU Memory: {cpu_mem:.2f} GB")
    if torch.cuda.is_available():
        print(f"  GPU Memory: {gpu_mem:.2f} GB")
    print(f"{'='*60}\n")

def clear_memory(verbose=True):
    """
    Clear Python memory cache, garbage collection, and PyTorch cache
    
    Args:
        verbose: if True, print memory before/after
    """
    if verbose:
        print("\nClearing memory...")
        cpu_before, gpu_before = get_memory_usage()
        print(f"  Before: CPU={cpu_before:.2f}GB", end="")
        if torch.cuda.is_available():
            print(f", GPU={gpu_before:.2f}GB", end="")
        print()
    
    # Python garbage collection
    gc.collect()
    
    # PyTorch GPU cache
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


# ================================================================================
# ## Normalizing Flow Components
# 
# DINGO uses **normalizing flows** to transform a simple base distribution (e.g., Gaussian) into a complex posterior distribution.
# ================================================================================


class AffineCouplingLayer(nn.Module):
    """
    Affine coupling layer for normalizing flows
    
    Splits input, transforms one half conditioned on the other:
    x2_new = x2 * exp(s(x1, context)) + t(x1, context)
    """
    def __init__(self, dim, context_dim, hidden_dim=128, mask_type='half'):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim  # Store for later access
        
        # Create mask (which dimensions to transform)
        self.register_buffer('mask', torch.zeros(dim))
        if mask_type in ['half', 'even']:
            self.mask[::2] = 1
        elif mask_type == 'odd':
            self.mask[1::2] = 1
        
        # IMPROVED: Deeper networks with batch normalization
        # Scale network - learns multiplicative transformation
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
            nn.Tanh()  # Bounded to [-1, 1] for numerical stability
        )
        
        # Translation network - learns additive transformation
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
        """
        Forward (data -> latent) or reverse (latent -> data) transformation
        
        Args:
            x: input tensor [batch_size, dim]
            context: conditioning context (embedded data) [batch_size, context_dim]
            reverse: if True, compute inverse transformation
        
        Returns:
            output: transformed tensor
            log_det: log determinant of Jacobian
        """
        masked_x = x * self.mask
        
        scale_input = torch.cat([masked_x, context], dim=1)
        translation_input = torch.cat([masked_x, context], dim=1)
        
        s = self.scale_net(scale_input)
        t = self.translation_net(translation_input)
        
        # s is in [-1, 1] from Tanh, exp(s) in [0.368, 2.718]
        # This provides controlled scale expansion for numerical stability
        s = s * (1 - self.mask)
        t = t * (1 - self.mask)
        
        if not reverse:
            y = x * torch.exp(s) + t
            log_det = s.sum(dim=1)
        else:
            y = (x - t) * torch.exp(-s)
            log_det = -s.sum(dim=1)
        
        return y, log_det

print("✓ Improved coupling layer with batch normalization defined")

class NormalizingFlow(nn.Module):   # Just the normalising flow class. 
    """
    Normalizing flow: stack of coupling layers
    Transforms base distribution into complex posterior
    """
    def __init__(self, param_dim=1, context_dim=64, num_layers=6, hidden_dim=128):
        super().__init__()
        self.param_dim = param_dim
        self.context_dim = context_dim
        
        # Stack of coupling layers with alternating masks
        self.layers = nn.ModuleList([
            AffineCouplingLayer(
                dim=param_dim,
                context_dim=context_dim,
                hidden_dim=hidden_dim,
                mask_type='even' if i % 2 == 0 else 'odd'
            )
            for i in range(num_layers)
        ])
        
        # Base distribution: standard Gaussian 
        self.register_buffer('base_mean', torch.zeros(param_dim))
        self.register_buffer('base_std', torch.ones(param_dim))
    
    def forward(self, params, context):
        """
        Forward pass: compute log probability of parameters given context
        
        Args:
            params: parameter values [batch_size, param_dim]
            context: embedded observed data [batch_size, context_dim]
        
        Returns:
            log_prob: log p(params | context)
        """
        z = params
        log_det_sum = 0
        
        # Apply flow transformations
        for layer in self.layers:
            z, log_det = layer(z, context, reverse=False)
            log_det_sum += log_det
        
        # Compute log probability under base distribution
        log_prob_base = -0.5 * (torch.log(2 * np.pi * self.base_std**2) + 
                                 ((z - self.base_mean) / self.base_std)**2)
        log_prob_base = log_prob_base.sum(dim=1)
        
        # Apply change of variables
        log_prob = log_prob_base + log_det_sum
        
        return log_prob
    
    def sample(self, context, num_samples=1):
        """
        Sample from posterior p(params | context)
        
        Args:
            context: embedded observed data [batch_size, context_dim]
            num_samples: number of samples per context
        
        Returns:
            samples: parameter samples [batch_size * num_samples, param_dim]
        """
        batch_size = context.shape[0]
        
        # Repeat context for multiple samples
        context_repeated = context.repeat_interleave(num_samples, dim=0)
        
        # Sample from base distribution
        z = torch.randn(batch_size * num_samples, self.param_dim, device=context.device)
        
        # Apply inverse flow transformations
        for layer in reversed(self.layers):
            z, _ = layer(z, context_repeated, reverse=True)
        
        return z

print("✓ Normalizing flow defined")

class EmbeddingNetwork(nn.Module):  # All this class does is take a data vector of length 100 (the noisy sine wave) and compress it into a context vector of length 64
    """
    Neural network to embed observed data into context vector
    Similar to DINGO's data compression network
    """
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
        """
        Args:
            data: observed data [batch_size, data_dim]
        
        Returns:
            context: embedded representation [batch_size, context_dim]
        """
        return self.network(data)

print("✓ Embedding network defined")

class Conv1DEmbeddingNetwork(nn.Module):

    def __init__(self, data_dim=5868, context_dim=512, num_filters=[64, 128, 256]):
        super().__init__()
        self.data_dim = data_dim
        self.context_dim = context_dim
        
        # Conv layers with batch norm and ReLU
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
        
        # Global average pooling (adaptive to handle variable sizes)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Final dense layer
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
        """
        Args:
            data: observed waveform [batch_size, data_dim]
        
        Returns:
            context: embedded representation [batch_size, context_dim]
        """
        # Reshape to add channel dimension: (batch, 1, data_dim)
        x = data.unsqueeze(1)
        
        # Apply conv layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Global average pooling: (batch, channels, 1)
        x = self.global_pool(x)
        
        # Flatten: (batch, channels)
        x = x.view(x.size(0), -1)
        
        # Final dense layer to context
        context = self.fc(x)
        
        return context


class LSTMEmbeddingNetwork(nn.Module):
    """
    LSTM-based embedding network for waveform data
    
    Better at capturing temporal dependencies in waveforms
    """
    def __init__(self, data_dim=7241, context_dim=512, hidden_dim=256, num_layers=2):
        super().__init__()
        self.data_dim = data_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        
        # Initial projection to embed scalar values
        self.input_proj = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU()
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, context_dim),
            nn.LayerNorm(context_dim)
        )
    
    def forward(self, data):
        """
        Args:
            data: observed waveform [batch_size, data_dim]
        
        Returns:
            context: embedded representation [batch_size, context_dim]
        """
        batch_size = data.size(0)
        
        # Reshape to (batch_size, data_dim, 1) for input_proj
        x = data.unsqueeze(-1)  # [batch, data_dim, 1]
        
        # Project each time point: [batch, data_dim, 32]
        x = self.input_proj(x)
        
        # LSTM expects (batch, seq_len, features)
        # x is already (batch, data_dim, 32) which is correct
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: [batch, data_dim, hidden_dim*2]
        
        # Use final hidden state from both directions
        h_forward = h_n[-2, :, :]  # [batch, hidden_dim]
        h_backward = h_n[-1, :, :]  # [batch, hidden_dim]
        final_state = torch.cat([h_forward, h_backward], dim=1)  # [batch, hidden_dim*2]
        
        # Project to context dimension
        context = self.output_proj(final_state)
        
        return context


class DINGOModel(nn.Module):
    """
    Complete DINGO-style neural posterior estimation model
    
    Architecture:
    observed_data -> EmbeddingNet -> context -> NormalizingFlow -> log p(params | data)
    
    Improvements for multi-mode inference:
    - Larger embedding network (captures more information from data)
    - Deeper flow (better approximation of complex posteriors)
    - Mode-aware context (separate embeddings for different aspects of signal)
    - GPU support for accelerated training
    """
    def __init__(self, data_dim=100, param_dim=1, context_dim=64, 
                 num_flow_layers=6, hidden_dim=128, device=None, embedding_type='simple'):
        super().__init__()
        
        # Store embedding type as attribute
        self.embedding_type = embedding_type
        
        # Choose embedding architecture based on embedding_type
        if embedding_type == 'lstm':
            # Use LSTM for large sequential data
            self.embedding_net = LSTMEmbeddingNetwork(
                data_dim=data_dim,
                context_dim=context_dim,
                hidden_dim=256,
                num_layers=2
            )
        elif embedding_type == 'conv1d':
            # Use Conv1D for large data
            self.embedding_net = Conv1DEmbeddingNetwork(
                data_dim=data_dim,
                context_dim=context_dim,
                num_filters=[64, 128, 256]
            )
        elif embedding_type == 'simple':
            # Use fully-connected for small data
            self.embedding_net = nn.Sequential(
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
        
        self.flow = NormalizingFlow(
            param_dim=param_dim,
            context_dim=context_dim,
            num_layers=num_flow_layers,
            hidden_dim=hidden_dim
        )
    
    def forward(self, params, data):
        """
        Compute log probability of parameters given data
        
        Args:
            params: parameter values [batch_size, param_dim]
            data: observed data [batch_size, data_dim]
        
        Returns:
            log_prob: log p(params | data)
        """
        context = self.embedding_net(data)
        log_prob = self.flow(params, context)
        return log_prob
    
    def sample_posterior(self, data, num_samples=1000):
        """
        Sample from posterior p(params | data)
        
        Args:
            data: observed data [batch_size, data_dim]
            num_samples: number of samples to draw
        
        Returns:
            samples: posterior samples [batch_size * num_samples, param_dim]
        """
        self.eval()
        with torch.no_grad():
            context = self.embedding_net(data)
            samples = self.flow.sample(context, num_samples=num_samples)
        return samples

def train_dingo_model(model, train_amplitudes_and_phases, train_data, 
                      num_epochs=100, batch_size=256, lr=3e-4, use_mixed_precision=True):
    """
    Train the DINGO-style model with improved optimization for multi-mode signals
    
    Args:
        use_mixed_precision: Use torch.cuda.amp for faster training (requires GPU)
    """
    # OPTIMIZATION: Move model and all data to GPU once at start
    model = model.to(DEVICE)
    train_amplitudes_and_phases = train_amplitudes_and_phases.to(DEVICE)
    train_data = train_data.to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr * 0.01
    )
    
    # Mixed precision training scaler (for GPU)
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() and use_mixed_precision else None
    
    # OPTIMIZATION: Enable cudnn auto-tuner for faster operations
    torch.backends.cudnn.benchmark = True
    
    num_simulations = len(train_amplitudes_and_phases)
    
    print(f"Training improved DINGO-style model for {num_epochs} epochs...\n")
    
    losses = []
    best_loss = -float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        # OPTIMIZATION: Create weighted sampling on GPU
        num_modes = (train_amplitudes_and_phases > 0).sum(dim=1)
        
        # Create weighted sampling distribution (higher weight for 4-5 modes)
        mode_weights = torch.ones_like(num_modes, dtype=torch.float32)
        mode_weights[num_modes == 4] = 1.5
        mode_weights[num_modes == 5] = 2.0
        
        # Sample with replacement using weights
        indices = torch.multinomial(mode_weights, num_simulations, replacement=True)
        
        for i in range(0, num_simulations, batch_size):
            batch_indices = indices[i:min(i+batch_size, num_simulations)]
            batch_data = train_data[batch_indices]
            batch_amplitudes_and_phases = train_amplitudes_and_phases[batch_indices]
            
            optimizer.zero_grad()
            
            # Mixed precision training (speeds up GPU computation)
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    # Forward pass: compute log probability
                    log_prob = model(batch_amplitudes_and_phases, batch_data)
                    # Loss: negative log likelihood
                    loss = -log_prob.mean()
                
                # Backward pass with mixed precision
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training on CPU
                log_prob = model(batch_amplitudes_and_phases, batch_data)
                loss = -log_prob.mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            
            optimizer.zero_grad()  # Moved after step for better GPU pipelining
            
            epoch_loss += -loss.item()
            num_batches += 1
        
        avg_log_prob = epoch_loss / num_batches
        losses.append(avg_log_prob)
        
        # Learning rate scheduling
        scheduler.step()
        
        if avg_log_prob > best_loss:
            best_loss = avg_log_prob
        
        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{num_epochs}, Avg Log Prob: {avg_log_prob:.4f}, Best: {best_loss:.4f}, LR: {current_lr:.2e}")
    
    print("\n Training complete!")
    return losses

def train_dingo_model_pycbc(model, train_params, train_data, 
                             num_epochs=100, batch_size=256, lr=1e-4, use_mixed_precision=True):
    # Move model and data to device
    model = model.to(DEVICE)
    train_params = train_params.to(DEVICE)
    train_data = train_data.to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr * 0.01
    )
    
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() and use_mixed_precision else None
    torch.backends.cudnn.benchmark = True
    
    num_samples = len(train_params)
    print(f"\nTraining PyCBC DINGO model for {num_epochs} epochs...")
    print(f"  Samples: {num_samples}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}\n")
    
    losses = []
    best_loss = -float('inf')
    patience_counter = 0
    patience = 20
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0
        batch_losses = []
        
        # Shuffle indices
        indices = torch.randperm(num_samples, device=DEVICE)
        
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:min(i + batch_size, num_samples)]
            batch_params = train_params[batch_indices]
            batch_data = train_data[batch_indices]
            
            optimizer.zero_grad()
            
            if scaler is not None:
                # Mixed precision training
                with torch.cuda.amp.autocast():
                    log_prob = model(batch_params, batch_data)
                    loss = -log_prob.mean()
                    
                    # Get embedding for regularization
                    context = model.embedding_net(batch_data)
                    # Regularize: context should have non-zero variance across batch
                    # This forces embedding to use the input data
                    context_std = context.std(dim=0).mean()
                    reg_loss = 10.0 * torch.clamp(1.5 - context_std, min=0)  # Target context_std > 1.5
                    
                    total_loss = loss + reg_loss
                
                scaler.scale(total_loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training
                log_prob = model(batch_params, batch_data)
                loss = -log_prob.mean()
                
                # Embedding regularization - force context variance
                context = model.embedding_net(batch_data)
                context_std = context.std(dim=0).mean()
                reg_loss = 10.0 * torch.clamp(1.5 - context_std, min=0)  # Target context_std > 1.5
                total_loss = loss + reg_loss
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            epoch_loss += -loss.item()
            batch_losses.append(-loss.item())
            num_batches += 1
        
        avg_log_prob = epoch_loss / num_batches
        losses.append(avg_log_prob)
        
        # Monitor context variance on last batch
        with torch.no_grad():
            context_check = model.embedding_net(batch_data)
            context_std_check = context_check.std(dim=0).mean().item()
        
        scheduler.step()
        
        if avg_log_prob > best_loss:
            best_loss = avg_log_prob
        #    patience_counter = 0
        #else:
        #    patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            batch_std = np.std(batch_losses)
            print(f"Epoch {epoch+1:3d}/{num_epochs}, Avg Log Prob: {avg_log_prob:7.4f}, Best: {best_loss:7.4f}, Std: {batch_std:6.4f}, Context_std: {context_std_check:6.4f}, LR: {current_lr:.2e}, Patience: {patience_counter}/{patience}")
        
        # Early stopping
        #if patience_counter >= patience:
        #    print(f"\n⚠ Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
        #    break
    
    print("\n LSTM-based training complete!")
    return losses

def infer_with_dingo(model, observed_data, num_samples=5000):
    """
    Perform inference using the DINGO-style model
    
    Args:
        model: trained DINGO model
        observed_data: observed sine wave [data_dim]
        num_samples: number of posterior samples
    
    Returns:
        samples: posterior samples [num_samples, param_dim] in physical parameter space
        statistics: dict with mean, median, std, quantiles
    """
    model.eval()
    
    # Prepare data on device - convert to tensor if needed
    if isinstance(observed_data, np.ndarray):
        data_tensor = torch.FloatTensor(observed_data)
    else:
        data_tensor = observed_data
    
    # Add batch dimension: (data_dim,) -> (1, data_dim)
    if data_tensor.dim() == 1:
        data_tensor = data_tensor.unsqueeze(0)
    
    data_tensor = data_tensor.to(DEVICE)
    
    # Sample from posterior
    with torch.no_grad():
        samples = model.sample_posterior(data_tensor, num_samples=num_samples)
        samples = samples.cpu().numpy()  # [num_samples, param_dim]
    
    # Samples are in normalized space (z-score) - keep them as-is
    # No denormalization needed since we're comparing against normalized true values
    
    # For 1D parameters, flatten; for multi-D, keep as is
    if samples.shape[1] == 1:
        samples = samples.flatten()
        # Compute statistics
        statistics = {
            'mean': np.mean(samples),
            'median': np.median(samples),
            'std': np.std(samples),
            'q05': np.percentile(samples, 5),
            'q95': np.percentile(samples, 95),
        }
    else:
        # For multi-dimensional parameters, return samples as-is
        # Statistics computation will be done per parameter
        statistics = None
    
    return samples, statistics


# ================================================================================
# We start with a uniform prior distribution of amplitudes in parameter space (which we generate sine waves with). We pass amplitude-sine wave pairs into the network during training. The network learns to map amplitudes from parameter space to latent space (standard Gaussian N(0,1)), such that correct amplitude-context pairs map near z=0 (high probability) and incorrect pairs map far from z=0 (low probability). Once trained, during inference, we sample from the Gaussian in latent space and map backwards to parameter space to get amplitude samples that are likely given the observed data.
# 
# Once trained, during inference, we sample from the standard Gaussian N(0,1) in latent space. We know that values near z=0 are more likely in latent space. However, we DON'T know what the posterior looks like in parameter space beforehand—that's determined by the learned, context-dependent transformation. By applying the reverse flow (conditioned on the observed data's context), we transform the simple Gaussian into a complex posterior distribution that reflects our uncertainty about the amplitude given that specific observation. The flow doesn't just translate the posterior to a new center—it warps, stretches, and reshapes the distribution based on what it learned during training about which amplitudes are consistent with which observations.
# 
# 
# 
# 
# ### Architecture Summary:
# 
# ```
# Raw Data (100D) 
#     ↓
# Embedding Network
#     ↓
# Context Vector (64D)
#     ↓
# Normalizing Flow (8 layers)
#     ↓
# Posterior p(amplitude | data)
# ```
# 
# This architecture is similar to what's used in real gravitational-wave inference with DINGO!
# ================================================================================


def prepare_pycbc_data(num_samples=10000):
    config = {
        'mass1': lambda size: np.random.uniform(10, 50, size=size),
        'mass2': lambda size: np.random.uniform(10, 50, size=size),
        'spin1z': lambda size: np.random.uniform(-0.5, 0.5, size=size),
    }

    print(f"\nCalling pycbc_data_generator with {num_samples} samples...")
    try:
        # Generate with H1 and L1 projection (default detectors)
        result = data_generator.pycbc_data_generator(
            config, 
            num_samples=num_samples,
            batch_size=16, 
            num_workers=1,  # Minimum 1 worker required
            allow_padding=True,
            chunk_size=5000  # Larger chunks for efficiency
        )
        print("✓ Data generator completed successfully")
    except Exception as e:
        print(f"❌ Error in data_generator: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Access the loaders
    print("Accessing dataloaders from result...")
    try:
        train_loader = result['train_loader']
        val_loader = result['val_loader']
        test_loader = result['test_loader']
        print(f"✓ Got dataloaders: train={len(train_loader)} batches, val={len(val_loader)} batches, test={len(test_loader)} batches")
    except Exception as e:
        print(f"❌ Error accessing dataloaders: {e}")
        print(f"Result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
        raise

    print("\nProcessing training data from dataloaders...")
    data = []
    params = []
    for i, (waveforms, batch_params) in enumerate(train_loader):
        if i % 1000 == 0:
            print(f"  Processing batch {i}/{len(train_loader)}")
        train_params = torch.FloatTensor(batch_params)
        train_data = torch.FloatTensor(waveforms)
        # Extract only H1 channel (index 0) and flatten
        h1_data = train_data[:, 0, :].reshape(train_data.shape[0], -1)
        data.append(h1_data)
        params.append(train_params)
    all_data = torch.cat(data, dim=0)
    all_params = torch.cat(params, dim=0)
    print(f"  Training data: {all_data.shape}, params: {all_params.shape}")

    print("\nProcessing test data from dataloaders...")
    data_test = []
    params_test = []
    for i, (waveforms, batch_params) in enumerate(test_loader):
        if i % 100 == 0:
            print(f"  Processing test batch {i}/{len(test_loader)}")
        test_params = torch.FloatTensor(batch_params)
        test_data = torch.FloatTensor(waveforms)
        # Extract only H1 channel (index 0) and flatten
        h1_data = test_data[:, 0, :].reshape(test_data.shape[0], -1)
        data_test.append(h1_data)
        params_test.append(test_params)
    all_test_data = torch.cat(data_test, dim=0)
    all_test_params = torch.cat(params_test, dim=0)
    print(f"  Test data: {all_test_data.shape}, params: {all_test_params.shape}")
    print("✓ Data preparation complete\n")

    return all_data, all_params, all_test_data, all_test_params



#Configure EVERYTHING -------------------------------------------------------------------------------------------------------------------------------------------

# Configure training data size
NUM_TRAINING_SAMPLES = 10000  # Adjust this to control dataset size

pycbc_data, pycbc_params, pycbc_test_data, pycbc_test_params = prepare_pycbc_data(num_samples=NUM_TRAINING_SAMPLES)

# Parameters are already normalized by the data generator (z-score normalization)
print(f"Using normalized parameters from data generator")
print(f"  Training samples: {len(pycbc_params)}")
print(f"  Test samples: {len(pycbc_test_params)}\n")


# Model architecture parameters
PARAM_DIM = 3               
CONTEXT_DIM = 512           
NUM_FLOW_LAYERS = 4         
HIDDEN_DIM = 128            
EMBEDDING_TYPE = 'lstm'   

# Training parameters
NUM_EPOCHS = 30             
BATCH_SIZE = 64            
LEARNING_RATE = 1e-4        

# Train DINGO model on PyCBC data
model = DINGOModel(
    data_dim=pycbc_data.shape[1],
    param_dim=PARAM_DIM,
    context_dim=CONTEXT_DIM,
    num_flow_layers=NUM_FLOW_LAYERS,
    hidden_dim=HIDDEN_DIM,
    device=DEVICE,
    embedding_type=EMBEDDING_TYPE
)

print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Training on {len(pycbc_data)} samples for {NUM_EPOCHS} epochs")
print(f"Learning rate: {LEARNING_RATE}, Batch size: {BATCH_SIZE}\n")

losses = train_dingo_model_pycbc(
    model, 
    pycbc_params,  # Already normalized by data generator
    pycbc_data, 
    num_epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LEARNING_RATE
)

print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Training on {len(pycbc_data)} samples for {NUM_EPOCHS} epochs\n")


# TESTING PYCBC DATA INFERENCE ------------------------------------------------------------------------------------------------------------------------------------------------
# with 1000 samples
#finds differences between inferred mean and true value for each parameter for every sample
#------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------

print("\n" + "=" * 80)
print("TESTING PYCBC PARAMETER INFERENCE - 1000 SAMPLES")
print("=" * 80)

# Test on 1000 samples from the test set
num_test_samples = min(1000, len(pycbc_test_data))
test_indices = list(range(num_test_samples))

# Collect mean differences for all samples
param_names = ['mass1', 'mass2', 'spin1z']
mean_errors = {param: [] for param in param_names}
mean_differences = {param: [] for param in param_names}

print(f"\nInferring posteriors for {num_test_samples} test samples...")
for i, test_idx in enumerate(test_indices):
    if i % 100 == 0:
        print(f"  Processing sample {i+1}/{num_test_samples}")
    
    observed_data = pycbc_test_data[test_idx].numpy()
    true_params = pycbc_test_params[test_idx].numpy()
    
    # Generate posterior samples (in normalized space)
    posterior_samples, stats = infer_with_dingo(model, observed_data, num_samples=10000)
    
    # Calculate mean differences (in normalized space)
    for param_idx in range(3):
        param_samples = posterior_samples[:, param_idx]
        true_val = true_params[param_idx]
        inferred_mean = np.mean(param_samples)
        error = inferred_mean - true_val  # Signed difference
        abs_error = abs(error)
        
        mean_errors[param_names[param_idx]].append(abs_error)
        mean_differences[param_names[param_idx]].append(error)

print(f"\n✓ Completed inference on {num_test_samples} samples")

# Print summary statistics
print("\nParameter Inference Summary (1000 samples):")
for param_idx, param in enumerate(param_names):
    errors = mean_errors[param]
    diffs = mean_differences[param]
    print(f"\n{param}:")
    print(f"  Mean absolute error: {np.mean(errors):.4f}")
    print(f"  Std dev of errors:   {np.std(errors):.4f}")
    print(f"  Min error:           {np.min(errors):.4f}")
    print(f"  Max error:           {np.max(errors):.4f}")
    print(f"  Median error:        {np.median(errors):.4f}")

# Create visualization with 3 histograms of mean differences
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Prepare model information text box
total_params = sum(p.numel() for p in model.parameters())
model_info_text = (
    f"Model Architecture:\n"
    f"  Embedding: {model.embedding_type.upper()}\n"
    f"  Flow Layers: {len(model.flow.layers)}\n"
    f"  Context Dim: {model.flow.context_dim}\n"
    f"  Hidden Dim: {model.flow.layers[0].hidden_dim}\n"
    f"  Total Parameters: {total_params:,}"
)

# Add text box to the first subplot
axes[0].text(0.02, 0.98, model_info_text, transform=axes[0].transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

for param_idx, param in enumerate(param_names):
    ax = axes[param_idx]
    diffs = mean_differences[param]
    errors = mean_errors[param]
    
    # Plot histogram of differences
    ax.hist(diffs, bins=50, alpha=0.7, color='steelblue', edgecolor='black', density=False)
    
    # Mark zero line (perfect inference)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Perfect Inference (0)')
    
    # Mark mean error
    mean_diff = np.mean(diffs)
    ax.axvline(mean_diff, color='green', linestyle='-', linewidth=2, label=f'Mean: {mean_diff:.4f}')
    
    # Labels and title
    ax.set_xlabel(f'Mean Difference (Inferred - True)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    param_label = f'{param} (M$_\odot$)' if param_idx < 2 else f'{param}'
    ax.set_title(f'{param_label} Inference Errors\n(1000 test samples)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)

# Create Plots directory if it doesn't exist
import os
from datetime import datetime
os.makedirs("Plots", exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plot_filename = f"Plots/PyCBC_Parameter_Inference_Single_Test_{timestamp}.png"

try:
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {plot_filename}")
except Exception as e:
    print(f"\n✗ Failed to save plot: {e}")
    import traceback
    traceback.print_exc()
print("=" * 80)

