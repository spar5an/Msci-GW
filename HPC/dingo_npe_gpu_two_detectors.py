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
                 num_flow_layers=6, hidden_dim=128, device=None, embedding_type='simple', 
                 time_delay_value=0.0):
        super().__init__()
        
        # Store embedding type as attribute
        self.embedding_type = embedding_type
        self.time_delay_value = time_delay_value
        
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
        
        # Adjust context dimension for flow to account for appended time delay
        flow_context_dim = context_dim + 1  # +1 for time delay
        
        self.flow = NormalizingFlow(
            param_dim=param_dim,
            context_dim=flow_context_dim,
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
        
        # Append time delay as an additional context feature
        batch_size = context.shape[0]
        time_delay_tensor = torch.full((batch_size, 1), self.time_delay_value, 
                                       dtype=context.dtype, device=context.device)
        context = torch.cat([context, time_delay_tensor], dim=1)
        
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
            
            # Append time delay as an additional context feature
            batch_size = context.shape[0]
            time_delay_tensor = torch.full((batch_size, 1), self.time_delay_value, 
                                           dtype=context.dtype, device=context.device)
            context = torch.cat([context, time_delay_tensor], dim=1)
            samples = self.flow.sample(context, num_samples=num_samples)
        return samples

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

def infer_with_dingo(model, observed_data, num_samples=5000, param_norm_info=None, param_names=None):
    """
    Perform inference using the DINGO-style model
    
    Args:
        model: trained DINGO model
        observed_data: observed sine wave [data_dim]
        num_samples: number of posterior samples
        param_norm_info: dict with normalization info for each parameter
        param_names: list of parameter names (e.g., ['mass1', 'mass2', 'spin1z'])
    
    Returns:
        samples: posterior samples [num_samples, param_dim] in physical parameter space (denormalized)
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
    
    # DENORMALIZATION FIX (2026-02-06): Convert from normalized to physical space
    # To revert: set DENORMALIZE_PARAMETERS = False in main config section
    if param_norm_info is not None and param_names is not None and DENORMALIZE_PARAMETERS:
        # Convert from normalized space to physical space
        for j, param_name in enumerate(param_names):
            if param_name in param_norm_info:
                info = param_norm_info[param_name]
                original_mean = info['mean']
                original_std = info['std']
                # Denormalize: physical_value = normalized_value * std + mean
                samples[:, j] = samples[:, j] * original_std + original_mean
    
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


def denormalize_params(normalized_params, param_norm_info, param_names):
    """
    Convert normalized parameters back to physical space
    
    ADDED (2026-02-06): Revert by not calling this function when DENORMALIZE_PARAMETERS=False
    
    Args:
        normalized_params: array or tensor of normalized parameters [shape: (..., param_dim)]
        param_norm_info: dict with normalization info for each parameter
        param_names: list of parameter names
    
    Returns:
        physical_params: denormalized parameters
    """
    if isinstance(normalized_params, torch.Tensor):
        physical_params = normalized_params.clone()
    else:
        physical_params = np.array(normalized_params, copy=True)
    
    for j, param_name in enumerate(param_names):
        if param_name in param_norm_info:
            info = param_norm_info[param_name]
            original_mean = info['mean']
            original_std = info['std']
            # Denormalize: physical_value = normalized_value * std + mean
            if isinstance(physical_params, torch.Tensor):
                physical_params[..., j] = physical_params[..., j] * original_std + original_mean
            else:
                physical_params[..., j] = physical_params[..., j] * original_std + original_mean
    
    return physical_params


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


def calculate_detector_time_delay(ra: float, dec: float, det1: str = 'H1', det2: str = 'L1') -> float:
    """
    Calculate the GPS time delay between two detectors for a gravitational wave arriving from (ra, dec).
    Uses PyCBC's built-in time_delay_from_earth_center method.

    Returns
        Time delay in seconds (det2 relative to det1)
    """
    try:
        from pycbc.detector import Detector
        
        detector1 = Detector(det1)
        detector2 = Detector(det2)
        
        # PyCBC's built-in method to calculate time delay
        delay = detector1.time_delay_from_earth_center(detector2, ra, dec)
        
        return float(delay)
    except Exception as e:
        print(f"Warning: Failed to calculate time delay: {e}. Returning 0.0")
        return 0.0


def prepare_pycbc_data(num_samples=10000):
    config = {
        'mass1': lambda size: np.random.uniform(10, 50, size=size),
        'mass2': lambda size: np.random.uniform(10, 50, size=size),
        'spin1z': lambda size: np.random.uniform(-0.5, 0.5, size=size),
        'spin2z': lambda size: np.random.uniform(-0.5, 0.5, size=size),
    }
    
    # GPS time delay will be calculated at default sky location (north pole)
    default_ra = 0.0
    default_dec = np.pi / 2.0
    time_delay_default = calculate_detector_time_delay(default_ra, default_dec, 'H1', 'L1')

    print(f"\nCalling pycbc_data_generator with {num_samples} samples...")
    print(f"  GPS time delay (north pole): {time_delay_default*1000:.3f} ms")
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

    # Access the loaders and metadata
    print("Accessing dataloaders and metadata from result...")
    try:
        train_loader = result['train_loader']
        val_loader = result['val_loader']
        test_loader = result['test_loader']
        metadata = result['metadata']
        param_norm_info = metadata['parameter_normalization']
        print(f"✓ Got dataloaders: train={len(train_loader)} batches, val={len(val_loader)} batches, test={len(test_loader)} batches")
        print(f"✓ Parameter normalization info stored")
    except Exception as e:
        print(f"❌ Error accessing dataloaders/metadata: {e}")
        print(f"Result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
        raise

    print("\nProcessing training data from dataloaders...")
    print("  (TWO-DETECTOR MODE: Concatenating H1 and L1 streams)")
    data = []
    params = []
    for i, (waveforms, batch_params) in enumerate(train_loader):
        if i % 1000 == 0:
            print(f"  Processing batch {i}/{len(train_loader)}")
        train_params = torch.FloatTensor(batch_params)
        train_data = torch.FloatTensor(waveforms)
        # Extract H1 and L1 channels (indices 0 and 1) and concatenate along feature dimension
        h1_data = train_data[:, 0, :].reshape(train_data.shape[0], -1)
        l1_data = train_data[:, 1, :].reshape(train_data.shape[0], -1)
        concatenated_data = torch.cat([h1_data, l1_data], dim=1)  # (batch, 2*time_steps)
        
        data.append(concatenated_data)
        params.append(train_params)
    all_data = torch.cat(data, dim=0)
    all_params = torch.cat(params, dim=0)
    print(f"  Training data: {all_data.shape}, params: {all_params.shape}")

    print("\nProcessing test data from dataloaders...")
    print("  (TWO-DETECTOR MODE: Concatenating H1 and L1 streams)")
    data_test = []
    params_test = []
    for i, (waveforms, batch_params) in enumerate(test_loader):
        if i % 100 == 0:
            print(f"  Processing test batch {i}/{len(test_loader)}")
        test_params = torch.FloatTensor(batch_params)
        test_data = torch.FloatTensor(waveforms)
        # Extract H1 and L1 channels (indices 0 and 1) and concatenate along feature dimension
        h1_data = test_data[:, 0, :].reshape(test_data.shape[0], -1)
        l1_data = test_data[:, 1, :].reshape(test_data.shape[0], -1)
        concatenated_data = torch.cat([h1_data, l1_data], dim=1)  # (batch, 2*time_steps)
        
        data_test.append(concatenated_data)
        params_test.append(test_params)
    all_test_data = torch.cat(data_test, dim=0)
    all_test_params = torch.cat(params_test, dim=0)
    print(f"  Test data: {all_test_data.shape}, params: {all_test_params.shape}")
    print("✓ Data preparation complete (two-detector concatenation)\n")
    
    # Store time_delay_default for later use during embedding
    return all_data, all_params, all_test_data, all_test_params, param_norm_info, time_delay_default



#Configure EVERYTHING -------------------------------------------------------------------------------------------------------------------------------------------
DENORMALIZE_PARAMETERS = True

# Configure training data size
NUM_TRAINING_SAMPLES = 50000  # Adjust this to control dataset size

# Model architecture parameters
PARAM_DIM = 4               
CONTEXT_DIM = 512           
NUM_FLOW_LAYERS = 5         
HIDDEN_DIM = 128            
EMBEDDING_TYPE = 'simple'  # 'simple', 'conv1d', or 'lstm'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training parameters
NUM_EPOCHS = 200             
BATCH_SIZE = 64            
LEARNING_RATE = 1e-4

# ====================================================================================
# PRINT CONFIGURATION (read from this script, not from PBS)
# ====================================================================================
print("\n" + "="*70)
print("MODEL ARCHITECTURE PARAMETERS")
print("="*70)
print(f"  Parameter Dimension (PARAM_DIM):        {PARAM_DIM}")
print(f"  Context Dimension (CONTEXT_DIM):        {CONTEXT_DIM}")
print(f"  Number of Flow Layers (NUM_FLOW_LAYERS): {NUM_FLOW_LAYERS}")
print(f"  Hidden Dimension (HIDDEN_DIM):          {HIDDEN_DIM}")
print(f"  Embedding Type (EMBEDDING_TYPE):        {EMBEDDING_TYPE}")

print("\n" + "="*70)
print("TRAINING PARAMETERS")
print("="*70)
print(f"  Number of Epochs (NUM_EPOCHS):          {NUM_EPOCHS}")
print(f"  Batch Size (BATCH_SIZE):                {BATCH_SIZE}")
print(f"  Learning Rate (LEARNING_RATE):          {LEARNING_RATE}")
print(f"  Training Samples (NUM_TRAINING_SAMPLES): {NUM_TRAINING_SAMPLES:,}")
print(f"  Denormalize Output (DENORMALIZE_PARAMETERS): {DENORMALIZE_PARAMETERS}")

print("\n" + "="*70)
print("COMPUTED METRICS")
print("="*70)
estimated_batches_per_epoch = NUM_TRAINING_SAMPLES // BATCH_SIZE
total_batches = estimated_batches_per_epoch * NUM_EPOCHS
print(f"  Estimated Batches per Epoch:            {estimated_batches_per_epoch}")
print(f"  Total Batch Updates:                    {total_batches}")
print(f"  Physics Parameters:                     mass1, mass2, spin1z")
print("="*70 + "\n")

pycbc_data, pycbc_params, pycbc_test_data, pycbc_test_params, param_norm_info, GPS_TIME_DELAY = prepare_pycbc_data(num_samples=NUM_TRAINING_SAMPLES)

# Parameters are already normalized by the data generator (z-score normalization)
print(f"Using normalized parameters from data generator")
print(f"  Training samples: {len(pycbc_params)}")
print(f"  Test samples: {len(pycbc_test_params)}")
print(f"  Data dimension: {pycbc_data.shape[1]} (2 detectors concatenated)")
print(f"  GPS time delay: {GPS_TIME_DELAY*1000:.3f} ms (will be appended after embedding)")
print(f"  Normalization info stored for denormalization\n")     

# Train DINGO model on PyCBC data
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
param_names = ['mass1', 'mass2', 'spin1z', 'spin2z']
mean_errors = {param: [] for param in param_names}
mean_differences = {param: [] for param in param_names}

print(f"\nInferring posteriors for {num_test_samples} test samples...")
if DENORMALIZE_PARAMETERS:
    print(f"Denormalizing to physical parameter space...")
else:
    print(f"Computing errors in NORMALIZED space (original behavior)...")
    
for i, test_idx in enumerate(test_indices):
    if i % 100 == 0:
        print(f"  Processing sample {i+1}/{num_test_samples}")
    
    observed_data = pycbc_test_data[test_idx].numpy()
    true_params_normalized = pycbc_test_params[test_idx].numpy()
    
    # DENORMALIZATION FIX: Denormalize true parameters if enabled
    if DENORMALIZE_PARAMETERS:
        true_params = denormalize_params(true_params_normalized, param_norm_info, param_names)
    else:
        true_params = true_params_normalized
    
    # Generate posterior samples (DENORMALIZED if DENORMALIZE_PARAMETERS=True)
    posterior_samples, stats = infer_with_dingo(model, observed_data, num_samples=10000, 
                                               param_norm_info=param_norm_info if DENORMALIZE_PARAMETERS else None,
                                               param_names=param_names if DENORMALIZE_PARAMETERS else None)
    
    # Calculate mean differences (in PHYSICAL space if denormalized, or NORMALIZED space if not)
    for param_idx in range(4):
        param_samples = posterior_samples[:, param_idx]
        true_val = true_params[param_idx]
        inferred_mean = np.mean(param_samples)
        error = inferred_mean - true_val  # Signed difference
        abs_error = abs(error)
        
        mean_errors[param_names[param_idx]].append(abs_error)
        mean_differences[param_names[param_idx]].append(error)

print(f"\n✓ Completed inference on {num_test_samples} samples")

# Print summary statistics
if DENORMALIZE_PARAMETERS:
    print("\nParameter Inference Summary (1000 samples in PHYSICAL SPACE):")
else:
    print("\nParameter Inference Summary (1000 samples in NORMALIZED SPACE):")
    
for param_idx, param in enumerate(param_names):
    errors = mean_errors[param]
    diffs = mean_differences[param]
    print(f"\n{param}:")
    print(f"  Mean absolute error: {np.mean(errors):.4f}")
    print(f"  Std dev of errors:   {np.std(errors):.4f}")
    print(f"  Min error:           {np.min(errors):.4f}")
    print(f"  Max error:           {np.max(errors):.4f}")
    print(f"  Median error:        {np.median(errors):.4f}")

# Store posterior samples for selected samples to display in additional rows
sample_indices = [0, 250, 500, 750, 900]  # Select 5 samples to display
sample_posteriors = {}
if DENORMALIZE_PARAMETERS:
    print(f"\nGenerating posterior samples for visualization (samples {sample_indices}, DENORMALIZED)...")
else:
    print(f"\nGenerating posterior samples for visualization (samples {sample_indices}, NORMALIZED SPACE)...")
    
for sample_idx in sample_indices:
    observed_data = pycbc_test_data[sample_idx].numpy()
    posterior_samples, stats = infer_with_dingo(model, observed_data, num_samples=10000,
                                               param_norm_info=param_norm_info if DENORMALIZE_PARAMETERS else None,
                                               param_names=param_names if DENORMALIZE_PARAMETERS else None)
    sample_posteriors[sample_idx] = posterior_samples
    print(f"  ✓ Generated posteriors for sample {sample_idx}")

# Create visualization with 6 rows x 4 columns (summary + 5 sample posteriors)
fig, axes = plt.subplots(6, 4, figsize=(20, 20))

# Prepare model information text box
total_params = sum(p.numel() for p in model.parameters())
model_info_text = (
    f"Model Architecture:\n"
    f"  Embedding: {model.embedding_type.upper()}\n"
    f"  Flow Layers: {len(model.flow.layers)}\n"
    f"  Input Dim (H1+L1): {pycbc_data.shape[1]}\n"
    f"  Context Dim: {model.flow.context_dim}\n"
    f"  Hidden Dim: {model.flow.layers[0].hidden_dim}\n"
    f"  Total Parameters: {total_params:,}\n"
    f"  Two-Detector Mode: H1+L1 concatenated"
)

# Add text box to the first subplot
axes[0, 0].text(0.02, 0.98, model_info_text, transform=axes[0, 0].transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Row 0: Summary statistics from all 1000 samples
for param_idx, param in enumerate(param_names):
    ax = axes[0, param_idx]
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

# Rows 1-5: Posterior samples from selected test samples
for row_idx, sample_idx in enumerate(sample_indices):
    posterior_samples = sample_posteriors[sample_idx]
    true_params_normalized = pycbc_test_params[sample_idx].numpy()
    
    # DENORMALIZATION FIX: Denormalize true parameters if enabled for display
    if DENORMALIZE_PARAMETERS:
        true_params = denormalize_params(true_params_normalized, param_norm_info, param_names)
    else:
        true_params = true_params_normalized
    
    for param_idx in range(4):
        ax = axes[row_idx + 1, param_idx]
        param_samples = posterior_samples[:, param_idx]  # Already denormalized if DENORMALIZE_PARAMETERS=True
        true_val = true_params[param_idx]
        
        # Plot histogram of posterior samples
        ax.hist(param_samples, bins=50, alpha=0.7, color='darkgreen', edgecolor='black', density=True)
        
        # Mark true value
        ax.axvline(true_val, color='red', linestyle='--', linewidth=2, label=f'True: {true_val:.4f}')
        
        # Mark mean of posterior
        posterior_mean = np.mean(param_samples)
        ax.axvline(posterior_mean, color='orange', linestyle='-', linewidth=2, label=f'Inferred: {posterior_mean:.4f}')
        
        # Labels and title
        param_label = f'{param_names[param_idx]} (M$_\odot$)' if param_idx < 2 else f'{param_names[param_idx]}'
        if DENORMALIZE_PARAMETERS:
            ax.set_xlabel(f'{param_label} Value (physical space)', fontsize=11)
        else:
            ax.set_xlabel(f'{param_label} Value (normalized space)', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{param_label} Posterior (Sample {sample_idx})', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3)

# Create Plots directory if it doesn't exist
import os
from datetime import datetime
os.makedirs("Plots", exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plot_filename = f"Plots/PyCBC_Parameter_Inference_TwoDetector_{timestamp}.png"

try:
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {plot_filename}")
except Exception as e:
    print(f"\n✗ Failed to save plot: {e}")
    import traceback
    traceback.print_exc()
print("=" * 80)
