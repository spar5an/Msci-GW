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
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
import DataGenerator as data_generator
import DataGeneratorRealPSD as data_generator_real_psd



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
import json
import hashlib

WAVEFORM_CACHE_VERSION = 1

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
















def _make_json_serializable(value):
    if isinstance(value, dict):
        return {str(k): _make_json_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_make_json_serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _sanitize_cache_token(value):
    token = str(value).strip().replace(' ', '_')
    allowed = []
    for char in token:
        if char.isalnum() or char in ('-', '_', '.'):
            allowed.append(char)
        else:
            allowed.append('-')
    sanitized = ''.join(allowed).strip('-_')
    return sanitized or 'unknown'


def get_waveform_cache_dir():
    base_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    cache_dir = os.path.join(base_dir, '..', 'waveform_cache', 'prepared_datasets')
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.abspath(cache_dir)


def build_waveform_cache_spec(num_samples, generator_backend=None, psd_csv=None):
    generator_backend = generator_backend or GENERATOR_BACKEND
    psd_csv = psd_csv if psd_csv is not None else REAL_PSD_CSV
    if generator_backend == 'real_psd' and psd_csv is not None:
        psd_csv = os.path.abspath(psd_csv)

    generator_name = 'pycbc_lorentz_violation_data_generator'

    generator_settings = {
        'generator_backend': generator_backend,
        'generator_name': generator_name,
        'num_samples': int(num_samples),
        'batch_size': int(DATA_GENERATOR_BATCH_SIZE),
        'num_workers': NUM_WORKERS,
        'chunk_size': 5000,
        'detectors': list(DATA_GENERATOR_DETECTORS),
        'add_noise': bool(ADD_NOISE),
        'whiten': bool(WHITEN),
        'theory_mode': THEORY_MODE,
        'modified_gravity': bool(MODIFIED_GRAVITY),
        'lorentz_violation': bool(LORENTZ_VIOLATION),
        'param_set': PARAM_SET,
        'model_params': list(MODEL_PARAMS),
        'use_reparameterized_targets': bool(USE_REPARAMETERIZED_TARGETS),
        'embedding_type': EMBEDDING_TYPE,
        'version': WAVEFORM_CACHE_VERSION,
        'approximant': 'IMRPhenomD',
        'f_lower': 30.0,
        'time_resolution': 1.0 / 4096.0,
        'signal_length': 2.0,
        'f_final': 2048.0,
        'lambda_g_range': [LAMBDA_G_MIN, LAMBDA_G_MAX],
        'alpha_lv': ALPHA_LV,
        'A_lv_range': [A_LV_MIN, A_LV_MAX],
        'psd_csv': psd_csv if generator_backend == 'real_psd' else None,
    }

    serializable_settings = _make_json_serializable(generator_settings)
    settings_json = json.dumps(serializable_settings, sort_keys=True, separators=(',', ':'))
    settings_hash = hashlib.sha256(settings_json.encode('utf-8')).hexdigest()[:12]

    backend_token = _sanitize_cache_token(generator_backend)
    mode_token = THEORY_MODE
    noise_token = 'noisy' if ADD_NOISE else 'clean'
    whiten_token = 'white' if WHITEN else 'raw'
    param_token = _sanitize_cache_token(PARAM_SET)
    filename = (
        f"waveforms_N{int(num_samples)}_{backend_token}_{mode_token}_{noise_token}_{whiten_token}_{param_token}_{settings_hash}.pt"
    )

    return {
        'cache_dir': get_waveform_cache_dir(),
        'cache_path': os.path.join(get_waveform_cache_dir(), filename),
        'settings': serializable_settings,
        'settings_hash': settings_hash,
        'filename': filename,
    }


def load_prepared_waveform_cache(num_samples, generator_backend=None, psd_csv=None):
    if not USE_WAVEFORM_CACHE:
        return None

    cache_spec = build_waveform_cache_spec(num_samples, generator_backend=generator_backend, psd_csv=psd_csv)
    cache_path = cache_spec['cache_path']
    if not os.path.exists(cache_path):
        print(f"No waveform cache found for this configuration: {cache_spec['filename']}")
        return None

    print(f"Loading prepared waveform cache: {cache_path}")
    cache_payload = torch.load(cache_path, map_location='cpu')

    if cache_payload.get('settings_hash') != cache_spec['settings_hash']:
        print("Cache hash mismatch, ignoring cached dataset.")
        return None

    prepared_outputs = cache_payload.get('prepared_outputs')
    if prepared_outputs is None:
        print("Cache file is missing prepared outputs, regenerating dataset.")
        return None

    print("Loaded prepared waveform dataset from cache.")
    return prepared_outputs


def save_prepared_waveform_cache(num_samples, prepared_outputs, generator_metadata, component_param_names, generator_backend=None, psd_csv=None):
    if not USE_WAVEFORM_CACHE:
        return

    cache_spec = build_waveform_cache_spec(num_samples, generator_backend=generator_backend, psd_csv=psd_csv)
    cache_payload = {
        'settings': cache_spec['settings'],
        'settings_hash': cache_spec['settings_hash'],
        'prepared_outputs': prepared_outputs,
        'generator_metadata': generator_metadata,
        'component_param_names': list(component_param_names),
        'cache_version': WAVEFORM_CACHE_VERSION,
    }
    torch.save(cache_payload, cache_spec['cache_path'])
    print(f"Saved prepared waveform cache: {cache_spec['cache_path']}")
















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
            # Use fully-connected embedding
            # LayerNorm on input prevents overflow in the large first linear layer,
            # especially important under mixed-precision (float16) training
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
                             num_epochs=100, batch_size=256, lr=1e-4, use_mixed_precision=True,
                             val_params=None, val_data=None):
    
    # Move model and data to device
    model = model.to(DEVICE)
    train_params = train_params.to(DEVICE)
    train_data = train_data.to(DEVICE)
    
    # Debug: Check for NaN in input data
    if torch.isnan(train_params).any() or torch.isnan(train_data).any():
        print(" ERROR: Input data contains NaN values!")
        print(f"  train_params NaN count: {torch.isnan(train_params).sum().item()}")
        print(f"  train_data NaN count: {torch.isnan(train_data).sum().item()}")
        raise ValueError("Input data contains NaN values")
    
    # Debug: Check data statistics
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
    patience_counter = 0
    patience = 20
    nan_detected = False
    first_batch_debug = True
    
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
            
            try:
                if scaler is not None:
                    # Mixed precision training
                    with torch.amp.autocast('cuda'):
                        log_prob = model(batch_params, batch_data)
                        
                        # Debug first batch
                        if first_batch_debug and epoch == 0:
                            print(f"✓ First batch output check:")
                            print(f"  log_prob shape: {log_prob.shape}")
                            print(f"  log_prob min/max: {log_prob.min():.4f} / {log_prob.max():.4f}")
                            print(f"  log_prob NaN count: {torch.isnan(log_prob).sum().item()}")
                            print(f"  log_prob Inf count: {torch.isinf(log_prob).sum().item()}")
                            first_batch_debug = False
                        
                        loss = -log_prob.mean()
                        
                        # Clamp loss to prevent NaN
                        if torch.isnan(loss) or torch.isinf(loss):
                            print(f"\n⚠ NaN/Inf detected in loss at epoch {epoch+1}, batch {i//batch_size + 1}")
                            print(f"  mean(log_prob) = {log_prob.mean()}")
                            print(f"  loss = {loss}")
                            loss = torch.tensor(0.0, device=DEVICE)
                        
                        # Get embedding for regularization
                        context = model.embedding_net(batch_data)
                        # Regularize: context should have non-zero variance across batch
                        context_std = context.std(dim=0).mean()
                        reg_loss = 10.0 * torch.clamp(1.5 - context_std, min=0)
                        
                        total_loss = loss + reg_loss
                    
                    scaler.scale(total_loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Standard training
                    log_prob = model(batch_params, batch_data)
                    loss = -log_prob.mean()
                    
                    # Clamp loss to prevent NaN
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"\n⚠ NaN/Inf detected in loss at epoch {epoch+1}, batch {i//batch_size + 1}")
                        loss = torch.tensor(0.0, device=DEVICE)
                    
                    # Embedding regularization
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
        
        # Monitor context variance on last batch
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
        try:
            samples = model.sample_posterior(data_tensor, num_samples=num_samples)
            samples_np = samples.cpu().numpy()  # [num_samples, param_dim]
            
            # Check for NaN in samples
            if np.isnan(samples_np).any():
                nan_count = np.isnan(samples_np).sum()
                print(f"  ⚠ Warning: NaN values detected in samples ({nan_count}/{samples_np.size} values)")
                # Replace NaN with 0 (neutral value in normalized space)
                samples_np = np.nan_to_num(samples_np, nan=0.0)
            
            samples = samples_np
        except Exception as e:
            print(f"  ⚠ Warning: Inference failed: {e}")
            # Return dummy samples filled with 0
            samples = np.zeros((num_samples, data_tensor.shape[1] if len(data_tensor.shape) > 1 else 1))
    
    # DENORMALIZATION FIX (2026-02-06): Convert from normalized to physical space
    if param_norm_info is not None and param_names is not None and DENORMALIZE_PARAMETERS:
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
        # Compute statistics - handle NaN gracefully
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
            statistics = {
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0,
                'q05': 0.0,
                'q95': 0.0,
            }
    else:
        # For multi-dimensional parameters, return samples as-is
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


def normalize_params_with_reference(param_array, param_names, reference_norm_info=None):
    """
    Z-score normalize parameter matrix.

    If reference_norm_info is provided, use those fixed stats (for test/inference).
    Otherwise compute stats from the provided array (for training set only).
    """
    arr = np.array(param_array, dtype=np.float32, copy=True)
    normed = np.zeros_like(arr, dtype=np.float32)

    norm_info = {}
    for j, name in enumerate(param_names):
        if reference_norm_info is None:
            mean = float(arr[:, j].mean())
            std = float(arr[:, j].std())
            if std <= 0:
                std = 1.0
        else:
            mean = float(reference_norm_info[name]['mean'])
            std = float(reference_norm_info[name]['std'])
            if std <= 0:
                std = 1.0

        normed[:, j] = (arr[:, j] - mean) / std
        norm_info[name] = {
            'mean': mean,
            'std': std,
            'min': float(arr[:, j].min()),
            'max': float(arr[:, j].max()),
            'method': 'zscore'
        }

    return normed, norm_info


def estimate_redshift_from_luminosity_distance(distance_mpc, h0_km_s_mpc=67.74):
    """Approximate redshift from luminosity distance using low-z Hubble law: z ≈ H0 * D_L / c."""
    if distance_mpc is None or distance_mpc <= 0:
        raise ValueError("distance_mpc must be positive to estimate redshift")
    c_km_s = 299792.458
    z = (h0_km_s_mpc * float(distance_mpc)) / c_km_s
    return float(max(z, 0.0))


def convert_source_to_detector_frame_masses(mass1_source, mass2_source, redshift=None,
                                            distance_mpc=None, h0_km_s_mpc=67.74):
    """
    Convert source-frame component masses to detector-frame masses.

    Uses m_det = m_src * (1 + z). If redshift is not provided, estimates z from
    luminosity distance via low-z Hubble law.
    """
    if redshift is None:
        redshift = estimate_redshift_from_luminosity_distance(distance_mpc, h0_km_s_mpc=h0_km_s_mpc)
    if redshift < 0:
        raise ValueError(f"redshift must be non-negative, got {redshift}")

    scale = 1.0 + float(redshift)
    mass1_detector = float(mass1_source) * scale
    mass2_detector = float(mass2_source) * scale
    return mass1_detector, mass2_detector, float(redshift)


# Extra (non-mass/spin) parameters that are passed through unchanged during
# reparameterization.  This default is overridden by PARAM_SET in the config
# section below.  'symmetric' → [] (4 params),  'component' → ['coa_phase','distance'] (6 params).
# NOTE: This default is overwritten in the config block; kept here for early import safety.
EXTRA_PARAM_NAMES = []

def transform_component_params_to_reparameterized(param_array, param_names):
    """
    Convert component parameters to symmetric target basis:
    [mass1, mass2, spin1z, spin2z, ...extra...] -> [chirp_mass, q, chi_eff, chi_a, ...extra...].
    Extra parameters (coa_phase, distance, etc.) are passed through unchanged.
    """
    required = ['mass1', 'mass2', 'spin1z', 'spin2z']
    missing = [p for p in required if p not in param_names]
    if missing:
        raise ValueError(f"Cannot reparameterize: missing parameters {missing}")

    i_m1 = param_names.index('mass1')
    i_m2 = param_names.index('mass2')
    i_s1 = param_names.index('spin1z')
    i_s2 = param_names.index('spin2z')

    arr = np.array(param_array, dtype=np.float32, copy=True)
    m1_raw = arr[:, i_m1]
    m2_raw = arr[:, i_m2]
    s1_raw = arr[:, i_s1]
    s2_raw = arr[:, i_s2]

    heavy_is_1 = m1_raw >= m2_raw
    m1 = np.where(heavy_is_1, m1_raw, m2_raw)
    m2 = np.where(heavy_is_1, m2_raw, m1_raw)
    s1 = np.where(heavy_is_1, s1_raw, s2_raw)
    s2 = np.where(heavy_is_1, s2_raw, s1_raw)

    total_mass = np.clip(m1 + m2, 1e-8, None)
    chirp_mass = np.power(np.clip(m1 * m2, 1e-12, None), 3.0 / 5.0) / np.power(total_mass, 1.0 / 5.0)
    q = np.clip(m2 / np.clip(m1, 1e-8, None), 1e-4, 1.0)
    chi_eff = (m1 * s1 + m2 * s2) / total_mass
    chi_a = 0.5 * (s1 - s2)

    columns = [chirp_mass, q, chi_eff, chi_a]
    transformed_names = ['chirp_mass', 'q', 'chi_eff', 'chi_a']

    # Pass through extra parameters unchanged
    param_names_list = list(param_names)
    for ep in EXTRA_PARAM_NAMES:
        if ep in param_names_list:
            columns.append(arr[:, param_names_list.index(ep)])
            transformed_names.append(ep)

    transformed = np.stack(columns, axis=1).astype(np.float32)
    return transformed, transformed_names


def transform_component_dict_to_reparameterized(param_dict):
    """Convert a single component-parameter dict to reparameterized dict.
    Reuses the array version to avoid duplicating math."""
    required = ['mass1', 'mass2', 'spin1z', 'spin2z']
    arr = np.array([[param_dict[k] for k in required]], dtype=np.float32)
    transformed, names = transform_component_params_to_reparameterized(arr, required)
    result = {n: float(transformed[0, i]) for i, n in enumerate(names)}
    # Pass through extra parameters unchanged
    for ep in EXTRA_PARAM_NAMES:
        if ep in param_dict:
            result[ep] = float(param_dict[ep])
    return result


def transform_reparameterized_to_component_samples(param_array, param_names):
    """
    Convert [chirp_mass, q, chi_eff, chi_a, ...extra...] samples back to
    [mass1, mass2, spin1z, spin2z, ...extra...] where mass1 >= mass2.
    Extra parameters (coa_phase, distance, etc.) are passed through unchanged.
    """
    required = ['chirp_mass', 'q', 'chi_eff', 'chi_a']
    missing = [p for p in required if p not in param_names]
    if missing:
        raise ValueError(f"Cannot invert reparameterization: missing parameters {missing}")

    idx_mc = param_names.index('chirp_mass')
    idx_q = param_names.index('q')
    idx_ce = param_names.index('chi_eff')
    idx_ca = param_names.index('chi_a')

    arr = np.array(param_array, dtype=np.float32, copy=True)
    mc = np.clip(arr[:, idx_mc], 1e-6, None)
    q = np.clip(arr[:, idx_q], 1e-4, 1.0)
    chi_eff = arr[:, idx_ce]
    chi_a = arr[:, idx_ca]

    total_mass = mc * np.power(1.0 + q, 6.0 / 5.0) / np.power(q, 3.0 / 5.0)
    m1 = total_mass / (1.0 + q)
    m2 = q * m1

    s1 = chi_eff + 2.0 * (m2 / np.clip(total_mass, 1e-8, None)) * chi_a
    s2 = chi_eff - 2.0 * (m1 / np.clip(total_mass, 1e-8, None)) * chi_a

    s1 = np.clip(s1, -0.999, 0.999)
    s2 = np.clip(s2, -0.999, 0.999)

    columns = [m1, m2, s1, s2]
    comp_names = ['mass1', 'mass2', 'spin1z', 'spin2z']

    # Pass through extra parameters unchanged
    param_names_list = list(param_names)
    for ep in EXTRA_PARAM_NAMES:
        if ep in param_names_list:
            columns.append(arr[:, param_names_list.index(ep)])
            comp_names.append(ep)

    comp = np.stack(columns, axis=1).astype(np.float32)
    return comp, comp_names


def convert_samples_to_eval_space(samples, source_param_names, output_component_distributions=False):
    """
    Convert parameter samples to chosen output space.
    """
    if output_component_distributions:
        converted, eval_names = transform_reparameterized_to_component_samples(samples, source_param_names)
        return converted, eval_names
    return samples, list(source_param_names)


def convert_vector_to_eval_space(param_vector, source_param_names, output_component_distributions=False):
    """
    Convert a single parameter vector to chosen output space.
    """
    vector_2d = np.array(param_vector, dtype=np.float32, copy=True).reshape(1, -1)
    converted, eval_names = convert_samples_to_eval_space(
        vector_2d,
        source_param_names,
        output_component_distributions=output_component_distributions
    )
    return converted[0], eval_names


def split_samples_into_symmetric_and_component(samples, source_param_names):
    """
    Return both symmetric and component representations for posterior samples.
    Extra parameters (coa_phase, distance, lambda_g) are included in both outputs.

    If only a subset of symmetric/component params is available (e.g., missing q or chi_a),
    the inverse transform cannot be performed. In that case, the component output is set to
    the same array/names as the symmetric output.
    """
    name_set = set(source_param_names)
    sym_required = {'chirp_mass', 'q', 'chi_eff', 'chi_a'}
    comp_required = {'mass1', 'mass2', 'spin1z', 'spin2z'}

    if sym_required.issubset(name_set):
        symmetric = np.array(samples, dtype=np.float32, copy=True)
        symmetric_names = list(source_param_names)
        component, component_names = transform_reparameterized_to_component_samples(symmetric, source_param_names)
        return symmetric, symmetric_names, component, component_names

    if comp_required.issubset(name_set):
        component = np.array(samples, dtype=np.float32, copy=True)
        component_names = list(source_param_names)
        symmetric, symmetric_names = transform_component_params_to_reparameterized(component, source_param_names)
        return symmetric, symmetric_names, component, component_names

    # Insufficient params for full conversion — return same data for both views
    arr = np.array(samples, dtype=np.float32, copy=True)
    names = list(source_param_names)
    return arr, names, arr, names


def split_vector_into_symmetric_and_component(param_vector, source_param_names):
    """Return both symmetric and component representations for one parameter vector."""
    vector_2d = np.array(param_vector, dtype=np.float32, copy=True).reshape(1, -1)
    symmetric, symmetric_names, component, component_names = split_samples_into_symmetric_and_component(
        vector_2d,
        source_param_names
    )
    return symmetric[0], symmetric_names, component[0], component_names


def calculate_detector_time_delay(ra: float, dec: float, det1: str = 'H1', det2: str = 'L1', gps_time: float = 0.0) -> float:
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

        # Compute detector-specific delays from geocenter, then difference.
        delay1 = detector1.time_delay_from_earth_center(ra, dec, gps_time)
        delay2 = detector2.time_delay_from_earth_center(ra, dec, gps_time)

        return float(delay2 - delay1)
    except Exception as e:
        print(f"Warning: Failed to calculate time delay: {e}. Returning 0.0")
        return 0.0


def prepare_pycbc_data(num_samples=10000, generator_backend=None, psd_csv=None):
    cached_outputs = load_prepared_waveform_cache(num_samples, generator_backend=generator_backend, psd_csv=psd_csv)
    if cached_outputs is not None:
        return cached_outputs

    config = {
        'mass1': lambda size: np.random.uniform(10, 180, size=size),
        'mass2': lambda size: np.random.uniform(10, 150, size=size),
        'spin1z': lambda size: np.random.uniform(0, 0.88, size=size),
        'spin2z': lambda size: np.random.uniform(0, 0.88, size=size),
        'coa_phase': lambda size: np.random.uniform(0, 2 * np.pi, size=size),
        'distance': lambda size: np.random.uniform(100, 2000, size=size),
    }

    # LV parameters are sampled here and passed through to the FD generator.
    config['lambda_g'] = lambda size: np.random.uniform(LAMBDA_G_MIN, LAMBDA_G_MAX, size=size)
    config['A_lv'] = lambda size: np.random.uniform(A_LV_MIN, A_LV_MAX, size=size)
    
    # GPS time delay will be calculated at default sky location (north pole)
    default_ra = 0.0
    default_dec = np.pi / 2.0
    time_delay_default = calculate_detector_time_delay(default_ra, default_dec, 'H1', 'L1')

    print(f"\nCalling pycbc_data_generator with {num_samples} samples...")
    print(f"  GPS time delay (north pole): {time_delay_default*1000:.3f} ms")
    print(f"  Generator backend: {generator_backend or GENERATOR_BACKEND}")
    try:
        import time
        t0 = time.time()
        
        gen_backend = generator_backend or GENERATOR_BACKEND
        gen_psd_csv = psd_csv if psd_csv is not None else REAL_PSD_CSV
        
        # Generate with H1 and L1 projection (default detectors).
        if gen_backend == 'real_psd':
            if not gen_psd_csv:
                raise ValueError("psd_csv must be provided when generator_backend='real_psd'")
            raise ValueError("Model_LV is LV-only and currently requires GENERATOR_BACKEND='standard' (no real_psd LV generator yet)")

        result = data_generator.pycbc_lorentz_violation_data_generator(
            config,
            num_samples=num_samples,
            alpha_lv=ALPHA_LV,
            batch_size=16,
            num_workers=NUM_WORKERS,
            chunk_size=5000,
            add_noise=ADD_NOISE,
        )
        print("  LV mode: sampled lambda_g/A_lv passed to generalized Lorentz-violation generator.")
        print(f"  LV settings: alpha_lv={ALPHA_LV}, A_lv in [{A_LV_MIN:.1e}, {A_LV_MAX:.1e}] m")
        t1 = time.time()
        print(f"✓ Waveform generation took {t1-t0:.2f} seconds.")
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
        component_param_names = metadata['parameter_names']
        param_norm_info = metadata['parameter_normalization']
        print(f"✓ Got dataloaders: train={len(train_loader)} batches, val={len(val_loader)} batches, test={len(test_loader)} batches")
        print(f"✓ Parameter normalization info stored")
        print(f"✓ Component parameter names: {component_param_names}")
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
    
    # Whitened GW strain has arbitrary magnitude — feeding raw values into
    # nn.Linear(16384, 256) causes overflow (especially under float16 autocast).
    print("  Normalizing waveform data (per-sample zero mean, unit variance)...")
    data_mean = all_data.mean(dim=1, keepdim=True)
    data_std = all_data.std(dim=1, keepdim=True).clamp(min=1e-8)
    all_data = (all_data - data_mean) / data_std
    
    # Remove any samples that became NaN (e.g., from bad whitening)
    nan_mask = torch.isnan(all_data).any(dim=1) | torch.isinf(all_data).any(dim=1)
    if nan_mask.any():
        n_bad = nan_mask.sum().item()
        print(f"  ⚠ Removing {n_bad} samples with NaN/Inf values")
        all_data = all_data[~nan_mask]
        all_params = all_params[~nan_mask]
    
    print(f"  Training data: {all_data.shape}, params: {all_params.shape}")
    print(f"  Data range: [{all_data.min():.2f}, {all_data.max():.2f}], mean={all_data.mean():.4f}, std={all_data.std():.4f}")

    print("\nProcessing validation data from dataloaders...")
    print("  (TWO-DETECTOR MODE: Concatenating H1 and L1 streams)")
    data_val = []
    params_val = []
    for i, (waveforms, batch_params) in enumerate(val_loader):
        val_params_batch = torch.FloatTensor(batch_params)
        val_data_batch = torch.FloatTensor(waveforms)
        h1_data = val_data_batch[:, 0, :].reshape(val_data_batch.shape[0], -1)
        l1_data = val_data_batch[:, 1, :].reshape(val_data_batch.shape[0], -1)
        concatenated_data = torch.cat([h1_data, l1_data], dim=1)
        data_val.append(concatenated_data)
        params_val.append(val_params_batch)
    all_val_data = torch.cat(data_val, dim=0)
    all_val_params = torch.cat(params_val, dim=0)

    print("  Normalizing validation waveform data (per-sample)...")
    val_data_mean = all_val_data.mean(dim=1, keepdim=True)
    val_data_std = all_val_data.std(dim=1, keepdim=True).clamp(min=1e-8)
    all_val_data = (all_val_data - val_data_mean) / val_data_std

    nan_mask_val = torch.isnan(all_val_data).any(dim=1) | torch.isinf(all_val_data).any(dim=1)
    if nan_mask_val.any():
        n_bad = nan_mask_val.sum().item()
        print(f"  ⚠ Removing {n_bad} val samples with NaN/Inf values")
        all_val_data = all_val_data[~nan_mask_val]
        all_val_params = all_val_params[~nan_mask_val]

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
    
    # Same normalization for test data
    print("  Normalizing test waveform data (per-sample)...")
    test_data_mean = all_test_data.mean(dim=1, keepdim=True)
    test_data_std = all_test_data.std(dim=1, keepdim=True).clamp(min=1e-8)
    all_test_data = (all_test_data - test_data_mean) / test_data_std
    
    nan_mask_test = torch.isnan(all_test_data).any(dim=1) | torch.isinf(all_test_data).any(dim=1)
    if nan_mask_test.any():
        n_bad = nan_mask_test.sum().item()
        print(f"  ⚠ Removing {n_bad} test samples with NaN/Inf values")
        all_test_data = all_test_data[~nan_mask_test]
        all_test_params = all_test_params[~nan_mask_test]

    # Parameters from data generator are z-score normalized in component space.
    # Convert back to physical component space first, then optionally reparameterize.
    all_params_physical = denormalize_params(all_params.numpy(), param_norm_info, component_param_names)
    all_test_params_physical = denormalize_params(all_test_params.numpy(), param_norm_info, component_param_names)

    if USE_REPARAMETERIZED_TARGETS:
        print("  Reparameterizing targets: [mass1, mass2, spin1z, spin2z] -> [chirp_mass, q, chi_eff, chi_a]")
        train_targets_physical, all_reparam_names = transform_component_params_to_reparameterized(
            all_params_physical, component_param_names
        )
        test_targets_physical, _ = transform_component_params_to_reparameterized(
            all_test_params_physical, component_param_names
        )
        target_param_names = list(all_reparam_names)
    else:
        print("  Using original component targets: [mass1, mass2, spin1z, spin2z]")
        train_targets_physical = np.array(all_params_physical, copy=True)
        test_targets_physical = np.array(all_test_params_physical, copy=True)
        target_param_names = list(component_param_names)

    # --- Subset to MODEL_PARAMS: keep only the columns the user selected ---
    selected_indices = []
    for mp in MODEL_PARAMS:
        if mp in target_param_names:
            selected_indices.append(target_param_names.index(mp))
        else:
            raise ValueError(
                f"MODEL_PARAMS entry '{mp}' not found in available target params: {target_param_names}"
            )
    train_targets_physical = train_targets_physical[:, selected_indices]
    test_targets_physical = test_targets_physical[:, selected_indices]
    target_param_names = list(MODEL_PARAMS)
    print(f"  Selected model target columns: {target_param_names}")

    train_targets_norm, target_param_norm_info = normalize_params_with_reference(
        train_targets_physical, target_param_names, reference_norm_info=None
    )
    test_targets_norm, _ = normalize_params_with_reference(
        test_targets_physical, target_param_names, reference_norm_info=target_param_norm_info
    )

    all_params = torch.from_numpy(train_targets_norm).float()
    all_test_params = torch.from_numpy(test_targets_norm).float()

    all_val_params_physical = denormalize_params(all_val_params.numpy(), param_norm_info, component_param_names)
    if USE_REPARAMETERIZED_TARGETS:
        val_targets_physical, _ = transform_component_params_to_reparameterized(
            all_val_params_physical, component_param_names
        )
    else:
        val_targets_physical = np.array(all_val_params_physical, copy=True)
    val_targets_physical = val_targets_physical[:, selected_indices]
    val_targets_norm, _ = normalize_params_with_reference(
        val_targets_physical, target_param_names, reference_norm_info=target_param_norm_info
    )
    all_val_params = torch.from_numpy(val_targets_norm).float()

    print(f"  Test data: {all_test_data.shape}, params: {all_test_params.shape}")
    print("✓ Data preparation complete (two-detector concatenation, normalized)\n")

    prepared_outputs = (
        all_data,
        all_params,
        all_val_data,
        all_val_params,
        all_test_data,
        all_test_params,
        target_param_norm_info,
        target_param_names,
        time_delay_default,
    )


    save_prepared_waveform_cache(
        num_samples=num_samples,
        prepared_outputs=prepared_outputs,
        generator_metadata=metadata,
        component_param_names=component_param_names,
        generator_backend=generator_backend,
        psd_csv=psd_csv,
    )

    return prepared_outputs
















#Configure EVERYTHING -------------------------------------------------------------------------------------------------------------------------------------------
DENORMALIZE_PARAMETERS = True

#  symmetric : Train on reparameterized params (chirp_mass, q, chi_eff, chi_a + extras)
#  component : Train on component params (mass1, mass2, spin1z, spin2z, coa_phase, distance)
PARAM_SET = 'symmetric'  # 'symmetric' or 'component'


THEORY_MODE = 'lv'
MODIFIED_GRAVITY = True
LORENTZ_VIOLATION = True

LAMBDA_G_MIN = 1e14
LAMBDA_G_MAX = 1e16
ALPHA_LV = 3.0
A_LV_MIN = 1e-35
A_LV_MAX = 1e-34
print(f"LV mode enabled: lambda_g range [{LAMBDA_G_MIN:.1e}, {LAMBDA_G_MAX:.1e}] m")
print(f"LV mode enabled: alpha_lv={ALPHA_LV}, A_lv range [{A_LV_MIN:.1e}, {A_LV_MAX:.1e}] m")

# choose which parameters the model trains on and infers 
# MODEL_PARAMS: Select which parameters to train/infer.
#   For PARAM_SET='symmetric', choose from: chirp_mass, q, chi_eff, chi_a
#   For PARAM_SET='component', choose from: mass1, mass2, spin1z, spin2z, coa_phase, distance
#   LV extras: lambda_g, A_lv
MODEL_PARAMS = ['chirp_mass', 'q', 'chi_eff', 'chi_a', 'lambda_g', 'A_lv']

# Configure training data size
NUM_TRAINING_SAMPLES = 35000  # Adjust this to control dataset size

# Model architecture parameters
CONTEXT_DIM = 512           
NUM_FLOW_LAYERS = 5         
HIDDEN_DIM = 128
EMBEDDING_TYPE = 'simple'  # 'simple', 'conv1d', or 'lstm'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training parameters
NUM_EPOCHS = 200             
BATCH_SIZE = 64            
LEARNING_RATE = 1e-4
RETRAIN = True             # Set to True to retrain even if saved model exists

# Data generation parameters
ADD_NOISE = True           # Add realistic detector noise to waveforms
WHITEN = True              # Needed — noise is colored (aLIGOZeroDetHighPower PSD)
USE_WAVEFORM_CACHE = True  # Reuse prepared waveform datasets when config matches exactly
GENERATOR_BACKEND = 'standard'  # 'standard' or 'real_psd' for using real detector PSDs
REAL_PSD_CSV = 'test7_o4_psds_all.csv'        # True to use real PSDs, or path to PSD CSV file when generator_backend='real_psd'
NUM_WORKERS = 2            # Single place to control DataGenerator worker count

DATA_GENERATOR_BATCH_SIZE = 8
DATA_GENERATOR_DETECTORS = ['H1', 'L1']

# Set this once to switch both the "_first2.csv" and "_all.csv" inputs used below.
# 'tutorial_o4_whitened', 'test7_o4_whitened', 'test7_gating_o4_whitened', 'pycbc_o4_whitened'
REAL_DATA_CSV_PREFIX = 'test7_o4_whitened'


def get_real_data_csv_path(csv_variant):
    base_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    return os.path.join(base_dir, f'{REAL_DATA_CSV_PREFIX}_{csv_variant}.csv')


# --- Derived config from PARAM_SET and MODEL_PARAMS ---------------------------
# Validate MODEL_PARAMS against PARAM_SET
_SYMMETRIC_ALL = ['chirp_mass', 'q', 'chi_eff', 'chi_a']
_COMPONENT_ALL = ['mass1', 'mass2', 'spin1z', 'spin2z', 'coa_phase', 'distance']
_THEORY_EXTRA_PARAMS = ['lambda_g', 'A_lv']

if PARAM_SET == 'symmetric':
    USE_REPARAMETERIZED_TARGETS = True
    _VALID_PARAMS = _SYMMETRIC_ALL + _THEORY_EXTRA_PARAMS
    # EXTRA_PARAM_NAMES controls which non-core params pass through reparameterization
    EXTRA_PARAM_NAMES = [p for p in MODEL_PARAMS if p not in _SYMMETRIC_ALL]
elif PARAM_SET == 'component':
    USE_REPARAMETERIZED_TARGETS = False
    _VALID_PARAMS = _COMPONENT_ALL + _THEORY_EXTRA_PARAMS
    EXTRA_PARAM_NAMES = [p for p in MODEL_PARAMS if p not in ['mass1', 'mass2', 'spin1z', 'spin2z']]
else:
    raise ValueError(f"Unknown PARAM_SET: {PARAM_SET}. Must be 'symmetric' or 'component'.")

_invalid = [p for p in MODEL_PARAMS if p not in _VALID_PARAMS]
if _invalid:
    raise ValueError(
        f"Invalid MODEL_PARAMS for PARAM_SET='{PARAM_SET}': {_invalid}\n"
        f"Valid choices: {_VALID_PARAMS}"
    )

PARAM_DIM = len(MODEL_PARAMS)

# ====================================================================================
# PRINT CONFIGURATION (read from this script, not from PBS)
# ====================================================================================
print("\n" + "="*70)
print("MODEL ARCHITECTURE PARAMETERS")
print(f"  Parameter Set (PARAM_SET):              {PARAM_SET}")
print(f"  Model Parameters (MODEL_PARAMS):        {MODEL_PARAMS}")
print(f"  Parameter Dimension (PARAM_DIM):        {PARAM_DIM}")
print(f"  Context Dimension (CONTEXT_DIM):        {CONTEXT_DIM}")
print(f"  Number of Flow Layers (NUM_FLOW_LAYERS): {NUM_FLOW_LAYERS}")
print(f"  Hidden Dimension (HIDDEN_DIM):          {HIDDEN_DIM}")
print(f"  Embedding Type (EMBEDDING_TYPE):        {EMBEDDING_TYPE}")

print("\n" + "="*70)
print("TRAINING PARAMETERS")
print(f"  Number of Epochs (NUM_EPOCHS):          {NUM_EPOCHS}")
print(f"  Batch Size (BATCH_SIZE):                {BATCH_SIZE}")
print(f"  Learning Rate (LEARNING_RATE):          {LEARNING_RATE}")
print(f"  Training Samples (NUM_TRAINING_SAMPLES): {NUM_TRAINING_SAMPLES:,}")
print(f"  Denormalize Output (DENORMALIZE_PARAMETERS): {DENORMALIZE_PARAMETERS}")
print(f"  Reparameterized Targets:                 {USE_REPARAMETERIZED_TARGETS}")
print(f"  Extra Params (pass-through):             {EXTRA_PARAM_NAMES}")
print(f"  Add Noise (ADD_NOISE):                  {ADD_NOISE}")
print(f"  Whiten Data (WHITEN):                   {WHITEN}")
print(f"  Theory Mode (THEORY_MODE):              {THEORY_MODE}")
print(f"    lambda_g range:                       [{LAMBDA_G_MIN:.1e}, {LAMBDA_G_MAX:.1e}] m")
print(f"    alpha_lv:                             {ALPHA_LV}")
print(f"    A_lv range:                           [{A_LV_MIN:.1e}, {A_LV_MAX:.1e}] m")


estimated_batches_per_epoch = NUM_TRAINING_SAMPLES // BATCH_SIZE
total_batches = estimated_batches_per_epoch * NUM_EPOCHS
print(f"  Estimated Batches per Epoch:            {estimated_batches_per_epoch}")
print(f"  Total Batch Updates:                    {total_batches}")
print(f"  Training Parameters:                    {', '.join(MODEL_PARAMS)}")


print("="*70 + "\n")

pycbc_data, pycbc_params, pycbc_val_data, pycbc_val_params, pycbc_test_data, pycbc_test_params, param_norm_info, model_param_names, GPS_TIME_DELAY = prepare_pycbc_data(num_samples=NUM_TRAINING_SAMPLES, generator_backend=GENERATOR_BACKEND, psd_csv=REAL_PSD_CSV)

PARAM_DIM = len(model_param_names)

# Parameters are normalized in the selected model-target space
print(f"Using normalized model targets")
print(f"  Training samples: {len(pycbc_params)}")
print(f"  Test samples: {len(pycbc_test_params)}")
print(f"  Data dimension: {pycbc_data.shape[1]} (2 detectors concatenated)")
print(f"  Target parameters: {model_param_names}")
print(f"  GPS time delay: {GPS_TIME_DELAY*1000:.3f} ms (will be appended after embedding)")
print(f"  Normalization info stored for denormalization\n")     

# Model save path - includes key parameters for identification
samples_str = f"{NUM_TRAINING_SAMPLES//1000}k" if NUM_TRAINING_SAMPLES >= 1000 else str(NUM_TRAINING_SAMPLES)
noise_str = "noisy" if ADD_NOISE else "clean"
whiten_str = "_whitened" if WHITEN else ""
theory_tag = "_LV"
MODEL_SAVE_PATH = f"dingo_N{samples_str}_F{NUM_FLOW_LAYERS}_C{CONTEXT_DIM}_H{HIDDEN_DIM}_E{NUM_EPOCHS}_{EMBEDDING_TYPE}_{noise_str}{whiten_str}_{PARAM_SET}{theory_tag}.pt"
print(f"Model will be saved as: {MODEL_SAVE_PATH}")

# Train DINGO model on PyCBC data (or load if exists)
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

if os.path.exists(MODEL_SAVE_PATH) and not RETRAIN:
    try:
        checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(DEVICE)
        losses = checkpoint.get('losses', [])

    except Exception as e:
        print(f"  Failed to load model: {e}")
        os.remove(MODEL_SAVE_PATH)  # Remove corrupted file
        
        losses, val_losses = train_dingo_model_pycbc(
            model,
            pycbc_params,
            pycbc_data,
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
            val_params=pycbc_val_params,
            val_data=pycbc_val_data,
        )
        
        # Save the newly trained model
        torch.save({
            'model_state_dict': model.state_dict(),
            'losses': losses,
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
                'add_noise': ADD_NOISE,
                'whiten': WHITEN,
            }
        }, MODEL_SAVE_PATH)

else:
    print(f"NO PRE-TRAINED MODEL FOUND")
    

    losses, val_losses = train_dingo_model_pycbc(
        model,
        pycbc_params,  
        pycbc_data,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        val_params=pycbc_val_params,
        val_data=pycbc_val_data,
    )

    
    # Save the trained model
    torch.save({
        'model_state_dict': model.state_dict(),
        'losses': losses,
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
        }
    }, MODEL_SAVE_PATH)
    print(f"\n✓ Model saved to {MODEL_SAVE_PATH}")

print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Data dimension: {pycbc_data.shape[1]}, Training samples: {len(pycbc_data)}\n")


# TESTING PYCBC DATA INFERENCE ------------------------------------------------------------------------------------------------------------------------------------------------
# with 1000 samples
#finds differences between inferred mean and true value for each parameter for every sample

















print("\n" + "=" * 80)
print("TESTING PYCBC PARAMETER INFERENCE - 1000 SAMPLES")
print("=" * 80)

# Test on 1000 samples from the test set
num_test_samples = min(1000, len(pycbc_test_data))
test_indices = list(range(num_test_samples))

# Collect mean differences for all samples
param_names = list(model_param_names)
OUTPUT_COMPONENT_DISTRIBUTIONS_FROM_SYMMETRIC = USE_REPARAMETERIZED_TARGETS

_, eval_param_names = convert_samples_to_eval_space(
    np.zeros((1, len(param_names)), dtype=np.float32),
    param_names,
    output_component_distributions=OUTPUT_COMPONENT_DISTRIBUTIONS_FROM_SYMMETRIC
)

mean_errors = {param: [] for param in eval_param_names}
mean_differences = {param: [] for param in eval_param_names}
sigma_deviations = {param: [] for param in eval_param_names}  # |true - mean| / std per sample

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
    
    # Denormalize true parameters if enabled
    if DENORMALIZE_PARAMETERS:
        true_params = denormalize_params(true_params_normalized, param_norm_info, param_names)
    else:
        true_params = true_params_normalized
    
    # Generate posterior samples (DENORMALIZED if DENORMALIZE_PARAMETERS=True)
    posterior_samples, stats = infer_with_dingo(model, observed_data, num_samples=10000, 
                                               param_norm_info=param_norm_info if DENORMALIZE_PARAMETERS else None,
                                               param_names=param_names if DENORMALIZE_PARAMETERS else None)

    posterior_eval, _ = convert_samples_to_eval_space(
        posterior_samples,
        param_names,
        output_component_distributions=OUTPUT_COMPONENT_DISTRIBUTIONS_FROM_SYMMETRIC
    )
    true_eval, _ = convert_vector_to_eval_space(
        true_params,
        param_names,
        output_component_distributions=OUTPUT_COMPONENT_DISTRIBUTIONS_FROM_SYMMETRIC
    )
    
    # Calculate mean differences (in PHYSICAL space if denormalized, or NORMALIZED space if not)
    for param_idx in range(len(eval_param_names)):
        param_samples = posterior_eval[:, param_idx]
        true_val = true_eval[param_idx]
        inferred_mean = np.mean(param_samples)
        error = inferred_mean - true_val  # Signed difference
        abs_error = abs(error)
        
        mean_errors[eval_param_names[param_idx]].append(abs_error)
        mean_differences[eval_param_names[param_idx]].append(error)
        
        posterior_std = np.std(param_samples)
        z_score = abs_error / posterior_std if posterior_std > 0 else float('inf')
        sigma_deviations[eval_param_names[param_idx]].append(z_score)

print(f"\n✓ Completed inference on {num_test_samples} samples")

# Print summary statistics
if DENORMALIZE_PARAMETERS:
    print("\nParameter Inference Summary (1000 samples in PHYSICAL SPACE):")
else:
    print("\nParameter Inference Summary (1000 samples in NORMALIZED SPACE):")
    
for param_idx, param in enumerate(eval_param_names):
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

    sym_samples, sym_names, comp_samples, comp_names = split_samples_into_symmetric_and_component(
        posterior_samples,
        param_names
    )

    sample_posteriors[sample_idx] = {
        'symmetric': sym_samples,
        'component': comp_samples,
    }
    print(f"  ✓ Generated posteriors for sample {sample_idx}")


# ===== VISUALIZATION =====
# Layout: (1 + 2*N + 1) rows x 6 columns
#   Row 0: Inference error histograms
#   For each selected sample:
#       Row A: symmetric posteriors (chirp_mass, q, chi_eff, chi_a, coa_phase, distance)
#       Row B: component posteriors (mass1, mass2, spin1z, spin2z, coa_phase, distance)
#   Final row: Real GW event component posteriors

# Published median source-frame parameters for GW250114_082203
# from LIGO/Virgo/KAGRA (arXiv:2509.08054, NRSur7dq4 waveform model):
#   mass1 = 33.6 (+1.2/-0.8) M_sun
#   mass2 = 32.2 (+0.8/-1.3) M_sun
#   spin magnitudes chi_1,2 <= 0.26 (90% credibility), consistent with ~0
#   chi_eff ~ 0 (small effective inspiral spin)
# We use spin1z=0.0 and spin2z=0.0 as best estimates (both consistent with zero).
# Source-frame component params from publication
GW250114_COMPONENT_TRUE_PARAMS_SOURCE = {
    'mass1': 33.6,    # M_sun (median, NRSur7dq4)
    'mass2': 32.2,    # M_sun (median, NRSur7dq4)
    'spin1z': 0.0,    # best estimate; magnitude ≤ 0.26 at 90% CL
    'spin2z': 0.0,    # best estimate; magnitude ≤ 0.26 at 90% CL
    'coa_phase': 0.0, # unknown; placeholder
    'distance': 410.0,# Mpc; used for low-z detector-frame mass conversion if z not set
}

# If you know the event redshift, set this to a float (e.g. 0.08) to override
# distance-based estimate. Leave as None to estimate from distance.
GW250114_REDSHIFT_OVERRIDE = None

m1_det, m2_det, GW250114_EFFECTIVE_Z = convert_source_to_detector_frame_masses(
    GW250114_COMPONENT_TRUE_PARAMS_SOURCE['mass1'],
    GW250114_COMPONENT_TRUE_PARAMS_SOURCE['mass2'],
    redshift=GW250114_REDSHIFT_OVERRIDE,
    distance_mpc=GW250114_COMPONENT_TRUE_PARAMS_SOURCE.get('distance', None)
)

# Full component true params in detector frame (used for plotting against model outputs)
GW250114_COMPONENT_TRUE_PARAMS_FULL = dict(GW250114_COMPONENT_TRUE_PARAMS_SOURCE)
GW250114_COMPONENT_TRUE_PARAMS_FULL['mass1'] = m1_det
GW250114_COMPONENT_TRUE_PARAMS_FULL['mass2'] = m2_det
print(
    f"GW250114 mass-frame conversion: source->detector with z={GW250114_EFFECTIVE_Z:.4f} | "
    f"m1: {GW250114_COMPONENT_TRUE_PARAMS_SOURCE['mass1']:.3f}->{m1_det:.3f}, "
    f"m2: {GW250114_COMPONENT_TRUE_PARAMS_SOURCE['mass2']:.3f}->{m2_det:.3f}"
)
# Derive symmetric (reparameterized) true params
GW250114_SYMMETRIC_TRUE_PARAMS = transform_component_dict_to_reparameterized(GW250114_COMPONENT_TRUE_PARAMS_FULL)
# Component true params trimmed to 4 core masses/spins (for derived-component plot rows)
GW250114_COMPONENT_TRUE_PARAMS = {
    k: v for k, v in GW250114_COMPONENT_TRUE_PARAMS_FULL.items()
    if k in ('mass1', 'mass2', 'spin1z', 'spin2z')
}

# Parameter name lists for plotting — derived from MODEL_PARAMS
# SYMMETRIC_PARAM_NAMES: the model's direct output params (used for symmetric-row plots)
# COMPONENT_PARAM_NAMES: derived component params (used for component-row plots)
SYMMETRIC_PARAM_NAMES = list(MODEL_PARAMS)

# Check whether we have enough params to convert to component space
_CAN_CONVERT_TO_COMPONENT = (
    (PARAM_SET == 'symmetric' and {'chirp_mass', 'q', 'chi_eff', 'chi_a'}.issubset(set(MODEL_PARAMS)))
    or (PARAM_SET == 'component' and {'mass1', 'mass2', 'spin1z', 'spin2z'}.issubset(set(MODEL_PARAMS)))
)

if _CAN_CONVERT_TO_COMPONENT:
    # Component names: always mass1..spin2z + any extra params in MODEL_PARAMS
    _BASE_COMP = ['mass1', 'mass2', 'spin1z', 'spin2z']
    _EXTRA_IN_MODEL = [p for p in MODEL_PARAMS if p not in _SYMMETRIC_ALL and p not in _BASE_COMP]
    COMPONENT_PARAM_NAMES = _BASE_COMP + _EXTRA_IN_MODEL
else:
    # Can't convert — component view is same as symmetric view
    COMPONENT_PARAM_NAMES = list(MODEL_PARAMS)
    if PARAM_SET == 'symmetric':
        print(f"  Note: MODEL_PARAMS does not include all 4 symmetric params — component conversion disabled")


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
    if param_name == 'A_lv':
        return r'$A_{\mathrm{LV}}$ (m)'
    return f'{param_name}'


def format_legend(param_name, value):
    if not np.isfinite(value):
        return 'nan'
    if param_name in ('lambda_g', 'A_lv'):
        return f'{value:.3g}'
    if param_name in ('q', 'chi_eff', 'chi_a', 'spin1z', 'spin2z'):
        return f'{value:.3f}'
    if param_name == 'coa_phase':
        return f'{value:.3f}'
    return f'{value:.2f}'


def plot_posterior_row(axes_row, samples, true_vals, param_names_list, color, row_title_prefix):
    """Plot a single row of posterior histograms (reused across all figures)."""
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


import matplotlib.gridspec as gridspec
import os
from datetime import datetime

NUM_PLOT_COLS = max(len(SYMMETRIC_PARAM_NAMES), len(COMPONENT_PARAM_NAMES))
if PARAM_SET == 'component' or not _CAN_CONVERT_TO_COMPONENT:
    # Only 1 row per sample (no separate symmetric+component rows)
    num_plot_rows = 1 + len(sample_indices) + 1
else:
    # Symmetric mode: error row + 2 rows (symmetric + component) per sample + real GW row
    num_plot_rows = 1 + (2 * len(sample_indices)) + 1
fig = plt.figure(figsize=(NUM_PLOT_COLS * 5, max(28, 3.8 * num_plot_rows)))
gs = gridspec.GridSpec(num_plot_rows, NUM_PLOT_COLS, figure=fig, hspace=0.45, wspace=0.35)

# Create axes array for all rows
axes = [[fig.add_subplot(gs[row, col]) for col in range(NUM_PLOT_COLS)] for row in range(num_plot_rows)]
axes = np.array(axes)

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
    f"  Epochs: {NUM_EPOCHS}\n"
    f"  Two-Detector Mode: H1+L1 concatenated"
)

# Place model info as a figure-level text annotation (top-center, outside any plot)
fig.text(0.50, 0.99, model_info_text, transform=fig.transFigure,
         fontsize=8, verticalalignment='top', horizontalalignment='center',
         fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Row 0: Summary statistics from all 1000 samples
for param_idx, param in enumerate(eval_param_names):
    ax = axes[0, param_idx]
    diffs = mean_differences[param]
    errors = mean_errors[param]
    
    # Check if we have valid data
    diffs_clean = np.array([d for d in diffs if not np.isnan(d)])
    errors_clean = np.array([e for e in errors if not np.isnan(e)])
    
    if len(diffs_clean) > 0:
        # Plot histogram of differences
        ax.hist(diffs_clean, bins=50, alpha=0.7, color='steelblue', edgecolor='black', density=False)
        
        # Mark zero line (perfect inference)
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Perfect Inference (0)')
        
        # Labels and title
        ax.set_xlabel(f'Mean Difference (Inferred - True)', fontsize=9)
        ax.set_ylabel('Frequency', fontsize=9)
        param_label = format_param_label(param)
        ax.set_title(f'{param_label} Inference Errors\n({num_test_samples} test samples)', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
    else:
        # No valid data - show warning
        ax.text(0.5, 0.5, f'No valid data\n(all NaN)', ha='center', va='center', 
                transform=ax.transAxes, fontsize=10, color='red', fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

# Rows 1..: symmetric mode with conversion → 2 rows per sample; otherwise → 1 row per sample
for row_idx, sample_idx in enumerate(sample_indices):
    posterior_sym = sample_posteriors[sample_idx]['symmetric']
    posterior_comp = sample_posteriors[sample_idx]['component']
    true_params_normalized = pycbc_test_params[sample_idx].numpy()
    
    if DENORMALIZE_PARAMETERS:
        true_params = denormalize_params(true_params_normalized, param_norm_info, param_names)
    else:
        true_params = true_params_normalized

    true_sym, _, true_comp, _ = split_vector_into_symmetric_and_component(true_params, param_names)

    if PARAM_SET == 'symmetric' and _CAN_CONVERT_TO_COMPONENT:
        # Two-row layout: symmetric row + component row
        row_sym = 1 + (2 * row_idx)
        row_comp = row_sym + 1
        plot_posterior_row(axes[row_sym], posterior_sym, true_sym,
                           SYMMETRIC_PARAM_NAMES, 'darkgreen',
                           f'Symmetric Posterior (Sample {sample_idx})')
    else:
        row_comp = 1 + row_idx

    # Component / direct posterior row
    if _CAN_CONVERT_TO_COMPONENT:
        comp_origin = 'Direct' if PARAM_SET == 'component' else 'Derived'
        plot_posterior_row(axes[row_comp], posterior_comp, true_comp,
                           COMPONENT_PARAM_NAMES, 'darkgreen',
                           f'{comp_origin} Component (Sample {sample_idx})')
    else:
        plot_posterior_row(axes[row_comp], posterior_sym, true_sym,
                           SYMMETRIC_PARAM_NAMES, 'darkgreen',
                           f'Model Output (Sample {sample_idx})')

# Row 6: Test against real GW event (second waveform from the selected real-data CSV)
print("\n" + "=" * 80)
try:
    import pandas as pd
    
    csv_path = get_real_data_csv_path('first2')
    print(f"Testing against real GW event from {os.path.basename(csv_path)} (second event)...")
    gw_csv = pd.read_csv(csv_path)
    
    # Filter for the second event (event_rank == 2)
    event_2_data = gw_csv[gw_csv['event_rank'] == 2].copy()
    if len(event_2_data) == 0:
        print("  ⚠ Second event not found in CSV, skipping real GW event row")
        real_gw_row = None
    else:
        event_name = event_2_data['event_name'].iloc[0]
        event_gps = event_2_data['gps'].iloc[0]
        
        # Separate by detector
        h1_data = event_2_data[event_2_data['detector'] == 'H1'].sort_values('t_seconds')
        l1_data = event_2_data[event_2_data['detector'] == 'L1'].sort_values('t_seconds')
        
        if len(h1_data) == 0 or len(l1_data) == 0:
            print(f"  ⚠ Missing detector data for event {event_name}, skipping real GW event row")
            real_gw_row = None
        else:
            # Extract strain data
            h1_strain_full = h1_data['whitened_strain'].values.astype(np.float32)
            l1_strain_full = l1_data['whitened_strain'].values.astype(np.float32)
            h1_times = h1_data['t_seconds'].values
            l1_times = l1_data['t_seconds'].values
            
            # Get expected length per detector from training data
            expected_length = pycbc_data.shape[1] // 2  # Total input / 2 detectors
            print(f"  Model expects {expected_length} samples per detector ({expected_length/4096:.2f}s at 4096 Hz)")
            print(f"  Real GW data has {len(h1_strain_full)} H1 samples, {len(l1_strain_full)} L1 samples")
            
            # Crop around merger (t=0): take last 'expected_length' samples ending near t=0
            # Find index closest to t=0 and take samples ending there
            h1_t0_idx = np.argmin(np.abs(h1_times))  # Index closest to t=0
            l1_t0_idx = np.argmin(np.abs(l1_times))
            
            # Take 'expected_length' samples ending shortly after t=0 to capture the merger
            # Add small buffer (e.g., 0.1s = 410 samples) after merger
            buffer_samples = int(0.1 * 4096)  # 0.1 seconds buffer after merger
            h1_end_idx = min(h1_t0_idx + buffer_samples, len(h1_strain_full))
            l1_end_idx = min(l1_t0_idx + buffer_samples, len(l1_strain_full))
            h1_start_idx = max(0, h1_end_idx - expected_length)
            l1_start_idx = max(0, l1_end_idx - expected_length)
            
            h1_strain = h1_strain_full[h1_start_idx:h1_start_idx + expected_length]
            l1_strain = l1_strain_full[l1_start_idx:l1_start_idx + expected_length]
            
            # Pad if we don't have enough samples
            if len(h1_strain) < expected_length:
                h1_strain = np.pad(h1_strain, (expected_length - len(h1_strain), 0), mode='constant')
            if len(l1_strain) < expected_length:
                l1_strain = np.pad(l1_strain, (expected_length - len(l1_strain), 0), mode='constant')
            
            print(f"  Cropped to {len(h1_strain)} samples per detector")

            






            
            # Normalize to match training data preprocessing
            # Training normalizes the full concatenated vector (per-sample zero mean, unit std)
            real_gw_concat = np.concatenate([h1_strain, l1_strain])
            concat_mean = np.mean(real_gw_concat)
            concat_std = np.std(real_gw_concat) + 1e-8
            real_gw_concat = (real_gw_concat - concat_mean) / concat_std
            
            real_gw_input = real_gw_concat.reshape(1, -1)
            print(f"  Input shape: {real_gw_input.shape} (expected: (1, {pycbc_data.shape[1]}))")
            real_gw_input = torch.FloatTensor(real_gw_input).to(DEVICE)
            
            # Run inference — denormalize posteriors to physical space for comparison with published values
            print(f"  Running inference on {event_name} (GPS {event_gps:.1f})...")
            real_gw_posterior, _ = infer_with_dingo(model, real_gw_input[0].cpu().numpy(), 
                                                     num_samples=10000,
                                                     param_norm_info=param_norm_info if DENORMALIZE_PARAMETERS else None,
                                                     param_names=param_names if DENORMALIZE_PARAMETERS else None)

            real_gw_symmetric, _, real_gw_component, _ = split_samples_into_symmetric_and_component(real_gw_posterior, param_names)
            
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
    real_gw_row = None

# Plot real GW event posteriors in final row
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

# Create a run-specific output directory inside Plots.
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_plots_dir = os.path.join("Plots", f"run_{timestamp}")
os.makedirs(run_plots_dir, exist_ok=True)
print(f"Saving plots to: {run_plots_dir}")









# Save a plain-text summary of all key parameters used in this run.
param_summary_path = os.path.join(run_plots_dir, f"Parameter_Labels_{timestamp}.txt")
try:
    parameter_lines = [
        "DINGO LV RUN PARAMETER SUMMARY",
        "=" * 80,
        f"Timestamp: {timestamp}",
        f"Run plots directory: {run_plots_dir}",
        "",
        "[Training Parameters]",
        f"NUM_TRAINING_SAMPLES: {NUM_TRAINING_SAMPLES}",
        f"NUM_EPOCHS: {NUM_EPOCHS}",
        f"BATCH_SIZE: {BATCH_SIZE}",
        f"LEARNING_RATE: {LEARNING_RATE}",
        f"RETRAIN: {RETRAIN}",
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
        f"THEORY_MODE: {THEORY_MODE}",
        f"MODIFIED_GRAVITY: {MODIFIED_GRAVITY}",
        f"LORENTZ_VIOLATION: {LORENTZ_VIOLATION}",
        "Waveform model type: Lorentz-violating FD waveform generation",
        f"LAMBDA_G_MIN: {LAMBDA_G_MIN}",
        f"LAMBDA_G_MAX: {LAMBDA_G_MAX}",
        f"ALPHA_LV: {ALPHA_LV}",
        f"A_LV_MIN: {A_LV_MIN}",
        f"A_LV_MAX: {A_LV_MAX}",
        f"GENERATOR_BACKEND: {GENERATOR_BACKEND}",
        f"REAL_PSD_CSV: {REAL_PSD_CSV}",
        f"ADD_NOISE: {ADD_NOISE}",
        f"WHITEN: {WHITEN}",
        f"Noise generation mode: {'Detector noise added in LV generator' if ADD_NOISE else 'No detector noise (clean waveforms)'}",
        f"Whitening mode: {'Whitening enabled' if WHITEN else 'No whitening'}",
        f"USE_WAVEFORM_CACHE: {USE_WAVEFORM_CACHE}",
        f"NUM_WORKERS: {NUM_WORKERS}",
        f"DATA_GENERATOR_BATCH_SIZE: {DATA_GENERATOR_BATCH_SIZE}",
        f"DATA_GENERATOR_DETECTORS: {DATA_GENERATOR_DETECTORS}",
        f"GPS_TIME_DELAY (s): {GPS_TIME_DELAY}",
        "",
        "[Config Sampling Bounds (prepare_pycbc_data)]",
        "mass1: Uniform[10, 180] Msun",
        "mass2: Uniform[10, 150] Msun",
        "spin1z: Uniform[0, 0.88]",
        "spin2z: Uniform[0, 0.88]",
        "coa_phase: Uniform[0, 2*pi] rad",
        "distance: Uniform[100, 2000] Mpc",
        f"lambda_g: Uniform[{LAMBDA_G_MIN:.3e}, {LAMBDA_G_MAX:.3e}] m",
        f"A_lv: Uniform[{A_LV_MIN:.3e}, {A_LV_MAX:.3e}] m",
        f"alpha_lv: Fixed at {ALPHA_LV}",
    ]

    with open(param_summary_path, 'w') as f:
        f.write("\n".join(parameter_lines) + "\n")
    print(f"Saved parameter summary text file: {param_summary_path}")
except Exception as e:
    print(f"⚠ Failed to save parameter summary text file: {e}")

plot_filename = os.path.join(run_plots_dir, f"PyCBC_Parameter_Inference_TwoDetector_{timestamp}.png")

try:
    fig.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {plot_filename}")
except Exception as e:
    print(f"\n✗ Failed to save plot: {e}")
    import traceback
    traceback.print_exc()
print("=" * 80)
plt.close(fig)













# ===== SECOND IMAGE: Best, Average, and Real GW Event =====
# Symmetric mode → 6 rows (sym + comp for each of best / average / real GW)
# Component mode → 3 rows (comp only for each of best / average / real GW)

std_per_param = {}
for param in eval_param_names:
    std_per_param[param] = np.std(mean_differences[param])
    print(f"  std(differences) for {param}: {std_per_param[param]:.6f}")

num_eval_params = len(eval_param_names)
scores = np.zeros(num_test_samples)
for i in range(num_test_samples):
    for param in eval_param_names:
        s = std_per_param[param]
        if s > 0:
            scores[i] += abs(mean_differences[param][i]) / s
        else:
            scores[i] += 0.0

best_sample_idx = int(np.argmin(scores))
# Average sample: score closest to num_eval_params (= one std dev per param)
avg_target = float(num_eval_params)
avg_sample_idx = int(np.argmin(np.abs(scores - avg_target)))

print(f"  Best sample  index: {best_sample_idx}  (score={scores[best_sample_idx]:.4f})")
print(f"  Average sample index: {avg_sample_idx}  (score={scores[avg_sample_idx]:.4f}, target={avg_target:.1f})")

# --- Generate posteriors for best and average samples ---
special_posteriors = {}
for label, sidx in [('best', best_sample_idx), ('average', avg_sample_idx)]:
    obs = pycbc_test_data[sidx].numpy()
    post, _ = infer_with_dingo(model, obs, num_samples=10000,
                               param_norm_info=param_norm_info if DENORMALIZE_PARAMETERS else None,
                               param_names=param_names if DENORMALIZE_PARAMETERS else None)
    sym_s, _, comp_s, _ = split_samples_into_symmetric_and_component(post, param_names)

    true_norm = pycbc_test_params[sidx].numpy()
    if DENORMALIZE_PARAMETERS:
        true_p = denormalize_params(true_norm, param_norm_info, param_names)
    else:
        true_p = true_norm
    true_sym, _, true_comp, _ = split_vector_into_symmetric_and_component(true_p, param_names)

    special_posteriors[label] = {
        'sym_samples': sym_s,
        'comp_samples': comp_s,
        'true_sym': true_sym,
        'true_comp': true_comp,
        'index': sidx,
    }
    print(f"  ✓ Posteriors generated for {label} sample (index {sidx})")

# --- Pastel colours ---
PASTEL_GREEN  = '#77dd77'   # best sample
PASTEL_RED    = '#ff6961'   # average sample
PASTEL_PURPLE = '#b39ddb'   # real GW event

# --- Build figure (row count depends on PARAM_SET and _CAN_CONVERT_TO_COMPONENT) ---
if PARAM_SET == 'component' or not _CAN_CONVERT_TO_COMPONENT:
    fig2_num_rows = 3   # one row each: best, average, real GW
else:
    fig2_num_rows = 6   # symmetric + component pairs for each

fig2 = plt.figure(figsize=(NUM_PLOT_COLS * 5, 3.8 * fig2_num_rows))
gs2 = gridspec.GridSpec(fig2_num_rows, NUM_PLOT_COLS, figure=fig2, hspace=0.55, wspace=0.35)
axes2 = [[fig2.add_subplot(gs2[r, c]) for c in range(NUM_PLOT_COLS)] for r in range(fig2_num_rows)]
axes2 = np.array(axes2)

best = special_posteriors['best']
avg  = special_posteriors['average']

if PARAM_SET == 'component' or not _CAN_CONVERT_TO_COMPONENT:
    # --- 3 rows: one view per sample ---
    _fig2_names = COMPONENT_PARAM_NAMES if PARAM_SET == 'component' else SYMMETRIC_PARAM_NAMES
    _fig2_suffix = 'Direct Component' if PARAM_SET == 'component' else 'Symmetric'
    _fig2_best_samples = best['comp_samples'] if PARAM_SET == 'component' else best['sym_samples']
    _fig2_best_true    = best['true_comp']    if PARAM_SET == 'component' else best['true_sym']
    _fig2_avg_samples  = avg['comp_samples']  if PARAM_SET == 'component' else avg['sym_samples']
    _fig2_avg_true     = avg['true_comp']     if PARAM_SET == 'component' else avg['true_sym']
    _fig2_gw_true_dict = GW250114_COMPONENT_TRUE_PARAMS if PARAM_SET == 'component' else GW250114_SYMMETRIC_TRUE_PARAMS

    plot_posterior_row(axes2[0], _fig2_best_samples, _fig2_best_true,
                       _fig2_names, PASTEL_GREEN,
                       f'Best Sample (idx {best["index"]}) – {_fig2_suffix}')
    plot_posterior_row(axes2[1], _fig2_avg_samples, _fig2_avg_true,
                       _fig2_names, PASTEL_RED,
                       f'Average Sample (idx {avg["index"]}) – {_fig2_suffix}')
    if real_gw_row is not None:
        _gw_true = np.array([_fig2_gw_true_dict.get(p, np.nan) for p in _fig2_names], dtype=np.float32)
        _fig2_gw_key = 'posterior_samples' if PARAM_SET == 'component' else 'symmetric_samples'
        plot_posterior_row(axes2[2], real_gw_row[_fig2_gw_key], _gw_true,
                           _fig2_names, PASTEL_PURPLE,
                           f'{real_gw_row["event_name"]} – {_fig2_suffix}')
    else:
        fill_empty_row(axes2[2], NUM_PLOT_COLS)
else:
    # --- Symmetric mode: 6 rows (symmetric + component for each) ---
    plot_posterior_row(axes2[0], best['sym_samples'], best['true_sym'],
                       SYMMETRIC_PARAM_NAMES, PASTEL_GREEN,
                       f'Best Sample (idx {best["index"]}) – Symmetric')
    plot_posterior_row(axes2[1], best['comp_samples'], best['true_comp'],
                       COMPONENT_PARAM_NAMES, PASTEL_GREEN,
                       f'Best Sample (idx {best["index"]}) – Derived Component')

    plot_posterior_row(axes2[2], avg['sym_samples'], avg['true_sym'],
                       SYMMETRIC_PARAM_NAMES, PASTEL_RED,
                       f'Average Sample (idx {avg["index"]}) – Symmetric')
    plot_posterior_row(axes2[3], avg['comp_samples'], avg['true_comp'],
                       COMPONENT_PARAM_NAMES, PASTEL_RED,
                       f'Average Sample (idx {avg["index"]}) – Derived Component')

    if real_gw_row is not None:
        gw_sym_true = np.array([GW250114_SYMMETRIC_TRUE_PARAMS.get(p, np.nan) for p in SYMMETRIC_PARAM_NAMES], dtype=np.float32)
        gw_comp_true = np.array([GW250114_COMPONENT_TRUE_PARAMS.get(p, np.nan) for p in COMPONENT_PARAM_NAMES], dtype=np.float32)
        plot_posterior_row(axes2[4], real_gw_row['symmetric_samples'], gw_sym_true,
                           SYMMETRIC_PARAM_NAMES, PASTEL_PURPLE,
                           f'{real_gw_row["event_name"]} – Symmetric')
        plot_posterior_row(axes2[5], real_gw_row['posterior_samples'], gw_comp_true,
                           COMPONENT_PARAM_NAMES, PASTEL_PURPLE,
                           f'{real_gw_row["event_name"]} – Derived Component')
    else:
        fill_empty_row(axes2[4], NUM_PLOT_COLS)
        fill_empty_row(axes2[5], NUM_PLOT_COLS)

plot_filename2 = os.path.join(run_plots_dir, f"PyCBC_BestAvgReal_{timestamp}.png")
try:
    fig2.savefig(plot_filename2, dpi=150, bbox_inches='tight')
    print(f"\n✓ Second plot saved to: {plot_filename2}")
except Exception as e:
    print(f"\n✗ Failed to save second plot: {e}")
    import traceback
    traceback.print_exc()
plt.close(fig2)












# ===== THIRD IMAGE: All real whitened events summary =====
print("\n" + "=" * 80)
print("GENERATING THIRD IMAGE: All real whitened events summary")
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
    print(f"Model expects {expected_length} samples per detector ({expected_length/4096:.2f}s at 4096 Hz)")

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

        real_input = torch.FloatTensor(real_concat.reshape(1, -1)).to(DEVICE)

        try:
            posterior, _ = infer_with_dingo(
                model,
                real_input[0].cpu().numpy(),
                num_samples=all_real_num_samples,
                param_norm_info=param_norm_info if DENORMALIZE_PARAMETERS else None,
                param_names=param_names if DENORMALIZE_PARAMETERS else None,
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
        print("  ⚠ No valid all-event inferences available; skipping all-real summary plot")
    else:
        num_params_all = len(all_real_param_names)

        # Load per-event catalog values (if available) for overlay markers.
        stats_candidates = [
            os.path.join(os.path.dirname(__file__), 'gw_events_stats.csv'),
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

                    m1_det = np.nan
                    m2_det = np.nan
                    if np.isfinite(m1_src) and np.isfinite(m2_src):
                        try:
                            z_in = float(z_val) if np.isfinite(z_val) else None
                            d_in = float(d_l_mpc) if np.isfinite(d_l_mpc) else 410.0
                            m1_det, m2_det, _ = convert_source_to_detector_frame_masses(
                                m1_src, m2_src, redshift=z_in, distance_mpc=d_in
                            )
                        except Exception:
                            m1_det = np.nan
                            m2_det = np.nan

                    # Prefer explicit spin columns if present; otherwise unavailable.
                    s1_val = np.nan
                    s2_val = np.nan
                    for c in ('spin1z', 'spin_1z', 'chi1z'):
                        if c in row and np.isfinite(row[c]):
                            s1_val = float(row[c])
                            break
                    for c in ('spin2z', 'spin_2z', 'chi2z'):
                        if c in row and np.isfinite(row[c]):
                            s2_val = float(row[c])
                            break

                    actual_by_event[event_name_key] = {
                        'mass1': m1_det,
                        'mass2': m2_det,
                        'spin1z': s1_val,
                        'spin2z': s2_val,
                    }
            except Exception as stats_err:
                print(f"  ⚠ Could not load gw_events_stats.csv for actual-value overlay: {stats_err}")
        else:
            print("  ⚠ gw_events_stats.csv not found; actual-value overlay disabled")

        # One parameter per row, and make the figure much wider for readability.
        n_cols_all = 1
        n_rows_all = num_params_all
        fig_width = max(24.0, 0.42 * len(all_real_results) + 12.0)
        fig_height = max(3.5 * n_rows_all, 7.0)
        fig_all, axes_all = plt.subplots(
            n_rows_all,
            n_cols_all,
            figsize=(fig_width, fig_height),
            squeeze=False,
            sharex=True,
        )
        axes_flat_all = axes_all.flatten()

        event_labels = [f"{r['event_rank']}: {r['event_name']}" for r in all_real_results]
        x = np.arange(len(all_real_results))

        for p_idx, pname in enumerate(all_real_param_names):
            ax = axes_flat_all[p_idx]
            med = []
            q05 = []
            q95 = []

            for r in all_real_results:
                arr = r[all_real_key][:, p_idx]
                med.append(np.median(arr))
                q05.append(np.percentile(arr, 5))
                q95.append(np.percentile(arr, 95))

            med = np.array(med)
            q05 = np.array(q05)
            q95 = np.array(q95)
            yerr = np.vstack([med - q05, q95 - med])

            ax.errorbar(
                x,
                med,
                yerr=yerr,
                fmt='o',
                capsize=3,
                markersize=4,
                linewidth=1.2,
                color='tab:blue',
                ecolor='tab:blue',
            )

            # Overlay catalog/actual values where available.
            actual_vals = np.array([
                actual_by_event.get(r['event_name'], {}).get(pname, np.nan)
                for r in all_real_results
            ], dtype=np.float64)
            valid_actual = np.isfinite(actual_vals)
            if np.any(valid_actual):
                ax.scatter(
                    x[valid_actual],
                    actual_vals[valid_actual],
                    marker='D',
                    s=26,
                    color='tab:orange',
                    edgecolors='black',
                    linewidths=0.35,
                    label='Actual value',
                    zorder=3,
                )
            elif pname in ('spin1z', 'spin2z'):
                ax.text(
                    0.01,
                    0.93,
                    'Actual spin values unavailable in gw_events_stats.csv',
                    transform=ax.transAxes,
                    fontsize=8,
                    color='tab:orange',
                    ha='left',
                    va='top',
                )

            ax.set_title(format_param_label(pname), fontsize=10, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(event_labels, rotation=65, ha='right', fontsize=8)
            ax.tick_params(axis='y', labelsize=8)
            ax.grid(True, alpha=0.25)
            ax.legend(loc='best', fontsize=8)

        for k in range(num_params_all, len(axes_flat_all)):
            axes_flat_all[k].set_visible(False)

        fig_all.suptitle(
            f'All Real Whitened Events: Posterior Median +/- 90% CI + Actual Values ({all_real_title_suffix} params)',
            fontsize=13,
            fontweight='bold',
        )
        fig_all.tight_layout(rect=[0, 0.03, 1, 0.965])
        fig_all.savefig(all_real_plot_filename, dpi=150, bbox_inches='tight')
        plt.close(fig_all)
        print(f"  ✓ Third plot saved to: {all_real_plot_filename}")

except Exception as e:
    print(f"  ⚠ Failed to generate all-real summary plot: {e}")
    import traceback
    traceback.print_exc()













# ===== FOURTH IMAGE: Corner plot (2D posteriors, DINGO-paper style) =====
# Diagonal: 1D marginal KDE curves
# Lower triangle: 2D contour plots at 50 % and 90 % credible regions
# True values marked as dashed crosshairs

print("\n" + "=" * 80)
print("GENERATING FOURTH IMAGE: Corner plot (2D posteriors)")
print("=" * 80)

def _contour_levels(z_grid, credible_fracs=(0.5, 0.9)):
    """Return density thresholds that enclose *credible_fracs* of the probability."""
    z_sorted = np.sort(z_grid.ravel())[::-1]
    cumsum = np.cumsum(z_sorted)
    cumsum /= cumsum[-1]
    levels = []
    for frac in sorted(credible_fracs):
        idx = np.searchsorted(cumsum, frac)
        idx = min(idx, len(z_sorted) - 1)
        levels.append(z_sorted[idx])
    return sorted(levels)  # ascending

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
        {label: array-like of true values} – one per entry in samples_dict,
        keyed by the same label.  Pass None to omit truth markers.
    """
    n = len(corner_param_names)
    fig, axes = plt.subplots(n, n, figsize=(3.2 * n, 3.2 * n))
    if n == 1:
        axes = np.array([[axes]])

    # Hide upper triangle
    for i in range(n):
        for j in range(i + 1, n):
            axes[i, j].set_visible(False)

    for label, (samples, color) in samples_dict.items():
        for i in range(n):
            # --- Diagonal: 1D marginal ---
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

            # --- Lower triangle: 2D contours ---
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
                    # fallback: 2D histogram
                    try:
                        ax2.hist2d(xdata, ydata, bins=40, cmap='Greens', alpha=0.6)
                    except Exception:
                        pass

    # --- Truth crosshairs ---
    if true_values_dict is not None:
        for label, tv_arr in true_values_dict.items():
            for i in range(n):
                axes[i, i].axvline(tv_arr[i], color='black', ls='--', lw=1.0, alpha=0.7)
                for j in range(i):
                    axes[i, j].axvline(tv_arr[j], color='black', ls='--', lw=0.8, alpha=0.5)
                    axes[i, j].axhline(tv_arr[i], color='black', ls='--', lw=0.8, alpha=0.5)

    # --- Axis labels ---
    for i in range(n):
        axes[n - 1, i].set_xlabel(format_param_label(corner_param_names[i]), fontsize=11)
        if i > 0:
            axes[i, 0].set_ylabel(format_param_label(corner_param_names[i]), fontsize=11)
        # Remove tick labels on interior cells
        if i < n - 1:
            axes[i, i].set_xticklabels([])
        for j in range(i):
            if j > 0:
                axes[i, j].set_yticklabels([])
            if i < n - 1:
                axes[i, j].set_xticklabels([])
        axes[i, i].set_yticks([])   # no y-ticks on 1D diagonal
        axes[i, i].tick_params(labelsize=8)
        for j in range(i):
            axes[i, j].tick_params(labelsize=8)

    # --- Legend: collect handles from the first diagonal cell ---
    handles, labels_leg = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels_leg, loc='upper right', fontsize=10,
                   frameon=True, framealpha=0.9)

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

# --- Determine which parameters and samples go into the corner plot ---
# Use only the best sample for the corner plot
if PARAM_SET == 'component' and _CAN_CONVERT_TO_COMPONENT:
    # Corner plot of component params (+ lambda_g if MG is on)
    CORNER_PARAM_NAMES = list(COMPONENT_PARAM_NAMES)
    corner_samples_dict = {
        f'Best (idx {best["index"]})': (best['comp_samples'], PASTEL_GREEN),
    }
    corner_true_dict = {
        f'Best (idx {best["index"]})': best['true_comp'],
    }
else:
    # Corner plot of the symmetric/model parameters (+ lambda_g if MG is on)
    CORNER_PARAM_NAMES = list(SYMMETRIC_PARAM_NAMES)
    corner_samples_dict = {
        f'Best (idx {best["index"]})': (best['sym_samples'], PASTEL_GREEN),
    }
    corner_true_dict = {
        f'Best (idx {best["index"]})': best['true_sym'],
    }

corner_origin = 'direct' if PARAM_SET == 'component' else PARAM_SET
plot_filename3 = os.path.join(run_plots_dir, f"PyCBC_CornerPlot_{PARAM_SET}_{timestamp}.png")
make_corner_plot(
    corner_samples_dict,
    CORNER_PARAM_NAMES,
    true_values_dict=corner_true_dict,
    title=f'Best Sample – 2D Posterior Corner Plot ({corner_origin} parameters, {NUM_EPOCHS} epochs)',
    filename=plot_filename3
)












# NEXT PLOT: Coverage plot (distribution of sigma deviations)
# Subplots: (1) GR simulated, (2) Modified-GR simulated, (3) GR real, (4) Modified-GR real
from scipy.stats import norm as scipy_norm

sigma_thresholds = np.linspace(0, 5, 500)
gaussian_coverage = np.array([2 * scipy_norm.cdf(s) - 1 for s in sigma_thresholds])

# Split eval params into GR and modified-theory groups
modified_theory_params = [p for p in eval_param_names if p in ('lambda_g', 'A_lv')]
gr_params = [p for p in eval_param_names if p not in modified_theory_params]

# --- Simulated data coverage (from existing sigma_deviations) ---
sim_coverage_gr = {}
for param in gr_params:
    z_scores = np.array(sigma_deviations[param])
    sim_coverage_gr[param] = np.array([np.mean(z_scores <= s) for s in sigma_thresholds])

sim_coverage_mg = {}
for param in modified_theory_params:
    z_scores = np.array(sigma_deviations[param])
    sim_coverage_mg[param] = np.array([np.mean(z_scores <= s) for s in sigma_thresholds])

# --- Real data coverage (compute z-scores from all_real_results + catalog) ---
real_sigma_deviations = {}  # param -> list of z-scores across real events
real_coverage_gr = {}
real_n_events = 0
try:
    # Build catalog true values in BOTH component and symmetric spaces
    real_catalog_sym = {}  # event_name -> {chirp_mass, q, chi_eff}
    if 'actual_by_event' in dir() or 'actual_by_event' in locals():
        # actual_by_event has component params; also derive symmetric ones from catalog CSV
        stats_candidates_5 = [
            os.path.join(os.path.dirname(__file__), 'gw_events_stats.csv'),
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
                mc_src = row.get('chirp_mass_source', np.nan)

                if np.isfinite(m1_src) and np.isfinite(m2_src):
                    z_in = float(z_val) if np.isfinite(z_val) else None
                    d_in = float(d_l) if np.isfinite(d_l) else 410.0
                    try:
                        m1_det, m2_det, z_used = convert_source_to_detector_frame_masses(
                            m1_src, m2_src, redshift=z_in, distance_mpc=d_in
                        )
                        q_cat = min(m1_det, m2_det) / max(m1_det, m2_det) if max(m1_det, m2_det) > 0 else np.nan
                        mc_det = (m1_det * m2_det) ** 0.6 / (m1_det + m2_det) ** 0.2 if (m1_det + m2_det) > 0 else np.nan
                        real_catalog_sym[ev_name] = {
                            'chirp_mass': mc_det,
                            'q': q_cat,
                            'chi_eff': float(chi_eff_cat) if np.isfinite(chi_eff_cat) else np.nan,
                            'mass1': m1_det,
                            'mass2': m2_det,
                        }
                    except Exception:
                        pass

    # Compute z-scores for each real event and each GR parameter
    if 'all_real_results' in dir() or 'all_real_results' in locals():
        for res in all_real_results:
            ev = res['event_name']
            if ev not in real_catalog_sym:
                continue
            cat = real_catalog_sym[ev]
            # Use symmetric posteriors for symmetric params, component for component params
            sym_post = res.get('symmetric_samples')
            comp_post = res.get('posterior_samples')
            for param in gr_params:
                true_val = cat.get(param, np.nan)
                if not np.isfinite(true_val):
                    continue
                # Pick the right posterior array and find param column
                if param in ['chirp_mass', 'q', 'chi_eff', 'chi_a']:
                    # These are in symmetric posterior columns
                    sym_names = ['chirp_mass', 'q', 'chi_eff', 'chi_a'] + [p for p in EXTRA_PARAM_NAMES if p not in ('lambda_g', 'A_lv')]
                    if param in sym_names and sym_post is not None and sym_post.shape[1] > sym_names.index(param):
                        col_idx = sym_names.index(param)
                        ps = sym_post[:, col_idx]
                    else:
                        continue
                elif param in ['mass1', 'mass2', 'spin1z', 'spin2z']:
                    comp_names = ['mass1', 'mass2', 'spin1z', 'spin2z'] + [p for p in EXTRA_PARAM_NAMES if p not in ('lambda_g', 'A_lv')]
                    if param in comp_names and comp_post is not None and comp_post.shape[1] > comp_names.index(param):
                        col_idx = comp_names.index(param)
                        ps = comp_post[:, col_idx]
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

        # Build coverage curves for real GR params
        for param in gr_params:
            if param in real_sigma_deviations and len(real_sigma_deviations[param]) > 0:
                z_arr = np.array(real_sigma_deviations[param])
                real_coverage_gr[param] = np.array([np.mean(z_arr <= s) for s in sigma_thresholds])
        real_n_events = max((len(v) for v in real_sigma_deviations.values()), default=0)

except Exception as real_cov_err:
    print(f"  ⚠ Could not compute real-event sigma coverage: {real_cov_err}")
    import traceback
    traceback.print_exc()

# --- Build the 2×2 figure ---
fig5, axes5 = plt.subplots(2, 2, figsize=(16, 12))

def _plot_coverage_subplot(ax, param_coverages, param_list, gaussian_cov, sigma_thresh, title, n_samples_label):
    """Helper to populate one sigma-coverage subplot."""
    colors_sub = plt.cm.tab10(np.linspace(0, 1, max(len(param_list), 1)))
    for idx, param in enumerate(param_list):
        if param in param_coverages:
            ax.plot(sigma_thresh, param_coverages[param] * 100, label=param, color=colors_sub[idx], alpha=0.8)
    # Combined worst-case if multiple params
    if len(param_coverages) > 1:
        stacked = np.column_stack([param_coverages[p] for p in param_list if p in param_coverages])
        if stacked.shape[1] > 0:
            # For each sigma threshold, fraction where ALL params are within that sigma
            all_zs = []
            for p in param_list:
                if p in param_coverages:
                    # Recover z-scores from the calling scope is tricky, so skip combined for now
                    pass
    ax.plot(sigma_thresh, gaussian_cov * 100, label='Ideal Gaussian', color='grey', linewidth=1.5, linestyle='--')
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

# Subplot 1: GR simulated
_plot_coverage_subplot(axes5[0, 0], sim_coverage_gr, gr_params, gaussian_coverage, sigma_thresholds,
                       f'GR Parameters – Simulated ({num_test_samples} samples)', num_test_samples)

# Subplot 2: Modified-GR simulated
_plot_coverage_subplot(axes5[0, 1], sim_coverage_mg, modified_theory_params, gaussian_coverage, sigma_thresholds,
                       f'Modified Theory – Simulated ({num_test_samples} samples)', num_test_samples)

# Subplot 3: GR real data
_plot_coverage_subplot(axes5[1, 0], real_coverage_gr, gr_params, gaussian_coverage, sigma_thresholds,
                       f'GR Parameters – Real Events ({real_n_events} events)', real_n_events)

# Subplot 4: Modified-GR real data (no known true lambda_g for real events)
_plot_coverage_subplot(axes5[1, 1], {}, modified_theory_params, gaussian_coverage, sigma_thresholds,
                       f'Modified Theory – Real Events (no true modified-theory values)', 0)

fig5.suptitle(f'Sigma Coverage – {NUM_EPOCHS} Epochs', fontsize=14, fontweight='bold')
fig5.tight_layout(rect=[0, 0, 1, 0.96])

plot_filename5 = os.path.join(run_plots_dir, f"PyCBC_SigmaCoverage_{PARAM_SET}_{timestamp}.png")
fig5.savefig(plot_filename5, dpi=150, bbox_inches='tight')
plt.close(fig5)
print(f"\n✓ Saved sigma coverage plot: {plot_filename5}")

# ===== Print sigma coverage statistics =====
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

if real_n_events > 0:
    print(f"\n{'='*65}")
    print(f"SIGMA COVERAGE STATISTICS – REAL EVENTS ({real_n_events} events)")
    print(f"{'='*65}")
    print(f"{'Parameter':<15} {'1σ':>8} {'2σ':>8} {'3σ':>8} {'Median z':>10}  {'N events':>8}")
    print(f"{'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*10}  {'-'*8}")
    for param in gr_params:
        if param in real_sigma_deviations and len(real_sigma_deviations[param]) > 0:
            z_arr = np.array(real_sigma_deviations[param])
            c1 = np.mean(z_arr <= 1) * 100
            c2 = np.mean(z_arr <= 2) * 100
            c3 = np.mean(z_arr <= 3) * 100
            med_z = np.median(z_arr)
            print(f"{param:<15} {c1:>7.1f}% {c2:>7.1f}% {c3:>7.1f}% {med_z:>10.3f}  {len(z_arr):>8}")
        else:
            print(f"{param:<15}      -        -        -          -         0")
    print(f"\nIdeal Gaussian:  {68.3:>7.1f}% {95.4:>7.1f}% {99.7:>7.1f}%")
print(f"{'='*65}")














# ===== VIOLIN PLOT: Posterior distributions for all inferred parameters =====
# Each parameter column is z-score normalised by its own posterior mean/std before
# plotting, so all violins share one y-axis in units of σ.
# y = 0        → posterior mean  (solid black line)
# y = ±1       → ±1σ             (dashed black lines)
# y = (true − mean) / std → true value (dashed red line)
print("\nGenerating violin plot of posterior distributions (z-score normalised)...")

best_post = special_posteriors['best']
violin_names   = list(SYMMETRIC_PARAM_NAMES)
violin_n_params = len(violin_names)

PASTEL_PALETTE = [
    '#FFB3BA',  # pastel pink
    '#BAFFC9',  # pastel green
    '#BAE1FF',  # pastel blue
    '#FFFFBA',  # pastel yellow
    '#E8BAFF',  # pastel lavender
    '#FFD9BA',  # pastel peach
    '#C4F0C5',  # pastel mint
    '#FFC8DD',  # pastel rose
]


def _violin_x_extent_v2(verts, y_level):
    """Return (x_min, x_max) of a violin polygon at *y_level* via linear interpolation."""
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
    """
    Draw all parameter violins on one axes using z-score normalisation.
    Each violin's samples are shifted/scaled so mean=0, std=1.
    Mean → solid black horizontal line at y=0.
    ±1σ  → dashed black lines at y=±1.
    True → dashed red line at y=(true-mean)/std.
    """
    n_params = len(param_names_list)

    for pidx, pname in enumerate(param_names_list):
        col_data = samples[:, pidx]
        col_clean = col_data[~np.isnan(col_data)]
        if len(col_clean) == 0:
            continue

        p_mean = np.mean(col_clean)
        p_std  = np.std(col_clean)
        if p_std < 1e-30:
            continue

        # Z-score the samples: centred at 0, unit std
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

        # Mean line at z = 0 (solid black, edge-to-edge)
        xlo, xhi = _violin_x_extent_v2(verts, 0.0)
        if xlo is not None:
            ax.hlines(0.0, xlo, xhi, colors='black', linewidths=1.8, linestyles='solid')

        # ±1σ lines at z = ±1 (dashed black, edge-to-edge)
        for sigma_z in [-1.0, 1.0]:
            slo, shi = _violin_x_extent_v2(verts, sigma_z)
            if slo is not None:
                ax.hlines(sigma_z, slo, shi, colors='black', linewidths=1.2, linestyles='dashed')

        # True value at z = (true − mean) / std (dashed red, edge-to-edge)
        true_val = true_vals[pidx] if pidx < len(true_vals) else np.nan
        if np.isfinite(true_val):
            true_z = (true_val - p_mean) / p_std
            tlo, thi = _violin_x_extent_v2(verts, true_z)
            if tlo is not None:
                ax.hlines(true_z, tlo, thi, colors='red', linewidths=1.5, linestyles='dashed')
            else:
                # true value falls outside the violin body — draw a full-width marker
                ax.hlines(true_z, pidx - 0.35, pidx + 0.35,
                          colors='red', linewidths=1.5, linestyles='dashed')

    # Reference lines that span the full x-range
    ax.axhline(0.0, color='black', linewidth=0.5, linestyle='solid', alpha=0.25, zorder=0)
    ax.axhline(1.0, color='black', linewidth=0.5, linestyle='dashed', alpha=0.25, zorder=0)
    ax.axhline(-1.0, color='black', linewidth=0.5, linestyle='dashed', alpha=0.25, zorder=0)

    # Legend (proxy artists only)
    ax.plot([], [], color='black', linestyle='solid', linewidth=1.8, label='Mean (= 0)')
    ax.plot([], [], color='black', linestyle='dashed', linewidth=1.2, label='±1σ (= ±1)')
    ax.plot([], [], color='red', linestyle='dashed', linewidth=1.5, label='True value')

    ax.set_xticks(range(n_params))
    ax.set_xticklabels([format_param_label(p) for p in param_names_list], fontsize=10)
    ax.set_ylabel('Deviation from posterior mean (σ)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.3)


# --- Build 2-row figure: top = simulated best sample, bottom = real GW event ---
fig_violin, (ax_sim, ax_real) = plt.subplots(
    2, 1, figsize=(max(3.5 * violin_n_params, 12), 12), constrained_layout=True
)

# Row 1: simulated best sample
_draw_violins_zscore(ax_sim,
                     best_post['sym_samples'],
                     best_post['true_sym'],
                     violin_names,
                     PASTEL_PALETTE,
                     f'Simulated – Best Sample (idx {best_post["index"]})')

# Row 2: real GW event
if real_gw_row is not None:
    gw_sym_true = np.array([GW250114_SYMMETRIC_TRUE_PARAMS.get(p, np.nan)
                            for p in violin_names], dtype=np.float32)
    _draw_violins_zscore(ax_real,
                         real_gw_row['symmetric_samples'],
                         gw_sym_true,
                         violin_names,
                         PASTEL_PALETTE,
                         f'Real Event – {real_gw_row["event_name"]}')
else:
    ax_real.text(0.5, 0.5, 'Real GW data not available', ha='center', va='center',
                 transform=ax_real.transAxes, fontsize=14, color='gray', fontweight='bold')
    ax_real.set_title('Real Event', fontsize=12, fontweight='bold')
    ax_real.set_xticks(range(violin_n_params))
    ax_real.set_xticklabels([format_param_label(p) for p in violin_names], fontsize=10)

fig_violin.suptitle(
    f'Posterior Violin Plot – {NUM_EPOCHS} Epochs',
    fontsize=14, fontweight='bold'
)

plot_filename_violin = os.path.join(run_plots_dir,
                                    f"PyCBC_ViolinPlot_{PARAM_SET}_{timestamp}.png")
fig_violin.savefig(plot_filename_violin, dpi=150, bbox_inches='tight')
plt.close(fig_violin)
print(f"✓ Saved violin plot: {plot_filename_violin}")


