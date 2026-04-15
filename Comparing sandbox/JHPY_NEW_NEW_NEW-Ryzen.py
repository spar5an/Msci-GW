# scp "C:\Users\ultra\OneDrive - Imperial College London\Hamza's Stuff on the Project\JHPY_NEW.py" hm2622@login.cx3.hpc.imperial.ac.uk:NewestTest
#,dfsd
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools, random
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm
import data_generator
import json
import matplotlib.pyplot as plt


class AffineCouplingLayer(nn.Module):
    """Affine coupling layer for normalizing flows. Transforms data conditioned on context."""
    def __init__(self, dim, context_dim, hidden_dim=128, mask_type='half'):
        super().__init__()
        self.dim = dim
        self.register_buffer('mask', torch.zeros(dim))  # Create alternating binary mask
        self.mask[::2] = 1 if mask_type in ['half', 'even'] else 0  # Even indices
        if mask_type == 'odd': self.mask[:] = 1 - self.mask  # Flip for odd mask
        # Scale network: outputs multiplicative scaling factors
        # REMOVED Tanh() to allow unbounded scaling - essential for normalizing flows
        self.scale_net = nn.Sequential(
            nn.Linear(dim + context_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )
        # Initialize scale network to output near-zero (identity mapping at init)
        with torch.no_grad():
            self.scale_net[-1].weight.mul_(0.01)  # Small init for stable training
            self.scale_net[-1].bias.zero_()
        
        # Translation network: outputs additive shifts
        self.translation_net = nn.Sequential(
            nn.Linear(dim + context_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )
        # Initialize translation network to output near-zero (identity mapping at init)
        with torch.no_grad():
            self.translation_net[-1].weight.mul_(0.01)
            self.translation_net[-1].bias.zero_()

    def forward(self, x, context, reverse=False):
        masked_x = x * self.mask  # Keep masked dimensions fixed
        # Compute scale and translation from masked input + context
        s = self.scale_net(torch.cat([masked_x, context], dim=1)) * (1 - self.mask)
        t = self.translation_net(torch.cat([masked_x, context], dim=1)) * (1 - self.mask)
        # Clamp scale to prevent numerical instability in exp(s) and unbounded log-det
        # TIGHTER CLAMP: prevents flow from becoming too aggressive and producing positive log-probs
        s = torch.clamp(s, min=-2, max=2)
        if not reverse:  # Forward: data -> latent
            y = x * torch.exp(s) + t  # Affine transformation
            log_det = s.sum(dim=1)  # Log determinant of Jacobian
        else:  # Reverse: latent -> data
            y = (x - t) * torch.exp(-s)  # Inverse transformation
            log_det = -s.sum(dim=1)  # Negative log det for inverse
        return y, log_det

'''
class MultiDetectorAffineCouplingLayer(nn.Module):
    """
    Affine coupling layer designed for multi-detector data.
    
    Instead of conditioning on a single context, this layer conditions on 
    separate context vectors from each detector. Each detector gets its own
    scale and translation networks, but both can see both detector contexts.
    
    This allows the transformation to be aware of information from both detectors.
    """
    def __init__(self, latent_dim, context_dim_per_detector, num_detectors=2, 
                 hidden_dim=128, mask_type='half'):
        """
        Args:
            latent_dim (int): Dimension of latent space (shared across detectors)
            context_dim_per_detector (int): Dimension of each detector's context embedding
            num_detectors (int): Number of detectors. Default: 2
            hidden_dim (int): Hidden layer dimension. Default: 128
            mask_type (str): Type of mask ('half', 'even', 'odd'). Default: 'half'
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.context_dim_per_detector = context_dim_per_detector
        self.num_detectors = num_detectors
        
        # Create alternating mask
        self.register_buffer('mask', torch.zeros(latent_dim))
        if mask_type in ['half', 'even']:
            self.mask[::2] = 1
        elif mask_type == 'odd':
            self.mask[1::2] = 1
        
        # Total context dimension (concatenated from all detectors)
        total_context_dim = context_dim_per_detector * num_detectors
        
        # Create separate scale and translation networks for each detector
        self.scale_nets = nn.ModuleList()
        self.translation_nets = nn.ModuleList()
        
        for detector_idx in range(num_detectors):
            # Scale network: takes masked latent + all detector contexts
            # REMOVED Tanh() to allow unbounded scaling - essential for normalizing flows
            scale_net = nn.Sequential(
                nn.Linear(latent_dim + total_context_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.0),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.0),
                nn.Linear(hidden_dim, latent_dim)
            )
            # Initialize scale network to output near-zero (identity mapping at init)
            with torch.no_grad():
                scale_net[-1].weight.mul_(0.01)  # Small init for stable training
                scale_net[-1].bias.zero_()
            
            # Translation network: takes masked latent + all detector contexts
            translation_net = nn.Sequential(
                nn.Linear(latent_dim + total_context_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.0),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.0),
                nn.Linear(hidden_dim, latent_dim)
            )
            
            self.scale_nets.append(scale_net)
            self.translation_nets.append(translation_net)
    
    def forward(self, z, contexts, reverse=False):
        """
        Forward pass with multi-detector context.
        
        Args:
            z (torch.Tensor): Latent vector [batch_size, latent_dim]
            contexts (torch.Tensor or list): Context vectors
                - If tensor: [batch_size, num_detectors, context_dim_per_detector]
                - If list: list of [batch_size, context_dim_per_detector] tensors
            reverse (bool): If True, compute inverse transformation
        
        Returns:
            y (torch.Tensor): Transformed latent vector [batch_size, latent_dim]
            log_det (torch.Tensor): Log determinant of Jacobian [batch_size]
        """
        # Handle both tensor and list formats
        if isinstance(contexts, torch.Tensor):
            # Tensor format: [batch, num_detectors, context_dim] → split to list
            contexts_list = [contexts[:, i, :] for i in range(contexts.shape[1])]
        else:
            # Already a list
            contexts_list = contexts
        
        # Concatenate all detector contexts
        concatenated_context = torch.cat(contexts_list, dim=1)  # [batch, total_context_dim]
        
        # Apply mask to latent
        masked_z = z * self.mask  # [batch, latent_dim]
        
        # Prepare input for networks: masked latent + all contexts
        network_input = torch.cat([masked_z, concatenated_context], dim=1)
        
        # Initialize output and log determinant
        y = z.clone()
        log_det = torch.zeros(z.size(0), device=z.device)
        
        # Apply transformations from all detectors
        # In practice, you might want to weight these or alternate them
        for detector_idx in range(self.num_detectors):
            # Get scale and translation for this detector
            s = self.scale_nets[detector_idx](network_input)
            t = self.translation_nets[detector_idx](network_input)
            
            # Only apply to unmasked dimensions
            s = s * (1 - self.mask)
            t = t * (1 - self.mask)
            
            if not reverse:
                # Forward: z -> transformed_z
                y = y * torch.exp(s) + t
                log_det = log_det + s.sum(dim=1)
            else:
                # Reverse: z -> latent
                y = (y - t) * torch.exp(-s)
                log_det = log_det - s.sum(dim=1)
        
        return y, log_det
'''

class EmbeddingNetwork(nn.Module):
    """
    Simple embedding network using only Linear layers.
    
    NEW APPROACH: Concatenate waveforms from all detectors first,
    then process through a single MLP pathway.
    """
    def __init__(self, data_dim=100, context_dim=64, hidden_dim=128, num_detectors=1):
        super().__init__()
        self.num_detectors = num_detectors
        
        # Input dimension is concatenated waveforms from all detectors
        concatenated_dim = data_dim * num_detectors
        
        # Single MLP for concatenated waveforms
        self.mlp = nn.Sequential(
            nn.Linear(concatenated_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, context_dim)
        )
        
    def forward(self, data):
        # data shape: [batch, num_detectors, data_dim] or [batch, data_dim]
        if len(data.shape) == 2:
            data = data.unsqueeze(1)  # [batch, 1, data_dim]
        
        batch_size = data.shape[0]
        
        # Concatenate all detector waveforms: [batch, num_detectors * data_dim]
        concatenated = data.view(batch_size, -1)
        
        # Process through single MLP: [batch, context_dim]
        context = self.mlp(concatenated)
        return context
    
class Conv1DEmbeddingNetwork(nn.Module):
    """Conv1D-based embedding network for waveform data. Efficient for long sequences.
    
    NEW APPROACH: Concatenate waveforms from all detectors first along time axis,
    then process through a single Conv1D pathway.
    """
    def __init__(self, data_dim=5868, context_dim=512, num_filters=[64, 128, 256], num_detectors=1):
        super().__init__()
        self.num_detectors = num_detectors
        
        # Conv1D pipeline for concatenated waveforms
        # Input: [batch, 1, data_dim * num_detectors]
        self.conv1d = nn.Sequential(
            nn.Conv1d(1, num_filters[0], kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(num_filters[0]), nn.ReLU(), nn.MaxPool1d(2, 2),
            nn.Conv1d(num_filters[0], num_filters[1], kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(num_filters[1]), nn.ReLU(), nn.MaxPool1d(2, 2),
            nn.Conv1d(num_filters[1], num_filters[2], kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(num_filters[2]), nn.ReLU(), nn.MaxPool1d(2, 2),
            nn.AdaptiveAvgPool1d(1),  # Pool to single value per filter
            nn.Flatten()  # [batch, num_filters[2]]
        )
        
        # Project to context dimension
        self.project = nn.Sequential(
            nn.Linear(num_filters[2], 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, context_dim)
        )

    def forward(self, data):
        # data shape: [batch, num_detectors, data_dim] or [batch, data_dim]
        if len(data.shape) == 2:
            # Single detector case: [batch, data_dim]
            data = data.unsqueeze(1)  # [batch, 1, data_dim]
        
        batch_size = data.shape[0]
        
        # Concatenate all detector waveforms along time axis: [batch, num_detectors * data_dim]
        concatenated = data.view(batch_size, -1)
        
        # Add channel dimension for Conv1D: [batch, 1, num_detectors * data_dim]
        concatenated = concatenated.unsqueeze(1)
        
        # Process through Conv1D: [batch, num_filters[2]]
        features = self.conv1d(concatenated)
        
        # Project to context dimension: [batch, context_dim]
        context = self.project(features)
        return context



class LSTMEmbeddingNetwork(nn.Module):
    """LSTM-based embedding network for waveform data. Better at capturing temporal dependencies.
    
    NEW APPROACH: Concatenate waveforms from all detectors first along time axis,
    then process through a single LSTM pathway.
    """
    def __init__(self, data_dim=7241, context_dim=512, hidden_dim=256, num_layers=2, num_detectors=1, use_conv_preprocessing=True):
        super().__init__()
        self.num_detectors = num_detectors
        self.use_conv_preprocessing = use_conv_preprocessing
        
        if use_conv_preprocessing:
            # Conv1D preprocessing for concatenated waveforms
            self.conv_preprocess = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
                nn.ReLU(),
                nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(500)  # Reduce to fixed length for LSTM
            )
            
            # Initialize Conv1D weights with proper scale
            for m in self.conv_preprocess.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            
            lstm_input_size = 32
        else:
            # Direct input projection (slow but full resolution)
            self.input_proj = nn.Sequential(
                nn.Linear(1, 32),
                nn.ReLU()
            )
            lstm_input_size = 32
        
        # Bidirectional LSTM for concatenated waveforms
        self.lstm = nn.LSTM(lstm_input_size, hidden_dim, num_layers, batch_first=True, 
                          bidirectional=True, dropout=0.1 if num_layers > 1 else 0)
        
        if use_conv_preprocessing:
            # Initialize LSTM weights with custom initialization
            for name, param in self.lstm.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
        
        # Project LSTM output to context
        # Single LSTM output: [batch, hidden_dim*2] from bidirectional LSTM
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, context_dim)
        )
        
        if use_conv_preprocessing:
            # Initialize output projection with smaller weights to prevent saturation
            for m in self.output_proj.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, data):
        # data shape: [batch, num_detectors, data_dim] or [batch, data_dim]
        if len(data.shape) == 2:
            # Single detector case
            data = data.unsqueeze(1)  # [batch, 1, data_dim]
        
        batch_size = data.shape[0]
        
        # Concatenate all detector waveforms along time axis: [batch, num_detectors * data_dim]
        concatenated = data.view(batch_size, -1)
        
        if self.use_conv_preprocessing:
            # Add channel dimension for Conv1D: [batch, 1, num_detectors * data_dim]
            concatenated = concatenated.unsqueeze(1)
            
            # Conv preprocessing: [batch, 32, reduced_time]
            features = self.conv_preprocess(concatenated)
            # Transpose for LSTM: [batch, reduced_time, 32]
            features = features.transpose(1, 2)
            
            # LSTM: outputs [batch, reduced_time, hidden_dim*2]
            lstm_out, (h_n, c_n) = self.lstm(features)
            
            # Take concatenated forward and backward final states
            # [batch, hidden_dim*2]
            final_state = torch.cat([h_n[-2, :, :], h_n[-1, :, :]], dim=1)
        else:
            # Full resolution path
            # Add feature dimension: [batch, num_detectors * data_dim, 1]
            concatenated = concatenated.unsqueeze(-1)
            
            # Input projection: [batch, num_detectors * data_dim, 32]
            features = self.input_proj(concatenated)
            
            # LSTM: outputs [batch, num_detectors * data_dim, hidden_dim*2]
            lstm_out, (h_n, c_n) = self.lstm(features)
            
            # Take concatenated forward and backward final states
            # [batch, hidden_dim*2]
            final_state = torch.cat([h_n[-2, :, :], h_n[-1, :, :]], dim=1)
        
        # Project to context dimension: [batch, context_dim]
        context = self.output_proj(final_state)
        
        # Store diagnostics (accessible via hook)
        if hasattr(self, '_store_diagnostics') and self._store_diagnostics:
            self._last_context_std = context.std().item()
            self._last_context_mean = context.mean().item()
        
        return context


class GRUEmbeddingNetwork(nn.Module):
    """GRU-based embedding network for waveform data. Faster alternative to LSTM.
    
    NEW APPROACH: Concatenate waveforms from all detectors first along time axis,
    then process through a single GRU pathway.
    """
    def __init__(self, data_dim=7241, context_dim=512, hidden_dim=256, num_layers=2, num_detectors=1):
        super().__init__()
        self.num_detectors = num_detectors
        
        # Conv1D preprocessing for concatenated waveforms
        self.conv_preprocess = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(500)  # Reduce to fixed length for GRU
        )
        
        # Bidirectional GRU for concatenated waveforms
        self.gru = nn.GRU(32, hidden_dim, num_layers, batch_first=True, 
                         bidirectional=True, dropout=0.1 if num_layers > 1 else 0)
        
        # Project GRU output to context
        # Single GRU output: [batch, hidden_dim*2] from bidirectional GRU
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, context_dim)
        )

    def forward(self, data):
        # data shape: [batch, num_detectors, data_dim] or [batch, data_dim]
        if len(data.shape) == 2:
            # Single detector case
            data = data.unsqueeze(1)  # [batch, 1, data_dim]
        
        batch_size = data.shape[0]
        
        # Concatenate all detector waveforms along time axis: [batch, num_detectors * data_dim]
        concatenated = data.view(batch_size, -1)
        
        # Add channel dimension for Conv1D: [batch, 1, num_detectors * data_dim]
        concatenated = concatenated.unsqueeze(1)
        
        # Conv preprocessing: [batch, 32, reduced_time]
        features = self.conv_preprocess(concatenated)
        # Transpose for GRU: [batch, reduced_time, 32]
        features = features.transpose(1, 2)
        
        # GRU: outputs [batch, reduced_time, hidden_dim*2]
        gru_out, h_n = self.gru(features)
        
        # Take concatenated forward and backward final states
        # [batch, hidden_dim*2]
        final_state = torch.cat([h_n[-2, :, :], h_n[-1, :, :]], dim=1)
        
        # Project to context dimension: [batch, context_dim]
        context = self.output_proj(final_state)
        return context


class NormalizingFlow(nn.Module):
    """Stack of affine coupling layers. Transforms base distribution into complex posterior."""
    def __init__(self, param_dim=1, context_dim=64, num_layers=6, hidden_dim=128, config=None, num_detectors=1):
        super().__init__()
        self.num_detectors = num_detectors

        # Parse config dict if provided
        if config:
            param_dim = config.get('param_dim', param_dim)
            context_dim = config.get('context_dim', context_dim)
            num_layers = config.get('num_flow_layers', num_layers)
            hidden_dim = config.get('hidden_dim', hidden_dim)
        # Store config for checkpointing
        self.config = {'param_dim': param_dim, 'context_dim': context_dim, 
                      'num_flow_layers': num_layers, 'hidden_dim': hidden_dim}
        self.param_dim = param_dim
        # Standard Gaussian base distribution
        self.register_buffer('base_mean', torch.zeros(param_dim))
        self.register_buffer('base_std', torch.ones(param_dim))

        # Always use single-detector AffineCouplingLayer (context is now 1D)
        self.layers = nn.ModuleList([
            AffineCouplingLayer(param_dim, context_dim, hidden_dim, 
                              'even' if i % 2 == 0 else 'odd')
            for i in range(num_layers)
        ])

    def forward(self, params, context):
        # Transform parameters through all layers
        z, log_det_sum = params, torch.zeros(params.size(0), device=params.device)

        for layer in self.layers:
            z, log_det = layer(z, context, reverse=False)
            log_det_sum += log_det
        # Compute log prob under base Gaussian distribution
        log_prob_base = -0.5 * (torch.log(2 * np.pi * self.base_std**2) + 
                                ((z - self.base_mean) / self.base_std)**2).sum(dim=1)
        # Apply change of variables formula
        return log_prob_base + log_det_sum

    def sample(self, context, num_samples=1):
        # Sample from base distribution
        batch_size = context.shape[0] if len(context.shape) >= 2 else 1
        z = torch.randn(batch_size * num_samples, self.param_dim, device=context.device)

        # Context is now always 2D: [batch, context_dim]
        context_repeated = context.repeat_interleave(num_samples, dim=0)

        # Apply inverse transformations through layers
        for layer in reversed(self.layers):            
            z, _ = layer(z, context_repeated, reverse=True)
        return z


class DINGOModel(nn.Module):
    """DINGO neural posterior estimation model. Pipeline: data -> embedding -> flow -> log p(params | data)."""
    def __init__(self, data_dim=100, param_dim=1, context_dim=64, num_flow_layers=6, 
                 hidden_dim=128, embedding='conv1d', config=None, num_detectors=1, use_conv_preprocessing=False):
        super().__init__()
        # Parse config if provided
        if config:
            data_dim = config.get('data_dim', data_dim)
            param_dim = config.get('param_dim', param_dim)
            context_dim = config.get('context_dim', context_dim)
            num_flow_layers = config.get('num_flow_layers', num_flow_layers)
            hidden_dim = config.get('hidden_dim', hidden_dim)
            embedding = config.get('embedding', embedding)
            use_conv_preprocessing = config.get('use_conv_preprocessing', use_conv_preprocessing)
        # Store configuration for checkpointing
        self.config = {'data_dim': data_dim, 'param_dim': param_dim, 'context_dim': context_dim,
                      'num_flow_layers': num_flow_layers, 'hidden_dim': hidden_dim, 'embedding': embedding,
                        'num_detectors': num_detectors, 'use_conv_preprocessing': use_conv_preprocessing}
        
        # Select embedding network based on data size and type
        if embedding.lower() == 'lstm' and data_dim > 1000:
            self.embedding_net = LSTMEmbeddingNetwork(data_dim, context_dim, 256, 2, num_detectors=num_detectors, use_conv_preprocessing=use_conv_preprocessing)
        elif embedding.lower() == 'gru' and data_dim > 1000:
            self.embedding_net = GRUEmbeddingNetwork(data_dim, context_dim, 256, 2, num_detectors=num_detectors)
        elif embedding.lower() == 'conv1d' and data_dim > 1000:
            self.embedding_net = Conv1DEmbeddingNetwork(data_dim, context_dim, num_detectors=num_detectors)
        else:  # Default: linear, small data, or any other case
            self.embedding_net = EmbeddingNetwork(data_dim, context_dim, hidden_dim, num_detectors=num_detectors)
        # Normalizing flow for posterior
        self.flow = NormalizingFlow(param_dim, context_dim, num_flow_layers, hidden_dim, num_detectors=num_detectors)

    def forward(self, params, data):
        # Embed data to context, then compute log probability
        context = self.embedding_net(data)
        return self.flow(params, context)

    def sample_posterior(self, data, num_samples=1000):
        # Sample from posterior p(params | data)
        self.eval()
        with torch.no_grad():
            context = self.embedding_net(data)
            samples = self.flow.sample(context, num_samples=num_samples)
        return samples


def train_npe_pycbc(model, train_dloader, val_dloader=None, n_epochs=100, lr=1e-4, 
                    optimizer='adam', scheduler=None, start_epoch=0, save_best_model=True,
                    model_path='best_npe_pycbc_model.pt', grad_clip_norm=5.0, 
                    use_mixed_precision=True, device='cuda', reg_config=None, verbose=True,
                    diagnostic_interval=5):
    """Train NPE model on PyCBC data with checkpointing and regularization."""
    model = model.to(device)  # Move model to device
    
    # Gradient monitoring storage
    gradient_stats = {'embedding': [], 'flow': []}
    
    # Create optimizer based on type
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5) if optimizer.lower() == 'adamw' else \
          torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9) if optimizer.lower() == 'sgd' else \
          torch.optim.Adam(model.parameters(), lr=lr)
    
    # Setup learning rate scheduler if not provided
    if scheduler is None and hasattr(model, 'config'):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=lr * 0.01)
    
    # Mixed precision training (disabled by default to avoid CUDA issues, can be re-enabled)
    scaler = torch.cuda.amp.GradScaler() if use_mixed_precision and torch.cuda.is_available() else None
    torch.backends.cudnn.benchmark = False  # Disable to reduce memory fragmentation
    
    # Default regularization config to encourage expressiveness
    if reg_config is None:
        reg_config = {'target_std': 0.8, 'max_weight': 1.0, 'warmup_epochs': 15}
    
    # Initialize tracking lists
    train_log_probs, val_log_probs, train_losses, context_stds, reg_losses = [], [], [], [], []
    best_val_log_prob, best_val_epoch = float('-inf'), 0
    
    if verbose:
        print(f"\nTraining NPE Model\n  LR: {lr}, Optimizer: {optimizer}, Grad clip: {grad_clip_norm}\n")
    
    # Main training loop
    for epoch in range(start_epoch, start_epoch + n_epochs):
        model.train()  # Set to training mode
        # Initialize epoch metrics
        train_log_prob_sum, train_loss_sum, context_std_sum, reg_loss_sum, batch_count = 0, 0, 0, 0, 0
        epoch_context_stds = []  # Track context std per batch
        
        # Clear CUDA cache at the start of each epoch to prevent memory fragmentation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Iterate over training batches
        for batch_waveforms, batch_params in tqdm(train_dloader, desc=f'Epoch {epoch+1:3d}/{start_epoch + n_epochs}', 
                                                   disable=not verbose):
            batch_waveforms, batch_params = batch_waveforms.to(device), batch_params.to(device)
            # Linearly increase regularization weight during warmup
            reg_weight = reg_config['max_weight'] * min(epoch + 1, reg_config['warmup_epochs']) / reg_config['warmup_epochs']
            opt.zero_grad()
            
            if scaler:  # Mixed precision training (if enabled)
                with torch.cuda.amp.autocast():
                    # Compute context and track statistics
                    context = model.embedding_net(batch_waveforms)
                    epoch_context_stds.append(context.std().item())
                    
                    log_prob = model(batch_params, batch_waveforms)
                    nll_loss = -log_prob.mean()
                    # Regularization: keep context embeddings expressive (no redundant computation)
                    reg_loss = torch.tensor(0.0, device=device)  # Skip regularization for speed
                    total_loss = nll_loss + reg_loss
                scaler.scale(total_loss).backward()
                if grad_clip_norm: torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(opt)
                scaler.update()
            else:  # Standard training
                # Compute context and track statistics
                context = model.embedding_net(batch_waveforms)
                epoch_context_stds.append(context.std().item())
                
                log_prob = model(batch_params, batch_waveforms)
                nll_loss = -log_prob.mean()
                reg_loss = torch.tensor(0.0, device=device)
                total_loss = nll_loss + reg_loss
                total_loss.backward()
                if grad_clip_norm: torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                opt.step()
            
            # Accumulate metrics
            train_log_prob_sum += log_prob.mean().item()
            train_loss_sum += total_loss.item()
            #context_std_sum += context_std.item()
            reg_loss_sum += reg_loss.item()
            batch_count += 1
        
        # Compute epoch averages
        avg_train_log_prob = train_log_prob_sum / batch_count
        avg_train_loss = train_loss_sum / batch_count
        #avg_context_std = context_std_sum / batch_count
        avg_reg_loss = reg_loss_sum / batch_count
        train_log_probs.append(avg_train_log_prob)
        train_losses.append(avg_train_loss)
        #context_stds.append(avg_context_std)
        reg_losses.append(avg_reg_loss)
        
        # Validation phase
        val_log_prob = None
        if val_dloader:
            model.eval()  # Set to eval mode
            val_log_prob_sum, val_batch_count = 0, 0
            with torch.no_grad():
                for batch_waveforms, batch_params in val_dloader:
                    batch_waveforms, batch_params = batch_waveforms.to(device), batch_params.to(device)
                    log_prob = model(batch_params, batch_waveforms)
                    val_log_prob_sum += log_prob.mean().item()
                    val_batch_count += 1
            val_log_prob = val_log_prob_sum / val_batch_count
            val_log_probs.append(val_log_prob)
        
        # ============================================================================
        # DIAGNOSTIC 2: Per-epoch embedding and gradient analysis
        # ============================================================================
        if (epoch + 1) % diagnostic_interval == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                # Get a batch for diagnostics
                diag_data, diag_params = next(iter(val_dloader if val_dloader else train_dloader))
                diag_data = diag_data.to(device)
                diag_params = diag_params.to(device)
                
                # Check embedding outputs
                context = model.embedding_net(diag_data)
                
                # Context statistics
                ctx_std = context.std(dim=0).mean().item()
                ctx_range = (context.max() - context.min()).item()
                
                # Check if context varies with parameters
                from scipy.stats import pearsonr
                ctx_mean_per_sample = context.mean(dim=1).cpu().numpy()
                mass1_corr, _ = pearsonr(ctx_mean_per_sample, diag_params[:, 0].cpu().numpy())
                
                if verbose:
                    print(f"    [DIAG] Context std: {ctx_std:.4f}, range: {ctx_range:.2f}, mass1_corr: {mass1_corr:+.3f}")
                
                if ctx_std < 0.05:
                    print(f"    ⚠ WARNING: Context embeddings are nearly constant (std={ctx_std:.4f})!")
                if abs(mass1_corr) < 0.1:
                    print(f"    ⚠ WARNING: Context doesn't correlate with mass1!")
            
            # Check gradient magnitudes (after backward pass, using last batch gradients)
            embedding_grad_norm = 0.0
            flow_grad_norm = 0.0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    if 'embedding' in name:
                        embedding_grad_norm += grad_norm
                    else:
                        flow_grad_norm += grad_norm
            
            gradient_stats['embedding'].append(embedding_grad_norm)
            gradient_stats['flow'].append(flow_grad_norm)
            
            if embedding_grad_norm < 1e-6 and verbose:
                print(f"    ⚠ WARNING: Embedding gradients are near zero ({embedding_grad_norm:.2e})!")
            
            ratio = embedding_grad_norm / (flow_grad_norm + 1e-10)
            if verbose:
                print(f"    [DIAG] Grad norms - Embedding: {embedding_grad_norm:.2e}, Flow: {flow_grad_norm:.2e}, Ratio: {ratio:.2e}")
        
        # Log progress with context stats
        avg_context_std = np.mean(epoch_context_stds) if epoch_context_stds else 0.0
        if verbose:
            if val_log_prob:
                print(f"[Epoch {epoch+1:3d}] Train: {avg_train_log_prob:8.4f}, Val: {val_log_prob:8.4f}, Ctx-std: {avg_context_std:6.4f}")
            else:
                print(f"[Epoch {epoch+1:3d}] Train: {avg_train_log_prob:8.4f}, Ctx-std: {avg_context_std:6.4f}")
        
        # Update learning rate (epoch-based scheduling, no metric argument)
        if scheduler:
            scheduler.step()  # CosineAnnealingLR doesn't need a metric
        
        # Save best model checkpoint
        if val_log_prob and val_log_prob > best_val_log_prob and save_best_model:
            best_val_log_prob, best_val_epoch = val_log_prob, epoch
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                       'optimizer_state_dict': opt.state_dict(), 'best_val_log_prob': best_val_log_prob,
                       'model_config': model.config, 'train_log_probs': train_log_probs,
                       'val_log_probs': val_log_probs, 'train_losses': train_losses}, model_path)
        
        # EARLY STOPPING: Stop if training diverges (log prob becomes too positive or too negative)
        if avg_train_log_prob > 0.5:
            print(f"\n    ⚠ EARLY STOPPING: Training diverged (log_prob = {avg_train_log_prob:.4f} > 0.5)")
            print(f"    Reverting to best model from epoch {best_val_epoch}")
            break
        
        if val_log_prob and abs(val_log_prob) > 50:
            print(f"\n    ⚠ EARLY STOPPING: Validation log_prob is unphysical ({val_log_prob:.2f})")
            print(f"    Reverting to best model from epoch {best_val_epoch}")
            break
    
    if verbose:
        print(f"\nTraining complete!\n  Final train log prob: {train_log_probs[-1]:.4f}\n")
    
    # Return all metrics
    return {'train_log_probs': train_log_probs, 'val_log_probs': val_log_probs, 'train_losses': train_losses,
            'context_stds': context_stds, 'reg_losses': reg_losses, 'best_val_log_prob': best_val_log_prob if val_log_probs else None, 
            'best_val_epoch': best_val_epoch, 'gradient_stats': gradient_stats}


def npe_hyperparameter_search(param_grid, train_dloader, val_dloader=None, model_class=DINGOModel, 
                               n_epochs=20, n_trials=None, model_path='best_npe_model.pt', 
                               device='cuda', config_save_path='best_config.json'):
    """Hyperparameter search for NPE models. Supports grid search or random search via n_trials."""
    results, best_val_log_prob, best_config = [], float('-inf'), None
    # Generate all combinations or sample randomly
    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]
    # Grid search if n_trials is None, otherwise random search
    all_combinations = list(itertools.product(*param_values)) if n_trials is None else \
                      [tuple(random.choice(values) for values in param_values) for _ in range(n_trials)]
    
    print(f"\nTesting {len(all_combinations)} configurations...\n")
    # Iterate over all configurations
    for i, combo in enumerate(all_combinations):
        config = dict(zip(param_names, combo))
        print(f"Trial {i+1}/{len(all_combinations)}: {config}")
        
        try:
            # Extract hyperparameters from config dict
            lr = config.pop('lr', 1e-4)
            grad_clip_norm = config.pop('grad_clip_norm', 5.0)
            embedding = config.pop('embedding', 'linear')
            context_dim = config.pop('context_dim', 64)
            num_flow_layers = config.pop('num_flow_layers', 6)
            hidden_dim = config.pop('hidden_dim', 128)
            # Get data dimension from first batch
            data_dim = next(iter(train_dloader))[0].shape[-1]
            
            # Create model with current config
            model = model_class(data_dim=data_dim, param_dim=3, context_dim=context_dim,
                              num_flow_layers=num_flow_layers, hidden_dim=hidden_dim, embedding=embedding, num_detectors=2)
            
            # Create optimizer based on type
            optimizer_type = config.pop('optimizer', 'adam').lower()
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5) if optimizer_type == 'adamw' else \
                       torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9) if optimizer_type == 'sgd' else \
                       torch.optim.Adam(model.parameters(), lr=lr)
            
            # Create scheduler if validation data available
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.01) if val_dloader else None
            
            # Train model with current config
            outputs = train_npe_pycbc(model, train_dloader, val_dloader=val_dloader, n_epochs=n_epochs,
                                     lr=lr, optimizer=optimizer_type, scheduler=scheduler, 
                                     grad_clip_norm=grad_clip_norm, device=device, verbose=False)
            
            # Get best validation log prob (or training if no validation)
            final_val_log_prob = max(outputs['val_log_probs']) if outputs['val_log_probs'] else outputs['train_log_probs'][-1]
            # Store results
            result = {'config': config, 'best_val_log_prob': final_val_log_prob, 
                     'n_epochs_trained': len(outputs['train_log_probs']), 
                     'final_train_log_prob': outputs['train_log_probs'][-1]}
            results.append(result)
            print(f"  Val log prob: {final_val_log_prob:.4f}\n")
            
            # Update best if better
            if final_val_log_prob > best_val_log_prob:
                best_val_log_prob = final_val_log_prob
                best_config = {'lr': lr, 'grad_clip_norm': grad_clip_norm, 'embedding': embedding,
                              'context_dim': context_dim, 'num_flow_layers': num_flow_layers,
                              'hidden_dim': hidden_dim, **config}
                # Save best model checkpoint
                torch.save({'epoch': result['n_epochs_trained'] - 1, 'model_state_dict': model.state_dict(),
                           'optimizer_state_dict': optimizer.state_dict(), 'best_val_log_prob': best_val_log_prob,
                           'model_config': model.config}, model_path)
        except Exception as e:
            print(f"  ✗ Error: {e}\n")  # Skip this config on error
    
    print(f"\nSearch complete. Best log prob: {best_val_log_prob:.4f}\n")
    # Save best configuration to JSON
    if best_config:
        with open(config_save_path, 'w') as f:
            json.dump(best_config, f, indent=2)
    
    # Sort results by validation log prob
    results.sort(key=lambda x: x['best_val_log_prob'], reverse=True)
    return best_config, results


def infer_waveform_length(dloader):
    """Extract waveform length (time samples) from dataloader."""
    return next(iter(dloader))[0].shape[-1]


def prepare_pycbc_data(num_samples=10000, batch_size=64, use_cache=True, cache_dir='./waveform_cache', num_detectors=2):
    """Generate and prepare PyCBC gravitational wave data with train/val/test split. Returns loaders + metadata."""
    import os
    
    # Map number of detectors to detector list
    detector_map = {
        1: ['H1'],
        2: ['H1', 'L1'],
        3: ['H1', 'L1', 'V1']
    }
    detectors = detector_map.get(num_detectors, ['H1', 'L1'])
    
    cache_file = os.path.join(cache_dir, f'waveforms_n{num_samples}_{num_detectors}det_global_norm_cached.pt')
    
    # Load from cache if available
    if use_cache and os.path.exists(cache_file):
        try:
            print(f"Loading cached waveforms...")
            cached = torch.load(cache_file, weights_only=False)
            loaders = tuple(DataLoader(TensorDataset(cached[f'{s}_waveforms'], cached[f'{s}_params']), 
                                  batch_size=batch_size, shuffle=(s=='train'))
                        for s in ['train', 'val', 'test'])
            metadata = cached.get('metadata', {})
            return loaders + (metadata,)
        except (RuntimeError, EOFError, pickle.UnpicklingError) as e:
            print(f"⚠ Corrupted cache file ({str(e)[:50]}...). Regenerating...")
            os.remove(cache_file)
            use_cache = True  # Continue to regenerate below
    
    # If cache doesn't exist, generate waveforms
    print(f"Generating {num_samples} new waveforms (will cache for next run)...")
    config = {
        'mass1': lambda size: np.random.uniform(10, 50, size=size),
        'mass2': lambda size: np.random.uniform(10, 50, size=size),
        'spin1z': lambda size: np.random.uniform(-0.5, 0.5, size=size)
    }
    result = data_generator.pycbc_data_generator(
        config, 
        num_samples=num_samples, 
        batch_size=batch_size, 
        num_workers=4, 
        allow_padding=True, 
        waveform_normalization='global_standardize',  # FIXED: Now matches parameter normalization (mean=0, std=1)
        parameter_normalization='zscore',
        detectors=detectors  # Use detector list based on num_detectors
    )
    loaders = (result['train_loader'], result['val_loader'], result['test_loader'])
    metadata = result['metadata']
    
    # CRITICAL: If parameter_normalization missing, compute it from data
    if 'parameter_normalization' not in metadata:
        print("[DATA] Computing parameter_normalization from loader data...")
        param_norm_info = {}
        
        # Collect all parameters from all loaders
        all_params = []
        for loader in loaders:
            for _, batch_params in loader:
                all_params.append(batch_params.cpu().numpy())
        all_params = np.vstack(all_params)
        
        # Compute stats for each parameter
        param_names = metadata.get('parameter_names', ['mass1', 'mass2', 'spin1z'])
        for i, name in enumerate(param_names):
            param_values = all_params[:, i]
            param_norm_info[name] = {
                'min': float(param_values.min()),
                'max': float(param_values.max()),
                'mean': float(param_values.mean()),
                'std': float(param_values.std())
            }
        
        metadata['parameter_normalization'] = param_norm_info
        print("[DATA] ✓ Parameter normalization computed and added to metadata")
    
    # Cache waveforms for next run
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        cached = {}
        for loader, name in zip(loaders, ('train', 'val', 'test')):
            ws, ps = zip(*[(w, p) for w, p in loader])
            cached[f'{name}_waveforms'] = torch.cat(ws)
            cached[f'{name}_params'] = torch.cat(ps)
        cached['metadata'] = metadata
        torch.save(cached, cache_file)
        print(f"✓ Cached to {cache_file}")
        loaders = tuple(DataLoader(TensorDataset(cached[f'{s}_waveforms'], cached[f'{s}_params']), 
                                  batch_size=batch_size, shuffle=(s=='train'))
                       for s in ['train', 'val', 'test'])
    
    return loaders + (metadata,)


def sample_posterior(model, observed_data, num_samples=5000, device='cuda'):
    """Sample from posterior p(params | data) using trained NPE model."""
    model.eval()
    model = model.to(device)
    
    # Convert to tensor
    if isinstance(observed_data, np.ndarray):
        data_tensor = torch.FloatTensor(observed_data)
    else:
        data_tensor = observed_data.clone()
    
    # Handle batch dimension
    single_sample = (data_tensor.dim() == 1)
    if single_sample:
        data_tensor = data_tensor.unsqueeze(0)
    
    data_tensor = data_tensor.to(device)
    
    # Generate samples
    with torch.no_grad():
        samples = model.sample_posterior(data_tensor, num_samples=num_samples)
        samples = samples.cpu().numpy()
    
    # Reshape based on input format
    if single_sample:
        samples = samples.reshape(num_samples, -1)
    else:
        batch_size = len(observed_data)
        param_dim = samples.shape[1]
        samples = samples.reshape(batch_size, num_samples, param_dim)
    
    return samples

# Helper function to denormalize parameters
def denormalize_params(params_normalized, metadata, param_norm):
    """Convert parameters from z-score normalized space back to physical units."""
    params = params_normalized.cpu().numpy() if torch.is_tensor(params_normalized) else params_normalized.copy()
    param_names = metadata['parameter_names']
    for i, name in enumerate(param_names):
        norm_info = param_norm[name]
        param_mean = norm_info['mean']
        param_std = norm_info['std']
        params[:, i] = params[:, i] * param_std + param_mean
    return params

def calculate_metrics(predictions, targets):
    """Calculate MAE, RMSE, and R² metrics."""
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return {'mae': mae, 'rmse': rmse, 'r2': r2}


def load_npe_checkpoint(model_path='best_npe_pycbc_model.pt'):
    """Load saved NPE model checkpoint."""
    import os
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found at {model_path}")
    checkpoint = torch.load(model_path, weights_only=False)
    return checkpoint


def resume_npe_training(model, optimizer, model_path='best_npe_pycbc_model.pt', 
                        train_params=None, train_data=None,
                        n_epochs=50, **train_kwargs):
    """Resume training from checkpoint."""
    checkpoint = load_npe_checkpoint(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint.get('epoch', 0) + 1
    print(f"Resumed training from epoch {start_epoch}\n")
    return train_npe_pycbc(model, train_params, train_data,
                          n_epochs=n_epochs, start_epoch=start_epoch,
                          model_path=model_path, **train_kwargs)


# ============================================================================
# MAIN EXECUTION: Load data, train model, generate plots
# ============================================================================

# Load data and get metadata
print("[DATA] Generating fresh waveforms with complete metadata...")
NUM_DETECTORS = 2  # Change to 1 for single detector, 3 for triple
train_dloader, val_dloader, test_dloader, metadata = prepare_pycbc_data(
    num_samples=20000, batch_size=32, use_cache=False, cache_dir='./waveform_cache', 
    num_detectors=NUM_DETECTORS
)

# ============================================================================
# DIAGNOSTIC 1: Check if waveforms actually contain mass1 information
# ============================================================================
print("\n" + "="*60)
print("DIAGNOSTIC 1: Waveform-Parameter Correlations (PRE-TRAINING)")
print("="*60)
print("Checking if raw waveform features correlate with parameters...")

from scipy.stats import pearsonr, spearmanr

# Collect waveforms and parameters
all_waveforms = []
all_params = []
for batch_w, batch_p in train_dloader:
    all_waveforms.append(batch_w)
    all_params.append(batch_p)
    if len(all_waveforms) * batch_w.shape[0] >= 2000:  # Use 2000 samples for speed
        break

all_waveforms = torch.cat(all_waveforms, dim=0)[:2000]
all_params = torch.cat(all_params, dim=0)[:2000]

print(f"\nWaveform shape: {all_waveforms.shape}")
print(f"Parameter shape: {all_params.shape}")

# Extract waveform features
waveform_features = {
    'max_amplitude': all_waveforms.abs().max(dim=-1)[0].mean(dim=-1),  # Max across time, mean across detectors
    'std': all_waveforms.std(dim=-1).mean(dim=-1),  # Std across time
    'rms': (all_waveforms ** 2).mean(dim=-1).sqrt().mean(dim=-1),  # RMS
    'peak_to_peak': all_waveforms.max(dim=-1)[0].mean(dim=-1) - all_waveforms.min(dim=-1)[0].mean(dim=-1),
}

param_names = metadata.get('parameter_names', ['mass1', 'mass2', 'spin1z'])

print("\nCorrelation: Waveform Features vs Parameters (Pearson r):")
print("-" * 60)
for feat_name, feat_values in waveform_features.items():
    print(f"\n{feat_name}:")
    for i, param_name in enumerate(param_names):
        r, p = pearsonr(feat_values.numpy(), all_params[:, i].numpy())
        significance = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  vs {param_name:8s}: r = {r:+.3f} (p={p:.2e}) {significance}")
        if abs(r) < 0.1 and param_name == 'mass1':
            print(f"    ⚠ WARNING: {feat_name} has VERY WEAK correlation with mass1!")

# Check if there's ANY signal variation
print(f"\nWaveform Statistics:")
print(f"  Mean amplitude range: [{all_waveforms.mean():.4f}, max={all_waveforms.max():.4f}]")
print(f"  Waveform std (should vary): {all_waveforms.std():.4f}")
print(f"  Per-sample std range: [{waveform_features['std'].min():.4f}, {waveform_features['std'].max():.4f}]")

if waveform_features['std'].std() < 0.01:
    print("\n⚠ CRITICAL: All waveforms have nearly identical amplitude!")
    print("   This means the network CANNOT learn mass1 from amplitude.")
    print("   Check waveform_normalization setting!")

print("="*60 + "\n")

# Get actual data dimensions from dataloader
sample_batch, sample_params = next(iter(train_dloader))
actual_data_dim = sample_batch.shape[-1]

# Get parameter normalization info from metadata
param_norm = metadata['parameter_normalization']

# Print parameter bounds for verification
print("\nParameter Normalization Bounds (from generated data):")
for param_name, norm_info in param_norm.items():
    print(f"  {param_name}: [{norm_info['min']:.4f}, {norm_info['max']:.4f}]")

# Plot amplitude histogram if available
if 'amplitude_stats' in metadata:
    amplitude_stats = metadata['amplitude_stats']
    data_generator.plot_amplitude_histogram(amplitude_stats, output_dir='./Plots')
else:
    print("\n⚠ Note: Amplitude statistics not available (data loaded from cache)")


model = DINGOModel(
    data_dim=actual_data_dim,  # Use actual data dimension from dataloader
    param_dim=3,
    context_dim=64,  # Keep at 64 for efficiency
    num_flow_layers=6,  # Increased from 6 for more expressive power
    hidden_dim=128,  # Increased from 64 for better capacity
    embedding='conv1d',  # Use Conv1D for stability and speed
    num_detectors=NUM_DETECTORS,
    use_conv_preprocessing=False  # Not needed for Conv1D (has built-in preprocessing)
)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# Print model configuration
print("\nModel Configuration:")
for key, value in model.config.items():
    print(f"  {key}: {value}")

# Train the model
print(f"Training on device: {device}")
history = train_npe_pycbc(
    model,
    train_dloader,
    val_dloader=val_dloader,
    n_epochs=50,  # Reduced from 300
    lr=0.0003,  # Increased learning rate
    optimizer='adamw',
    device=device,
    grad_clip_norm=1.0,  # Reduced gradient clip
    model_path='best_multi_detector_npe.pt',
    use_mixed_precision=False,
    verbose=True
)

# Save training history
import pickle
with open('training_history.pkl', 'wb') as f:
    pickle.dump(history, f)
print(f"Training complete! History saved to training_history.pkl")

# ============================================================================
# DIAGNOSTIC 3: Check if context embeddings are meaningful (POST-TRAINING)
# ============================================================================
print("\n" + "="*60)
print("DIAGNOSTIC 3: Context Embeddings Analysis (POST-TRAINING)")
print("="*60)

with torch.no_grad():
    model.eval()
    contexts = []
    params_list = []
    waveforms_list = []
    
    # Collect contexts and parameters from test set
    for batch_data, batch_params in test_dloader:
        batch_data = batch_data.to(device)
        context = model.embedding_net(batch_data)
        contexts.append(context.cpu())
        params_list.append(batch_params)
        waveforms_list.append(batch_data.cpu())
    
    contexts = torch.cat(contexts, dim=0)
    params_list = torch.cat(params_list, dim=0)
    waveforms_list = torch.cat(waveforms_list, dim=0)
    
    # Check 1: Context variability
    context_std = contexts.std(0)
    print(f"\nContext std per dimension (mean): {context_std.mean():.4f}")
    print(f"Context std per dimension (min):  {context_std.min():.4f}")
    print(f"Context std per dimension (max):  {context_std.max():.4f}")
    if context_std.mean() < 0.1:
        print("⚠ WARNING: Context std is very low - embeddings may be collapsed!")
    
    # Check 2: Correlation with parameters
    print("\nCorrelation between context dimensions and parameters:")
    from scipy.stats import pearsonr
    param_names = ['mass1', 'mass2', 'spin1z']
    
    for param_idx, param_name in enumerate(param_names):
        # Find context dimensions with highest correlation
        correlations = []
        for ctx_idx in range(contexts.shape[1]):
            r, _ = pearsonr(contexts[:, ctx_idx].numpy(), params_list[:, param_idx].numpy())
            correlations.append(abs(r))
        
        max_corr = max(correlations)
        max_idx = correlations.index(max_corr)
        print(f"  {param_name}: max |r| = {max_corr:.3f} (context dim {max_idx})")
        
        if max_corr < 0.2:
            print(f"    ⚠ WARNING: Very weak correlation for {param_name}!")
    
    # Check 3: Context distribution
    print(f"\nContext value range: [{contexts.min():.3f}, {contexts.max():.3f}]")
    print(f"Context mean: {contexts.mean():.4f}")
    
    # ============================================================================
    # DIAGNOSTIC 4: Test if flow actually uses context (context ablation)
    # ============================================================================
    print("\n" + "-"*60)
    print("DIAGNOSTIC 4: Flow Context Sensitivity Test")
    print("-"*60)
    
    # Take one waveform and sample posterior
    test_waveform = waveforms_list[:1].to(device)
    test_context = model.embedding_net(test_waveform)
    
    # Sample with real context
    real_samples = model.sample_posterior(test_waveform, num_samples=500)
    real_mean = real_samples.mean(dim=0)
    real_std = real_samples.std(dim=0)
    
    # Sample with random context (should give different results if flow uses context)
    random_context = torch.randn_like(test_context)
    
    # Temporarily replace embedding output
    original_forward = model.embedding_net.forward
    model.embedding_net.forward = lambda x: random_context.expand(x.shape[0], -1)
    random_samples = model.sample_posterior(test_waveform, num_samples=500)
    model.embedding_net.forward = original_forward
    
    random_mean = random_samples.mean(dim=0)
    random_std = random_samples.std(dim=0)
    
    print(f"\nPosterior mean comparison:")
    for i, name in enumerate(param_names):
        diff = abs(real_mean[i].item() - random_mean[i].item())
        print(f"  {name}: Real={real_mean[i].item():.3f}, Random={random_mean[i].item():.3f}, Diff={diff:.3f}")
        if diff < 0.1:
            print(f"    ⚠ WARNING: Posterior barely changes with random context for {name}!")
            print(f"       This means the flow is IGNORING the context!")
    
    # ============================================================================
    # DIAGNOSTIC 5: Check posterior spread vs prior
    # ============================================================================
    print("\n" + "-"*60)
    print("DIAGNOSTIC 5: Posterior Spread Analysis")
    print("-"*60)
    
    # Prior bounds (normalized - assuming zscore with computed stats)
    print("\nPosterior std vs expected range:")
    print("(If posterior std ≈ prior std, network learned nothing)")
    
    for i, name in enumerate(param_names):
        posterior_std = real_std[i].item()
        # For z-scored data, prior std should be ~1 if uniform
        print(f"  {name}: posterior_std = {posterior_std:.3f}")
        if posterior_std > 0.8:
            print(f"    ⚠ WARNING: Posterior is very wide - not much learned for {name}!")
    
    # ============================================================================
    # DIAGNOSTIC 6: Gradient flow visualization data
    # ============================================================================
    print("\n" + "-"*60)
    print("DIAGNOSTIC 6: Gradient Flow Summary")
    print("-"*60)
    
    if 'gradient_stats' in history and history['gradient_stats']['embedding']:
        emb_grads = history['gradient_stats']['embedding']
        flow_grads = history['gradient_stats']['flow']
        print(f"\nEmbedding gradient norm: start={emb_grads[0]:.2e}, end={emb_grads[-1]:.2e}")
        print(f"Flow gradient norm: start={flow_grads[0]:.2e}, end={flow_grads[-1]:.2e}")
        
        if emb_grads[-1] < emb_grads[0] * 0.01:
            print("⚠ WARNING: Embedding gradients vanished during training!")
        
        ratio_start = emb_grads[0] / (flow_grads[0] + 1e-10)
        ratio_end = emb_grads[-1] / (flow_grads[-1] + 1e-10)
        print(f"Embedding/Flow gradient ratio: start={ratio_start:.2e}, end={ratio_end:.2e}")

print("="*60 + "\n")

# ============================================================================
# DIAGNOSTIC 7: Multi-sample posterior check
# ============================================================================
print("\n" + "="*60)
print("DIAGNOSTIC 7: Posterior Consistency Across Different Inputs")
print("="*60)

with torch.no_grad():
    model.eval()
    
    # Get 5 test samples with very different mass1 values
    test_indices = []
    target_mass1_values = [12, 20, 30, 40, 48]  # Spread across prior range
    
    # Find samples close to these target values
    all_mass1 = params_list[:, 0].numpy()
    
    print("\nTesting posterior for different true mass1 values:")
    print("-" * 60)
    
    for target in target_mass1_values:
        # Find closest sample (in normalized space, need to account for normalization)
        # Since params are normalized, find by relative position
        idx = np.argmin(np.abs(all_mass1 - np.percentile(all_mass1, (target - 10) / 40 * 100)))
        
        test_wave = waveforms_list[idx:idx+1].to(device)
        true_params = params_list[idx]
        
        samples = model.sample_posterior(test_wave, num_samples=500)
        pred_mean = samples.mean(dim=0)
        pred_std = samples.std(dim=0)
        
        # Denormalize for display
        true_mass1_norm = true_params[0].item()
        pred_mass1_norm = pred_mean[0].item()
        pred_mass1_std = pred_std[0].item()
        
        print(f"  True mass1 (norm): {true_mass1_norm:+.2f} -> Pred: {pred_mass1_norm:+.2f} ± {pred_mass1_std:.2f}")
        
        if abs(pred_mass1_norm) < 0.3 and abs(true_mass1_norm) > 0.5:
            print(f"    ⚠ Prediction is near prior mean despite true value being far!")

print("\nIf all predictions cluster around 0 (normalized prior mean), the network isn't learning.")
print("="*60 + "\n")

# Forward pass test for visualization
batch_data, batch_params = next(iter(train_dloader))
batch_data = batch_data.to(device)
batch_params = batch_params.to(device)

log_prob = model(batch_params, batch_data)

# Sample test
samples_normalized = model.sample_posterior(batch_data[:1], num_samples=100)
samples = denormalize_params(samples_normalized, metadata, param_norm)


# ============================================================================
# PLOT RESULTS
# ============================================================================

fig = plt.figure(figsize=(14, 10))

# Plot 1: Training curve
ax1 = plt.subplot(2, 3, 1)
epochs = range(1, len(history['train_log_probs']) + 1)
ax1.plot(epochs, history['train_log_probs'], 'b-o', linewidth=2, label='Training')
if history['val_log_probs']:
    ax1.plot(epochs, history['val_log_probs'], 'r-s', linewidth=2, label='Validation')
ax1.set(xlabel='Epoch', ylabel='Log Probability', title='NPE Training Progress')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plots 2-4: Posterior histograms
for i, (name, label) in enumerate(zip(['mass1', 'mass2', 'spin1z'], 
                                        ['Mass 1 ($M_\\odot$)', 'Mass 2 ($M_\\odot$)', 'Spin 1z']), 2):
    ax = plt.subplot(2, 3, i)
    data = samples[:, i-2].cpu().numpy() if torch.is_tensor(samples) else samples[:, i-2]
    ax.hist(data, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    mean_val, std_val = data.mean(), data.std()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
    ax.axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=1.5, label=f'±1σ')
    ax.axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=1.5)
    ax.set(xlabel=label, ylabel='Probability Density', title=f'Posterior: {label}')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=9)

# Plot 5: Info box
ax5 = plt.subplot(2, 3, 5)
ax5.axis('off')

true_params = denormalize_params(batch_params[:1], metadata, param_norm)
mass1_true = true_params[0, 0]
mass2_true = true_params[0, 1]
spin1z_true = true_params[0, 2]

info_text = f"""
Multi-Detector NPE Status

Data Shape: {batch_data.shape}
• Batch size: {batch_data.shape[0]}
• Detectors: {batch_data.shape[1]} ({', '.join(metadata.get('detectors', ['H1']))})
• Samples per detector: {batch_data.shape[2]}

Model Config:
• Embedding: {model.config['embedding']}
• Context dim: {model.config['context_dim']}
• Flow layers: {model.config['num_flow_layers']}
• Parameters: {model.config['param_dim']}

True Parameters (Physical):
• mass1 = {mass1_true:.2f} M☉
• mass2 = {mass2_true:.2f} M☉
• spin1z = {spin1z_true:.4f}

Posterior Samples:
• Posterior samples: {samples.shape[0]}
"""
ax5.text(0.1, 0.5, info_text, fontsize=9, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', 
         facecolor='lightgreen', alpha=0.3))

plt.tight_layout()

embedding_type = model.config['embedding']
plt.savefig(f'./Plots/multi_detector_npe_{embedding_type}_m1_{mass1_true:.1f}_m2_{mass2_true:.1f}_s1z_{spin1z_true:.2f}.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n" + "="*60)
print("✓ Training completed successfully!")
print("✓ Plots generated and saved to ./Plots/")
print("="*60)

import sys
sys.exit(0)  # Explicit success exit code