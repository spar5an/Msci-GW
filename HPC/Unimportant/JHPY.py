"""
Welcome to John-Hamza py.
This is a library containing the neural networks we are using for our project.
A lot of this is written by Claude.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import itertools
import random
import os
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm
import data_generator


################### Miscellaneous functions ###################

def simulate_sine_wave(frequency, num_points=1000, noise_std=0.1, amplitude=1.0, phase=0):
    """
    Generate noisy sine wave with given frequency, amplitude, and phase.

    Args:
        frequency: sine wave frequency
        num_points: number of time points. Defaults to 1000.
        noise_std: Gaussian noise std dev. Defaults to 0.1.
        amplitude: sine amplitude. Defaults to 1.0.
        phase: phase shift. Defaults to 0.

    Returns:
        ndarray: Noisy sine wave observations.
    """
    t = np.linspace(0, 6*np.pi, num_points)
    signal = amplitude * np.sin(2*np.pi*frequency * t + phase)
    noise = np.random.normal(0, noise_std, num_points)
    observed_data = signal + noise
    return observed_data

def generate_sine_data(num_simulations=10000, freq_low=0.5, freq_high=5.0, phase_low=-3, phase_high=3, amplitude_low=0.5, amplitude_high=3.0, num_points=1000, noise_std=0.1, batch_size=256):
    """
    Generate vectorized training dataset with sine signals and DataLoaders.

    Args:
        num_simulations: number of samples. Defaults to 10000.
        freq_low, freq_high: frequency range. Defaults to 0.5, 5.0.
        phase_low, phase_high: phase range. Defaults to -3, 3.
        amplitude_low, amplitude_high: amplitude range. Defaults to 0.5, 3.0.
        num_points: time points per signal. Defaults to 1000.
        noise_std: noise std dev. Defaults to 0.1.
        batch_size: DataLoader batch size. Defaults to 256.

    Returns:
        dict: Amplitudes, Phases, Frequencies tensors and Train/Test/Val_Loader DataLoaders.
    """
    print(f"generating {num_simulations} samples for training")
    
    # Sample parameter ranges
    frequencies = np.random.uniform(freq_low, freq_high, num_simulations)
    phases = np.random.uniform(phase_low, phase_high, num_simulations)
    amplitudes = np.random.uniform(amplitude_low, amplitude_high, num_simulations)

    t = np.linspace(0, 6*np.pi, num_points)

    # Generate signals with broadcasting
    signal = amplitudes[:, np.newaxis] * np.sin(
        2*np.pi*frequencies[:, np.newaxis] * t + phases[:, np.newaxis]
    )

    noise = np.random.normal(0, noise_std, (num_simulations, num_points))
    X = torch.FloatTensor(signal + noise)
    y = torch.FloatTensor(np.column_stack([amplitudes, frequencies, phases]))

    frequencies = torch.FloatTensor(frequencies).unsqueeze(1)
    phases = torch.FloatTensor(phases).unsqueeze(1)
    amplitudes = torch.FloatTensor(amplitudes).unsqueeze(1)

    data = TensorDataset(X, y)  
    
    train_data, val_data, test_data = random_split(data, lengths=[0.8,0.1,0.1])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    print("data generated")
    
    return {"Amplitudes": amplitudes, "Phases": phases, "Frequencies":frequencies,
            "Train_Loader": train_loader, "Test_Loader": test_loader, "Val_Loader": val_loader}


################### Neural Network Layers ###################

class AffineCouplingLayer(nn.Module):
    """
    Affine coupling layer for normalizing flows

    Splits input, transforms one half conditioned on the other:
    x2_new = x2 * exp(s(x1, context)) + t(x1, context)
    """
    def __init__(self, dim, context_dim, hidden_dim=128, mask_type='half'):
        super().__init__()
        self.dim = dim

        # Create alternating mask for coupling layer
        self.register_buffer('mask', torch.zeros(dim))
        if mask_type in ['half', 'even']:
            self.mask[::2] = 1
        elif mask_type == 'odd':
            self.mask[1::2] = 1

        # Scale and translation networks
        self.scale_net = nn.Sequential(
            nn.Linear(dim + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(hidden_dim, dim),
            nn.Tanh()  # Stabilize training
        )

        self.translation_net = nn.Sequential(
            nn.Linear(dim + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.0),
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

        # Compute scale and translation
        s = self.scale_net(scale_input)
        t = self.translation_net(translation_input)

        # Only apply to unmasked dimensions
        s = s * (1 - self.mask)
        t = t * (1 - self.mask)

        if not reverse:
            # Forward: x -> z
            y = x * torch.exp(s) + t
            log_det = s.sum(dim=1)
        else:
            # Reverse: z -> x
            y = (x - t) * torch.exp(-s)
            log_det = -s.sum(dim=1)

        return y, log_det

class EmbeddingNetwork(nn.Module):
    """
    Neural network to embed observed data into context vector.
    Similar to DINGO's data compression network.
    Supports multi-detector data with multiple processing strategies.
    """
    def __init__(self, data_dim=100, context_dim=64, hidden_dim=128, num_detectors=1, multi_detector_mode='concatenate'):
        super().__init__()

        self.num_detectors = num_detectors
        self.multi_detector_mode = multi_detector_mode
        self.data_dim = data_dim
        self.context_dim = context_dim

        if multi_detector_mode == 'concatenate':
            # Concatenate all detector data and process together
            self.network = nn.Sequential(
                nn.Linear(data_dim * num_detectors, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.0),  # Dropout layer (rate set by training function)
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.0),  # Dropout layer (rate set by training function)
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.0),  # Dropout layer (rate set by training function)
                nn.Linear(hidden_dim, context_dim)
            )

        elif multi_detector_mode == 'separate':
            # Process each detector separately, then combine
            self.detector_networks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(data_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.0),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.0),
                    nn.Linear(hidden_dim, context_dim // num_detectors)
                )
                for _ in range(num_detectors)
            ])

            # Combine detector embeddings
            self.combine_network = nn.Sequential(
                nn.Linear(context_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.0),
                nn.Linear(hidden_dim, context_dim)
            )

        elif multi_detector_mode == 'shared':
            # Shared network for all detectors, then combine
            self.shared_network = nn.Sequential(
                nn.Linear(data_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.0),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.0),
                nn.Linear(hidden_dim, context_dim // num_detectors)
            )

            self.combine_network = nn.Sequential(
                nn.Linear(context_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.0),
                nn.Linear(hidden_dim, context_dim)
            )
        else:
            raise ValueError(f"Unknown multi_detector_mode: {multi_detector_mode}")

    def forward(self, data):
        """
        Args:
            data: observed data
                - If num_detectors == 1: [batch_size, data_dim]
                - If num_detectors > 1: [batch_size, num_detectors, data_dim]

        Returns:
            context: embedded representation [batch_size, context_dim]
        """
        if self.num_detectors == 1:
            # Single detector case
            if self.multi_detector_mode == 'concatenate':
                return self.network(data)
            else:
                # For consistency with multi-detector modes
                return self.network(data) if hasattr(self, 'network') else self.shared_network(data)

        # Multi-detector case
        batch_size = data.shape[0]

        if self.multi_detector_mode == 'concatenate':
            # Flatten detectors: [batch, num_detectors, data_dim] -> [batch, num_detectors * data_dim]
            data_flat = data.reshape(batch_size, -1)
            return self.network(data_flat)

        elif self.multi_detector_mode == 'separate':
            # Process each detector separately
            detector_embeddings = []
            for i in range(self.num_detectors):
                embedding = self.detector_networks[i](data[:, i, :])
                detector_embeddings.append(embedding)

            # Concatenate and combine
            combined = torch.cat(detector_embeddings, dim=1)
            return self.combine_network(combined)

        elif self.multi_detector_mode == 'shared':
            # Process each detector with shared weights
            detector_embeddings = []
            for i in range(self.num_detectors):
                embedding = self.shared_network(data[:, i, :])
                detector_embeddings.append(embedding)

            # Concatenate and combine
            combined = torch.cat(detector_embeddings, dim=1)
            return self.combine_network(combined)


class Conv1DEmbeddingNetwork(nn.Module):
    """
    Conv1D-based embedding network for waveform data.
    Efficient for long sequences, good for GW waveforms.
    """
    def __init__(self, data_dim=5868, context_dim=512, num_filters=[64, 128, 256]):
        """
        Initialize Conv1D embedding network.

        Args:
            data_dim (int, optional): Input waveform dimension. Defaults to 5868.
            context_dim (int, optional): Output context dimension. Defaults to 512.
            num_filters (list, optional): Number of filters per conv layer. Defaults to [64, 128, 256].
        """
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
        
        # Global average pooling
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
        Forward pass: embed waveform to context vector.

        Args:
            data (torch.Tensor): Waveform data [batch_size, data_dim]

        Returns:
            torch.Tensor: Context embedding [batch_size, context_dim]
        """
        x = data.unsqueeze(1)  # [batch, 1, data_dim]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_pool(x)  # [batch, filters, 1]
        x = x.view(x.size(0), -1)  # [batch, filters]
        context = self.fc(x)
        return context


class LSTMEmbeddingNetwork(nn.Module):
    """
    LSTM-based embedding network for waveform data.
    Better at capturing temporal dependencies in waveforms.
    """
    def __init__(self, data_dim=7241, context_dim=512, hidden_dim=256, num_layers=2):
        """
        Initialize LSTM embedding network.

        Args:
            data_dim (int, optional): Input waveform dimension. Defaults to 7241.
            context_dim (int, optional): Output context dimension. Defaults to 512.
            hidden_dim (int, optional): LSTM hidden size. Defaults to 256.
            num_layers (int, optional): Number of LSTM layers. Defaults to 2.
        """
        super().__init__()
        self.data_dim = data_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        
        # Initial projection
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
            nn.Linear(hidden_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, context_dim),
            nn.LayerNorm(context_dim)
        )
    
    def forward(self, data):
        """
        Forward pass: embed waveform to context vector.

        Args:
            data (torch.Tensor): Waveform data [batch_size, data_dim]

        Returns:
            torch.Tensor: Context embedding [batch_size, context_dim]
        """
        x = data.unsqueeze(-1)  # [batch, data_dim, 1]
        x = self.input_proj(x)  # [batch, data_dim, 32]
        lstm_out, (h_n, c_n) = self.lstm(x)
        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        final_state = torch.cat([h_forward, h_backward], dim=1)
        context = self.output_proj(final_state)
        return context


################### Model Classes ###################

class NormalizingFlow(nn.Module):
    """
    Normalizing flow: stack of coupling layers
    Transforms base distribution into complex posterior
    """
    def __init__(self, param_dim=1, context_dim=64, num_layers=6, hidden_dim=128, config=None):
        super().__init__()

        # Support both positional args and config dict
        if config is not None:
            param_dim = config.get('param_dim', param_dim)
            context_dim = config.get('context_dim', context_dim)
            num_layers = config.get('num_flow_layers', config.get('num_layers', num_layers))
            hidden_dim = config.get('hidden_dim', hidden_dim)

        # Store config for checkpointing
        self.config = {
            'param_dim': param_dim,
            'context_dim': context_dim,
            'num_flow_layers': num_layers,
            'hidden_dim': hidden_dim
        }

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

        context_repeated = context.repeat_interleave(num_samples, dim=0)

        # Sample from base distribution
        z = torch.randn(batch_size * num_samples, self.param_dim, device=context.device)

        # Apply inverse flow transformations
        for layer in reversed(self.layers):
            z, _ = layer(z, context_repeated, reverse=True)

        return z

class DINGOModel(nn.Module):
    """
    Complete DINGO-style neural posterior estimation model

    Architecture:
    observed_data -> EmbeddingNet -> context -> NormalizingFlow -> log p(params | data)

    Supports multi-detector data with configurable processing strategies.
    """
    def __init__(self, data_dim=100, param_dim=1, context_dim=64, 
                 num_flow_layers=6, hidden_dim=128, embedding='conv1d',
                 config=None):
        """
        Initialize DINGO model.

        Args:
            data_dim (int, optional): Waveform dimension. Defaults to 100.
            param_dim (int, optional): Parameter space dimension. Defaults to 1.
            context_dim (int, optional): Context dimension. Defaults to 64.
            num_flow_layers (int, optional): Number of flow layers. Defaults to 6.
            hidden_dim (int, optional): Hidden layer size. Defaults to 128.
            embedding (str, optional): Embedding type: 'conv1d', 'lstm', 'linear'. Defaults to 'conv1d'.
            config (dict, optional): Config dict for flexibility. Defaults to None.
        """
        super().__init__()
        
        if config is not None:
            data_dim = config.get('data_dim', data_dim)
            param_dim = config.get('param_dim', param_dim)
            context_dim = config.get('context_dim', context_dim)
            num_flow_layers = config.get('num_flow_layers', num_flow_layers)
            hidden_dim = config.get('hidden_dim', hidden_dim)
            embedding = config.get('embedding', embedding)
        
        # Store config for checkpointing
        self.config = {
            'data_dim': data_dim,
            'param_dim': param_dim,
            'context_dim': context_dim,
            'num_flow_layers': num_flow_layers,
            'hidden_dim': hidden_dim,
            'embedding': embedding
        }
        
        # Choose embedding architecture
        if embedding.lower() == 'lstm' and data_dim > 1000:
            self.embedding_net = LSTMEmbeddingNetwork(
                data_dim=data_dim, context_dim=context_dim,
                hidden_dim=256, num_layers=2
            )
        elif embedding.lower() == 'conv1d' and data_dim > 1000:
            self.embedding_net = Conv1DEmbeddingNetwork(
                data_dim=data_dim, context_dim=context_dim,
                num_filters=[64, 128, 256]
            )
        elif embedding.lower() == 'linear':
            # Simple linear MLP embedding
            self.embedding_net = EmbeddingNetwork(
                data_dim=data_dim, context_dim=context_dim,
                hidden_dim=hidden_dim
            )
        else:  # fallback: default linear MLP
            self.embedding_net = EmbeddingNetwork(
                data_dim=data_dim, context_dim=context_dim,
                hidden_dim=hidden_dim
            )
        
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
            data: observed data
                - Single detector: [batch_size, data_dim]
                - Multi-detector: [batch_size, num_detectors, data_dim]

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
            data: observed data
                - Single detector: [batch_size, data_dim]
                - Multi-detector: [batch_size, num_detectors, data_dim]
            num_samples: number of samples to draw

        Returns:
            samples: posterior samples [batch_size * num_samples, param_dim]
        """
        self.eval()
        with torch.no_grad():
            context = self.embedding_net(data)
            samples = self.flow.sample(context, num_samples=num_samples)
        return samples

################### Training Functions ###################

def train_npe_pycbc(model, train_dloader, val_dloader=None,
                    n_epochs=100, lr=1e-4, optimizer='adam', 
                    scheduler=None, start_epoch=0, save_best_model=True, 
                    model_path='best_npe_pycbc_model.pt', grad_clip_norm=5.0, 
                    use_mixed_precision=True, device='cuda', reg_config=None,
                    verbose=True):
    """
    Train NPE model on PyCBC gravitational wave data with checkpointing.

    Args:
        model (nn.Module): DINGO NPE model to train
        train_dloader (DataLoader): Training data loader yielding (params, data) tuples
        val_dloader (DataLoader, optional): Validation data loader. Defaults to None.
        n_epochs (int, optional): Number of training epochs. Defaults to 100.
        lr (float, optional): Learning rate. Defaults to 1e-4.
        optimizer (str, optional): Optimizer type ('adam', 'adamw', 'sgd'). Defaults to 'adam'.
        scheduler (torch.optim.lr_scheduler, optional): LR scheduler. Defaults to None.
        start_epoch (int, optional): Starting epoch for resume. Defaults to 0.
        save_best_model (bool, optional): Save model checkpoint. Defaults to True.
        model_path (str, optional): Path for checkpoint. Defaults to 'best_npe_pycbc_model.pt'.
        grad_clip_norm (float, optional): Max gradient norm for clipping. Set to None to disable. Defaults to 5.0.
        use_mixed_precision (bool, optional): Use mixed precision training. Defaults to True.
        device (str, optional): Device ('cuda' or 'cpu'). Defaults to 'cuda'.
        reg_config (dict, optional): Regularization config with keys: 'target_std', 'max_weight', 'warmup_epochs'.
        verbose (bool, optional): Print progress. Defaults to True.

    Returns:
        dict: Contains 'train_log_probs', 'train_losses', 'context_stds', 'reg_losses', 'val_log_probs', 'best_val_log_prob', 'best_val_epoch'.
    """
    
    # Move model to device
    model = model.to(device)
    
    # Setup optimizer
    if optimizer.lower() == 'adamw':
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    elif optimizer.lower() == 'sgd':
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:  # default to Adam
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Setup learning rate scheduler if provided
    if scheduler is None and hasattr(model, 'config'):
        # Default: cosine annealing if no scheduler specified
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=n_epochs, eta_min=lr * 0.01
        )
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if use_mixed_precision and torch.cuda.is_available() else None
    torch.backends.cudnn.benchmark = True
    
    # Regularization configuration
    if reg_config is None:
        reg_config = {'target_std': 0.8, 'max_weight': 1.0, 'warmup_epochs': 15}
    
    reg_target_std = reg_config.get('target_std', 0.8)
    reg_max_weight = reg_config.get('max_weight', 1.0)
    reg_warmup_epochs = reg_config.get('warmup_epochs', 15)
    
    # Training variables
    train_log_probs = []
    val_log_probs = []
    train_losses = []
    context_stds = []
    reg_losses = []
    
    best_val_log_prob = float('-inf')
    best_val_epoch = 0
    
    if verbose:
        print(f"\nTraining NPE Model on PyCBC Data")
        print(f"  Learning rate: {lr}")
        print(f"  Optimizer: {optimizer}")
        print(f"  Gradient clipping: {grad_clip_norm if grad_clip_norm else 'disabled'}")
        print(f"  Regularization: target_std={reg_target_std}, max_weight={reg_max_weight}, warmup={reg_warmup_epochs} epochs")
        print(f"  Validation: {'Yes' if val_dloader else 'No'}\n")
    
    for epoch in range(start_epoch, start_epoch + n_epochs):
        # ============= TRAINING PHASE =============
        model.train()
        train_log_prob_sum = 0
        train_loss_sum = 0
        context_std_sum = 0
        reg_loss_sum = 0
        batch_count = 0
        
        batch_iterator = tqdm(
            train_dloader,
            desc=f'Epoch {epoch+1:3d}/{start_epoch + n_epochs}, training',
            disable=not verbose
        )
        
        for batch_params, batch_data in batch_iterator:
            batch_params = batch_params.to(device)
            batch_data = batch_data.to(device)
            
            # Compute regularization weight (linear warmup)
            reg_weight = reg_max_weight * min(epoch + 1, reg_warmup_epochs) / reg_warmup_epochs
            
            opt.zero_grad()
            
            if scaler is not None:
                # Mixed precision training
                with torch.cuda.amp.autocast():
                    log_prob = model(batch_params, batch_data)
                    nll_loss = -log_prob.mean()
                    
                    # Regularization: keep context embeddings expressive
                    context = model.embedding_net(batch_data)
                    context_std = context.std(dim=0).mean()
                    reg_loss = reg_weight * F.relu(reg_target_std - context_std) ** 2
                    
                    total_loss = nll_loss + reg_loss
                
                scaler.scale(total_loss).backward()
                
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                
                scaler.step(opt)
                scaler.update()
            else:
                # Standard training (CPU or mixed precision disabled)
                log_prob = model(batch_params, batch_data)
                nll_loss = -log_prob.mean()
                
                context = model.embedding_net(batch_data)
                context_std = context.std(dim=0).mean()
                reg_loss = reg_weight * F.relu(reg_target_std - context_std) ** 2
                
                total_loss = nll_loss + reg_loss
                
                total_loss.backward()
                
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                
                opt.step()
            
            # Accumulate metrics
            train_log_prob_sum += log_prob.mean().item()
            train_loss_sum += total_loss.item()
            context_std_sum += context_std.item()
            reg_loss_sum += reg_loss.item()
            batch_count += 1
        
        # Compute epoch averages
        avg_train_log_prob = train_log_prob_sum / batch_count
        avg_train_loss = train_loss_sum / batch_count
        avg_context_std = context_std_sum / batch_count
        avg_reg_loss = reg_loss_sum / batch_count
        
        train_log_probs.append(avg_train_log_prob)
        train_losses.append(avg_train_loss)
        context_stds.append(avg_context_std)
        reg_losses.append(avg_reg_loss)
        
        # ============= VALIDATION PHASE =============
        val_log_prob = None
        if val_dloader is not None:
            model.eval()
            val_log_prob_sum = 0
            val_batch_count = 0
            
            with torch.no_grad():
                for batch_params, batch_data in val_dloader:
                    batch_params = batch_params.to(device)
                    batch_data = batch_data.to(device)
                    
                    log_prob = model(batch_params, batch_data)
                    val_log_prob_sum += log_prob.mean().item()
                    val_batch_count += 1
            
            val_log_prob = val_log_prob_sum / val_batch_count
            val_log_probs.append(val_log_prob)
        
        # ============= LOGGING & CHECKPOINTING =============
        if verbose:
            if val_log_prob is not None:
                print(f"[Epoch {epoch+1:3d}] Train LogProb: {avg_train_log_prob:8.4f}, Loss: {avg_train_loss:8.4f}, "
                      f"CtxStd: {avg_context_std:6.4f}, RegLoss: {avg_reg_loss:8.6f} | "
                      f"Val LogProb: {val_log_prob:8.4f}")
            else:
                print(f"[Epoch {epoch+1:3d}] LogProb: {avg_train_log_prob:8.4f}, Loss: {avg_train_loss:8.4f}, "
                      f"CtxStd: {avg_context_std:6.4f}, RegLoss: {avg_reg_loss:8.6f}")
        
        # Learning rate scheduling
        if scheduler is not None:
            if hasattr(scheduler, 'step'):
                if val_log_prob is not None:
                    scheduler.step(val_log_prob)
                else:
                    scheduler.step()
        
        # Early stopping and checkpointing with validation
        if val_log_prob is not None:
            if val_log_prob > best_val_log_prob:
                if verbose:
                    print("  New best validation performance\n")
                best_val_log_prob = val_log_prob
                best_val_epoch = epoch
                
                if save_best_model:
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        'best_val_log_prob': best_val_log_prob,
                        'model_config': model.config if hasattr(model, 'config') else None,
                        'train_log_probs': train_log_probs,
                        'val_log_probs': val_log_probs,
                        'train_losses': train_losses
                    }
                    torch.save(checkpoint, model_path)
                    if verbose:
                        print(f"  Checkpoint saved to {model_path}\n")
        else:
            # Save checkpoint periodically if no validation
            if save_best_model and (epoch + 1) % 10 == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'model_config': model.config if hasattr(model, 'config') else None,
                    'train_log_probs': train_log_probs,
                    'train_losses': train_losses
                }
                torch.save(checkpoint, model_path)
                if verbose:
                    print(f"  Checkpoint saved to {model_path}\n")
    
    if verbose:
        print(f"\nTraining complete!")
        print(f"  Final training log prob: {train_log_probs[-1]:.4f}")
        if val_log_probs:
            print(f"  Best validation log prob: {best_val_log_prob:.4f}")
            print(f"  Best epoch: {best_val_epoch + 1}\n")
    
    return {
        'train_log_probs': train_log_probs,
        'val_log_probs': val_log_probs,
        'train_losses': train_losses,
        'context_stds': context_stds,
        'reg_losses': reg_losses,
        'best_val_log_prob': best_val_log_prob if val_log_probs else None,
        'best_val_epoch': best_val_epoch
    }

################### Inference Function ###################


def sample_posterior(model, observed_data, num_samples=5000, device='cuda'):
    """
    Sample from posterior p(params | data) using trained NPE model.

    Args:
        model (nn.Module): Trained NPE model
        observed_data (np.ndarray or torch.Tensor): Observed waveform data [data_dim] or [batch_size, data_dim]
        num_samples (int, optional): Number of posterior samples. Defaults to 5000.
        device (str, optional): Device ('cuda' or 'cpu'). Defaults to 'cuda'.

    Returns:
        np.ndarray: Posterior samples [batch_size, num_samples, param_dim] or [num_samples, param_dim] if single input.
    """
    model.eval()
    model = model.to(device)
    
    # Convert to tensor if needed
    if isinstance(observed_data, np.ndarray):
        data_tensor = torch.FloatTensor(observed_data)
    else:
        data_tensor = observed_data.clone()
    
    # Handle batch dimension
    single_sample = (data_tensor.dim() == 1)
    if single_sample:
        data_tensor = data_tensor.unsqueeze(0)
    
    data_tensor = data_tensor.to(device)
    
    with torch.no_grad():
        samples = model.sample_posterior(data_tensor, num_samples=num_samples)
        samples = samples.cpu().numpy()
    
    # Reshape to match input format
    if single_sample:
        # Return [num_samples, param_dim] for single input
        samples = samples.reshape(num_samples, -1)
    else:
        # Return [batch_size, num_samples, param_dim] for batch input
        batch_size = len(observed_data)
        param_dim = samples.shape[1]
        samples = samples.reshape(batch_size, num_samples, param_dim)
    
    return samples


################### Other Neural Network Functions ###################

def calculate_metrics(predictions, targets):
    """
    Calculate MAE, RMSE, and R² metrics.

    Args:
        predictions (np.ndarray): Model predictions
        targets (np.ndarray): Target values

    Returns:
        dict: Contains 'mae', 'rmse', 'r2'.
    """
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }



def load_npe_checkpoint(model_path='best_npe_pycbc_model.pt'):
    """
    Load saved NPE model checkpoint.

    Args:
        model_path (str, optional): Path to checkpoint. Defaults to 'best_npe_pycbc_model.pt'.

    Returns:
        dict: Checkpoint containing model_state_dict, optimizer_state_dict, metrics, and config.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found at {model_path}")
    
    checkpoint = torch.load(model_path, weights_only=False)
    return checkpoint


def resume_npe_training(model, optimizer, model_path='best_npe_pycbc_model.pt', 
                        train_params=None, train_data=None,
                        n_epochs=50, **train_kwargs):
    """
    Resume training from checkpoint.

    Args:
        model (nn.Module): Model to train
        optimizer (torch.optim.Optimizer): Optimizer
        model_path (str, optional): Path to checkpoint. Defaults to 'best_npe_pycbc_model.pt'.
        train_params, train_data: Training data
        n_epochs (int, optional): Additional epochs to train. Defaults to 50.
        **train_kwargs: Additional arguments for train_npe_pycbc()

    Returns:
        dict: Training results from continued training.
    """
    checkpoint = load_npe_checkpoint(model_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint.get('epoch', 0) + 1
    
    print(f"Resumed training from epoch {start_epoch}\n")
    
    return train_npe_pycbc(
        model, train_params, train_data,
        n_epochs=n_epochs,
        start_epoch=start_epoch,
        model_path=model_path,
        **train_kwargs
    )



def create_dingo_from_data(dataloader_result, param_dim=None, context_dim=64,
                           num_flow_layers=6, hidden_dim=128, multi_detector_mode='concatenate'):
    """
    Create a DINGOModel with dimensions automatically inferred from dataloader metadata.

    Args:
        dataloader_result: Result dict from pycbc_data_generator or load_dataloaders
        param_dim: Number of parameters to infer. If None, inferred from metadata
        context_dim: Context dimension. Defaults to 64
        num_flow_layers: Number of flow layers. Defaults to 6
        hidden_dim: Hidden dimension. Defaults to 128
        multi_detector_mode: 'concatenate', 'separate', or 'shared'. Defaults to 'concatenate'

    Returns:
        DINGOModel: Model configured with correct dimensions

    Examples:
        >>> data = load_dataloaders('my_data.pt')
        >>> model = create_dingo_from_data(data, context_dim=128, num_flow_layers=8)
    """
    metadata = dataloader_result['metadata']

    # Infer dimensions from metadata
    if 'waveform_shape' in metadata:
        # Format: (num_detectors, time_length)
        num_detectors, data_dim = metadata['waveform_shape']
    elif 'channels' in metadata and 'target_length' in metadata:
        num_detectors = len(metadata['channels'])
        data_dim = metadata['target_length']
    else:
        raise ValueError("Cannot infer data dimensions from metadata. Missing 'waveform_shape' or 'channels'/'target_length'")

    if param_dim is None:
        if 'parameter_names' in metadata:
            param_dim = len(metadata['parameter_names'])
        else:
            raise ValueError("Cannot infer param_dim from metadata. Please specify explicitly.")

    print(f"Creating DINGOModel with inferred dimensions:")
    print(f"  data_dim (time length per detector): {data_dim}")
    print(f"  num_detectors: {num_detectors}")
    print(f"  param_dim: {param_dim}")
    print(f"  context_dim: {context_dim}")
    print(f"  num_flow_layers: {num_flow_layers}")
    print(f"  multi_detector_mode: {multi_detector_mode}")

    model = DINGOModel(
        data_dim=data_dim,
        param_dim=param_dim,
        context_dim=context_dim,
        num_flow_layers=num_flow_layers,
        hidden_dim=hidden_dim
    )

    return model


def npe_hyperparameter_search(param_grid, train_dloader, val_dloader=None, model_class=DINGOModel, 
                               n_epochs=20, n_trials=None, model_path='best_npe_model.pt', 
                               device='cuda', use_pycbc=True, config_save_path='best_config.json'):
    """
    Hyperparameter search optimized for NPE models (DINGOModel with DataLoaders).

    Args:
        param_grid: Dict of parameter names to value lists. 
            Can include: 'lr', 'grad_clip_norm', 'embedding', 'context_dim', 'num_flow_layers'
        train_dloader: Training DataLoader yielding (params, data) tuples
        val_dloader: Validation DataLoader (optional for early stopping). Defaults to None.
        model_class: NPE model class. Defaults to DINGOModel.
        n_epochs: Epochs per configuration. Defaults to 20.
        n_trials: Random trials to try; None = all combinations. Defaults to None.
        model_path: Path to save best model. Defaults to 'best_npe_model.pt'.
        device: Device ('cuda' or 'cpu'). Defaults to 'cuda'.
        use_pycbc: Use train_npe_pycbc (True) or train_npe_model (False). Defaults to True.
        config_save_path: Path to save best configuration as JSON. Defaults to 'best_config.json'.

    Returns:
        tuple: (best_config dict, results list sorted by validation log prob)
    
    Example:
        >>> param_grid = {
        ...     'lr': [1e-4, 5e-4, 1e-3],
        ...     'embedding': ['linear', 'conv1d'],
        ...     'context_dim': [64, 128],
        ...     'num_flow_layers': [4, 6]
        ... }
        >>> best_config, results = npe_hyperparameter_search(param_grid, train_loader, val_loader)
    """
    results = []
    best_val_log_prob = float('-inf')
    best_config = None

    # Generate all combinations or sample randomly
    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]

    if n_trials is None:
        # Try all combinations (grid search)
        all_combinations = list(itertools.product(*param_values))
    else:
        # Random search: sample n_trials random combinations
        all_combinations = []
        for _ in range(n_trials):
            combo = tuple(random.choice(values) for values in param_values)
            all_combinations.append(combo)

    print(f"\nTesting {len(all_combinations)} configurations with {model_class.__name__}...")
    print(f"Training mode: {'train_npe_pycbc' if use_pycbc else 'train_npe_model'}\n")

    for i, combo in enumerate(all_combinations):
        config = dict(zip(param_names, combo))

        print(f"{'='*60}")
        print(f"Trial {i+1}/{len(all_combinations)}")
        print(f"Config: {config}")
        print(f"{'='*70}")

        try:
            # Extract hyperparameters from config
            lr = config.pop('lr', 1e-4)
            grad_clip_norm = config.pop('grad_clip_norm', 5.0)
            embedding = config.pop('embedding', 'linear')
            context_dim = config.pop('context_dim', 64)
            num_flow_layers = config.pop('num_flow_layers', 6)
            hidden_dim = config.pop('hidden_dim', 128)
            
            # Extract regularization parameters if provided
            reg_target_std = config.pop('reg_target_std', 0.8)
            reg_max_weight = config.pop('reg_max_weight', 1.0)
            reg_warmup_epochs = config.pop('reg_warmup_epochs', 15)
            reg_config = {
                'target_std': reg_target_std,
                'max_weight': reg_max_weight,
                'warmup_epochs': reg_warmup_epochs
            }
            
            # Create model with remaining config parameters
            model = model_class(
                context_dim=context_dim,
                num_flow_layers=num_flow_layers,
                hidden_dim=hidden_dim,
                embedding=embedding
            )
            
            # Setup optimizer and scheduler
            optimizer_type = config.pop('optimizer', 'adam').lower()
            if optimizer_type == 'adamw':
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
            elif optimizer_type == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            
            # Optional learning rate scheduler
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=n_epochs, eta_min=lr * 0.01
            ) if val_dloader else None

            if use_pycbc:
                # Use train_npe_pycbc (more modern, with DataLoaders)
                outputs = train_npe_pycbc(
                    model,
                    train_dloader,
                    val_dloader=val_dloader,
                    n_epochs=n_epochs,
                    lr=lr,
                    optimizer=optimizer_type,
                    scheduler=scheduler,
                    grad_clip_norm=grad_clip_norm,
                    device=device,
                    reg_config=reg_config,
                    save_best_model=False,
                    verbose=False
                )
                
                if outputs['val_log_probs']:
                    final_val_log_prob = max(outputs['val_log_probs'])
                else:
                    # If no validation, use final training log prob
                    final_val_log_prob = outputs['train_log_probs'][-1]
                
                result = {
                    'config': config.copy(),
                    'best_val_log_prob': final_val_log_prob,
                    'n_epochs_trained': len(outputs['train_log_probs']),
                    'final_train_log_prob': outputs['train_log_probs'][-1]
                }
            else:
                # Alternative: use training log prob if no validation available
                outputs = train_npe_pycbc(
                    model,
                    train_dloader,
                    val_dloader=None,
                    n_epochs=n_epochs,
                    lr=lr,
                    optimizer=optimizer_type,
                    scheduler=None,
                    grad_clip_norm=grad_clip_norm,
                    device=device,
                    reg_config=reg_config,
                    save_best_model=False,
                    verbose=False
                )
                
                final_val_log_prob = outputs['train_log_probs'][-1]
                
                result = {
                    'config': config.copy(),
                    'best_val_log_prob': final_val_log_prob,
                    'n_epochs_trained': len(outputs['train_log_probs']),
                    'final_train_log_prob': outputs['train_log_probs'][-1]
                }
            
            results.append(result)
            
            print(f"Final validation log prob: {final_val_log_prob:.4f}")
            print(f"Final training log prob: {result['final_train_log_prob']:.4f}\n")

            if final_val_log_prob > best_val_log_prob:
                best_val_log_prob = final_val_log_prob
                best_config = {'lr': lr, 'grad_clip_norm': grad_clip_norm, 'embedding': embedding,
                              'context_dim': context_dim, 'num_flow_layers': num_flow_layers,
                              'hidden_dim': hidden_dim, 'reg_target_std': reg_target_std,
                              'reg_max_weight': reg_max_weight, 'reg_warmup_epochs': reg_warmup_epochs,
                              **config}
                
                # Save best model checkpoint
                checkpoint = {
                    'epoch': result['n_epochs_trained'] - 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_log_prob': best_val_log_prob,
                    'model_config': model.config if hasattr(model, 'config') else None,
                    'train_log_probs': outputs['train_log_probs'],
                    'val_log_probs': outputs.get('val_log_probs', [])
                }
                torch.save(checkpoint, model_path)
                print(f"New best configuration found!\n")

        except Exception as e:
            print(f"✗ Error training with config {config}: {e}\n")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*60}")
    print("NPE HYPERPARAMETER SEARCH COMPLETE")
    print(f"{'='*70}\n")
    
    if best_config:
        print(f"Best configuration:")
        for key, value in best_config.items():
            print(f"  {key}: {value}")
        print(f"\nBest validation log prob: {best_val_log_prob:.4f}\n")
        
        # Save best config to JSON file
        import json
        config_to_save = best_config.copy()
        with open(config_save_path, 'w') as f:
            json.dump(config_to_save, f, indent=2)
        print(f"Best configuration saved to {config_save_path}\n")
    else:
        print("No successful trials completed.\n")

    results.sort(key=lambda x: x['best_val_log_prob'], reverse=True)

    return best_config, results

def infer_NPE(model, observed_data, num_samples=5000):
    """
    Perform inference of NPE
    
    Args:
        model: trained model
        observed_data: observed sine wave [data_dim]
        num_samples: number of posterior samples
    
    Returns:
        samples: posterior samples [num_samples, param_dim]
        statistics: dict with mean, median, std, quantiles
    """
    model.eval()

    data_tensor = torch.FloatTensor(observed_data).unsqueeze(0)

    with torch.no_grad():
        samples = model.sample_posterior(data_tensor, num_samples=num_samples)
        samples = samples.numpy()

    if samples.shape[1] == 1:
        samples = samples.flatten()
        statistics = {
            'mean': np.mean(samples),
            'median': np.median(samples),
            'std': np.std(samples),
            'q05': np.percentile(samples, 5),
            'q95': np.percentile(samples, 95),
        }
    else:
        statistics = None
    
    return samples, statistics

def prepare_pycbc_data():
    config = {
        'mass1': lambda size: np.random.uniform(10, 50, size=size),
        'mass2': lambda size: np.random.uniform(10, 50, size=size),
        'spin1z': lambda size: np.random.uniform(-0.5, 0.5, size=size),
    }

    # Generate with H1 only
    result = data_generator.pycbc_data_generator(
        config, 
        num_samples=10000, 
        batch_size=16, 
        num_workers=4,
        allow_padding=True,
        normalize_waveforms=True,
        detectors=['H1']
    )

    # Access the loaders
    train_loader = result['train_loader']
    val_loader = result['val_loader']
    test_loader = result['test_loader']

    return train_loader, val_loader, test_loader

train_dloader, val_dloader, test_dloader = prepare_pycbc_data()


param_grid = {
    'lr': [1e-4, 5e-4, 1e-3],
    'embedding': ['linear', 'conv1d'],
    'context_dim': [64, 128, 256, 512],
    'num_flow_layers': [4, 6, 8, 10, 12, 14, 16],
    'hidden_dim': [128, 256, 512],
    'optimizer': ['adam'],
    'grad_clip_norm': [1.0, 5.0, 10.0],
    'reg_target_std': [0.5, 0.8, 1.0],
    'reg_max_weight': [0.5, 1.0, 2.0],
    'reg_warmup_epochs': [10, 15, 20]
}

best_config, results = npe_hyperparameter_search(
    param_grid, 
    train_dloader, 
    val_dloader=val_dloader,
    n_epochs=30,
    n_trials=500  # search through (500) ALL THE COMBINATIONS!!!!!
)