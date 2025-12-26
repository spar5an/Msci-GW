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
        self.scale_net = nn.Sequential(
            nn.Linear(dim + context_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, dim), nn.Tanh()
        )
        # Translation network: outputs additive shifts
        self.translation_net = nn.Sequential(
            nn.Linear(dim + context_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x, context, reverse=False):
        masked_x = x * self.mask  # Keep masked dimensions fixed
        # Compute scale and translation from masked input + context
        s = self.scale_net(torch.cat([masked_x, context], dim=1)) * (1 - self.mask)
        t = self.translation_net(torch.cat([masked_x, context], dim=1)) * (1 - self.mask)
        if not reverse:  # Forward: data -> latent
            y = x * torch.exp(s) + t  # Affine transformation
            log_det = s.sum(dim=1)  # Log determinant of Jacobian
        else:  # Reverse: latent -> data
            y = (x - t) * torch.exp(-s)  # Inverse transformation
            log_det = -s.sum(dim=1)  # Negative log det for inverse
        return y, log_det


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
            scale_net = nn.Sequential(
                nn.Linear(latent_dim + total_context_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.0),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.0),
                nn.Linear(hidden_dim, latent_dim),
                nn.Tanh()
            )
            
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


class EmbeddingNetwork(nn.Module):
    """Neural network to embed observed data into context vectors."""
    def __init__(self, data_dim=100, context_dim=64, hidden_dim=128, num_detectors = 1):
        super().__init__()
        self.num_detectors = num_detectors

        # Separate embeddings for each detector
        self.network = nn.ModuleList([
            nn.Sequential(
                nn.Linear(data_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, context_dim)
            )
            for _ in range(num_detectors)
        ])
 
    def forward(self, data):
        #if len(data.shape) == 3 and data.shape[1] == 1:
        #    data = data.squeeze(1)
        #return self.network(data)
    
        # data shape: [batch, num_detectors, data_dim] or [batch, data_dim]
        if len(data.shape) == 2:
            # Single detector case
            return self.network[0](data)
        
        #batch_size = data.shape[0]
        contexts = []
        for i in range(self.num_detectors):
            detector_data = data[:, i, :]  # Extract data for detector i
            context = self.network[i](detector_data)  # Embed detector data
            contexts.append(context)
        return torch.stack(contexts, dim=1)  # Return [batch, num_detectors, context_dim]
    
class Conv1DEmbeddingNetwork(nn.Module):
    """Conv1D-based embedding network for waveform data. Efficient for long sequences."""
    def __init__(self, data_dim=5868, context_dim=512, num_filters=[64, 128, 256], num_detectors=1):
        super().__init__()
        self.num_detectors = num_detectors


        self.conv_pipelines = nn.ModuleList()
        for _ in range(num_detectors):
            pipeline = nn.Sequential(
                nn.Conv1d(1, num_filters[0], kernel_size=15, stride=2, padding=7),
                nn.BatchNorm1d(num_filters[0]), nn.ReLU(), nn.MaxPool1d(2, 2),
                nn.Conv1d(num_filters[0], num_filters[1], kernel_size=15, stride=2, padding=7),
                nn.BatchNorm1d(num_filters[1]), nn.ReLU(), nn.MaxPool1d(2, 2),
                nn.Conv1d(num_filters[1], num_filters[2], kernel_size=15, stride=2, padding=7),
                nn.BatchNorm1d(num_filters[2]), nn.ReLU(), nn.MaxPool1d(2, 2),
                nn.AdaptiveAvgPool1d(1),  # Pool to single value per filter
                nn.Flatten(),
                nn.Linear(num_filters[2], 512), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(512, context_dim), nn.LayerNorm(context_dim)  # Normalize output
            )
            self.conv_pipelines.append(pipeline)


    def forward(self, data):

        #batch_size = data.shape[0]
        outputs = []

        for i in range(self.num_detectors):
            detector_data = data[:, i, :]  # Extract data for detector i
            detector_data = detector_data.unsqueeze(1)  # Add channel dimension
            context = self.conv_pipelines[i](detector_data)  # Embed detector data
            output = context.squeeze(-1)
            outputs.append(output)
        return torch.stack(outputs, dim=1)  # Return [batch, num_detectors, context_dim]



class LSTMEmbeddingNetwork(nn.Module):
    """LSTM-based embedding network for waveform data. Better at capturing temporal dependencies."""
    def __init__(self, data_dim=7241, context_dim=512, hidden_dim=256, num_layers=2, num_detectors=1):
        super().__init__()
        self.num_detectors = num_detectors
        
        # Create separate LSTM pipelines for each detector
        self.lstm_pipelines = nn.ModuleList()
        for _ in range(num_detectors):
            input_proj = nn.Sequential(nn.Linear(1, 32), nn.ReLU())
            # Bidirectional LSTM captures forward and backward temporal patterns
            lstm = nn.LSTM(32, hidden_dim, num_layers, batch_first=True, 
                          bidirectional=True, dropout=0.1 if num_layers > 1 else 0)
            # Project LSTM outputs to context
            output_proj = nn.Sequential(
                nn.Linear(hidden_dim * 2, 512), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(512, context_dim), nn.LayerNorm(context_dim)
            )
            self.lstm_pipelines.append(nn.Sequential(input_proj, lstm, output_proj))

    def forward(self, data):
        # data shape: [batch, num_detectors, data_dim] or [batch, data_dim]
        if len(data.shape) == 2:
            # Single detector case
            x = data.unsqueeze(-1)  # [batch, data_dim, 1]
            x = self.lstm_pipelines[0][0](x)  # input_proj
            lstm_out, (h_n, c_n) = self.lstm_pipelines[0][1](x)  # lstm
            final_state = torch.cat([h_n[-2, :, :], h_n[-1, :, :]], dim=1)
            return self.lstm_pipelines[0][2](final_state)  # output_proj
        
        # Multi-detector case: [batch, num_detectors, data_dim]
        outputs = []
        for i in range(self.num_detectors):
            detector_data = data[:, i, :]  # [batch, data_dim]
            x = detector_data.unsqueeze(-1)  # [batch, data_dim, 1]
            x = self.lstm_pipelines[i][0](x)  # input_proj
            lstm_out, (h_n, c_n) = self.lstm_pipelines[i][1](x)  # lstm
            final_state = torch.cat([h_n[-2, :, :], h_n[-1, :, :]], dim=1)
            output = self.lstm_pipelines[i][2](final_state)  # output_proj
            outputs.append(output)
        return torch.stack(outputs, dim=1)  # [batch, num_detectors, context_dim]


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

        if num_detectors > 1:
            # Multi-detector affine coupling layers
            self.layers = nn.ModuleList([
                MultiDetectorAffineCouplingLayer(param_dim, context_dim, # Should it be context_dim // num_detectors?
                                                 num_detectors=num_detectors,
                                                 hidden_dim=hidden_dim, 
                                                 mask_type='even' if i % 2 == 0 else 'odd')
                for i in range(num_layers)
            ])
        else:

            # Stack coupling layers with alternating masks
            self.layers = nn.ModuleList([
                AffineCouplingLayer(param_dim, context_dim, hidden_dim, 
                                  'even' if i % 2 == 0 else 'odd')
                for i in range(num_layers)
            ])

    def forward(self, params, context):
        # Transform parameters through all layers
        z, log_det_sum = params, torch.zeros(params.size(0), device=params.device)

        for layer in self.layers:
            if isinstance(layer, MultiDetectorAffineCouplingLayer):
                # Split context for multi-detector layers
                z, log_det = layer(z, context, reverse=False)
            else:
                z, log_det = layer(z, context, reverse=False)
            log_det_sum += log_det
        # Compute log prob under base Gaussian distribution
        log_prob_base = -0.5 * (torch.log(2 * np.pi * self.base_std**2) + 
                                ((z - self.base_mean) / self.base_std)**2).sum(dim=1)
        # Apply change of variables formula
        return log_prob_base + log_det_sum

    def sample(self, context, num_samples=1):
        # Sample from base distribution
        batch_size = context.shape[0] if len(context.shape) >=2 else 1
        z = torch.randn(batch_size * num_samples, self.param_dim, device=context.device)

        if len(context.shape) == 2:
            context_repeated = context.repeat_interleave(num_samples, dim=0)
        elif len(context.shape) == 3:
            # For multi-detector context [batch, num_detectors, context_dim]
            context_repeated = context.repeat_interleave(num_samples, dim=0)

        # Apply inverse transformations through layers
        for layer in reversed(self.layers):            
            z, _ = layer(z, context_repeated, reverse=True)
        return z


class DINGOModel(nn.Module):
    """DINGO neural posterior estimation model. Pipeline: data -> embedding -> flow -> log p(params | data)."""
    def __init__(self, data_dim=100, param_dim=1, context_dim=64, num_flow_layers=6, 
                 hidden_dim=128, embedding='conv1d', config=None, num_detectors=1):
        super().__init__()
        # Parse config if provided
        if config:
            data_dim = config.get('data_dim', data_dim)
            param_dim = config.get('param_dim', param_dim)
            context_dim = config.get('context_dim', context_dim)
            num_flow_layers = config.get('num_flow_layers', num_flow_layers)
            hidden_dim = config.get('hidden_dim', hidden_dim)
            embedding = config.get('embedding', embedding)
        # Store configuration for checkpointing
        self.config = {'data_dim': data_dim, 'param_dim': param_dim, 'context_dim': context_dim,
                      'num_flow_layers': num_flow_layers, 'hidden_dim': hidden_dim, 'embedding': embedding,
                        'num_detectors': num_detectors}
        
        # Select embedding network based on data size and type
        if embedding.lower() == 'lstm' and data_dim > 1000:
            self.embedding_net = LSTMEmbeddingNetwork(data_dim, context_dim, 256, 2, num_detectors=num_detectors)
        elif embedding.lower() == 'conv1d' and data_dim > 1000:
            self.embedding_net = Conv1DEmbeddingNetwork(data_dim, context_dim, num_detectors=num_detectors)
        else:  # Linear MLP for small data
            self.embedding_net = EmbeddingNetwork(data_dim, context_dim, hidden_dim, num_detectors=num_detectors)
        # Normalizing flow for posterior
        self.flow = NormalizingFlow(param_dim, context_dim, num_flow_layers, hidden_dim, num_detectors=num_detectors)

    def forward(self, params, data):  # <- DATA_FLOW [3] WAVEFORMS RECEIVED IN MODEL.FORWARD()
        # Embed data to context, then compute log probability
        context = self.embedding_net(data)  # <- DATA_FLOW [4] EMBEDDING COMPUTES CONTEXT IN MODEL.FORWARD()
        return self.flow(params, context)  # <- DATA_FLOW [5] FLOW RECEIVES CONTEXT IN MODEL.FORWARD()

    def sample_posterior(self, data, num_samples=1000):  # <- DATA_FLOW [6] WAVEFORM DATA RECEIVED IN MODEL.SAMPLE_POSTERIOR (inference path)
        # Sample from posterior p(params | data)
        self.eval()
        with torch.no_grad():
            context = self.embedding_net(data)  # <- DATA_FLOW [7] EMBEDDING COMPUTES CONTEXT
            samples = self.flow.sample(context, num_samples=num_samples)  # <- DATA_FLOW [8] FLOW.SAMPLE() GENERATES SAMPLES
        return samples


def train_npe_pycbc(model, train_dloader, val_dloader=None, n_epochs=100, lr=1e-4, 
                    optimizer='adam', scheduler=None, start_epoch=0, save_best_model=True,
                    model_path='best_npe_pycbc_model.pt', grad_clip_norm=5.0, 
                    use_mixed_precision=True, device='cuda', reg_config=None, verbose=True):
    """Train NPE model on PyCBC data with checkpointing and regularization."""
    model = model.to(device)  # Move model to device
    # Create optimizer based on type
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5) if optimizer.lower() == 'adamw' else \
          torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9) if optimizer.lower() == 'sgd' else \
          torch.optim.Adam(model.parameters(), lr=lr)
    
    # Setup learning rate scheduler if not provided
    if scheduler is None and hasattr(model, 'config'):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=lr * 0.01)
    
    # Mixed precision training for faster computation
    scaler = torch.cuda.amp.GradScaler() if use_mixed_precision and torch.cuda.is_available() else None
    torch.backends.cudnn.benchmark = True  # Enable cuDNN autotuner
    
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
        
        # Iterate over training batches
        for batch_waveforms, batch_params in tqdm(train_dloader, desc=f'Epoch {epoch+1:3d}/{start_epoch + n_epochs}', 
                                                   disable=not verbose):  # <- DATA_FLOW [1] WAVEFORMS LOADED FROM DATALOADER
            batch_waveforms, batch_params = batch_waveforms.to(device), batch_params.to(device)  # <- DATA_FLOW [2] MOVE WAVEFORMS TO DEVICE
            # Linearly increase regularization weight during warmup
            reg_weight = reg_config['max_weight'] * min(epoch + 1, reg_config['warmup_epochs']) / reg_config['warmup_epochs']
            opt.zero_grad()  # Clear gradients
            
            if scaler:  # Mixed precision training
                with torch.cuda.amp.autocast():
                    log_prob = model(batch_params, batch_waveforms)  # <- DATA_FLOW [3] WAVEFORMS PASS TO MODEL.FORWARD()
                    nll_loss = -log_prob.mean()  # Negative log likelihood
                    # Regularization: keep context embeddings expressive
                    context = model.embedding_net(batch_waveforms)  # <- DATA_FLOW [4] COMPUTE REGULARIZATION: EMBED WAVEFORMS AGAIN
                    context_std = context.std(dim=0).mean()  # Average std across dimensions
                    reg_loss = reg_weight * F.relu(reg_config['target_std'] - context_std) ** 2
                    total_loss = nll_loss + reg_loss  # Combined loss
                scaler.scale(total_loss).backward()  # Backward with scaling
                if grad_clip_norm: torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(opt)  # Update weights
                scaler.update()  # Update scaler
            else:  # Standard training
                log_prob = model(batch_params, batch_waveforms)  # <- DATA_FLOW [3] WAVEFORMS PASS TO MODEL.FORWARD()
                nll_loss = -log_prob.mean()
                context = model.embedding_net(batch_waveforms)  # <- DATA_FLOW [4] COMPUTE REGULARIZATION: EMBED WAVEFORMS AGAIN
                context_std = context.std(dim=0).mean()
                reg_loss = reg_weight * F.relu(reg_config['target_std'] - context_std) ** 2
                total_loss = nll_loss + reg_loss
                total_loss.backward()  # Compute gradients
                if grad_clip_norm: torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                opt.step()  # Update weights
            
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
        
        # Validation phase
        val_log_prob = None
        if val_dloader:
            model.eval()  # Set to eval mode
            val_log_prob_sum, val_batch_count = 0, 0
            with torch.no_grad():  # No gradients needed for validation
                for batch_waveforms, batch_params in val_dloader:  # <- DATA_FLOW [6] VALIDATION WAVEFORMS LOADED FROM DATALOADER
                    batch_waveforms, batch_params = batch_waveforms.to(device), batch_params.to(device)  # <- DATA_FLOW [7] MOVE VALIDATION WAVEFORMS TO DEVICE
                    log_prob = model(batch_params, batch_waveforms)  # <- DATA_FLOW [8] VALIDATION WAVEFORMS PASS TO MODEL.FORWARD()
                    val_log_prob_sum += log_prob.mean().item()
                    val_batch_count += 1
            val_log_prob = val_log_prob_sum / val_batch_count
            val_log_probs.append(val_log_prob)
        
        # Log progress
        if verbose:
            if val_log_prob:
                print(f"[Epoch {epoch+1:3d}] Train: {avg_train_log_prob:8.4f}, Val: {val_log_prob:8.4f}")
        
        # Update learning rate
        if scheduler and val_log_prob:
            scheduler.step(val_log_prob)  # Validation-based scheduling
        elif scheduler:
            scheduler.step()  # Epoch-based scheduling
        
        # Save best model checkpoint
        if val_log_prob and val_log_prob > best_val_log_prob and save_best_model:
            best_val_log_prob, best_val_epoch = val_log_prob, epoch
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                       'optimizer_state_dict': opt.state_dict(), 'best_val_log_prob': best_val_log_prob,
                       'model_config': model.config, 'train_log_probs': train_log_probs,
                       'val_log_probs': val_log_probs, 'train_losses': train_losses}, model_path)
    
    if verbose:
        print(f"\nTraining complete!\n  Final train log prob: {train_log_probs[-1]:.4f}\n")
    
    # Return all metrics
    return {'train_log_probs': train_log_probs, 'val_log_probs': val_log_probs, 'train_losses': train_losses,
            'context_stds': context_stds, 'reg_losses': reg_losses, 'best_val_log_prob': best_val_log_prob if val_log_probs else None, 'best_val_epoch': best_val_epoch}


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


def prepare_pycbc_data():
    """Generate and prepare PyCBC gravitational wave data with train/val/test split."""
    config = {'mass1': lambda size: np.random.uniform(10, 50, size=size),
              'mass2': lambda size: np.random.uniform(10, 50, size=size),
              'spin1z': lambda size: np.random.uniform(-0.5, 0.5, size=size)}
    result = data_generator.pycbc_data_generator(config, num_samples=10000, batch_size=16, 
                                                num_workers=4, allow_padding=True,
                                                normalize_waveforms=True, detectors=['H1', 'L1'])
    # Return train/val/test dataloaders
    return result['train_loader'], result['val_loader'], result['test_loader']


train_dloader, val_dloader, test_dloader = prepare_pycbc_data()


def sample_posterior(model, observed_data, num_samples=5000, device='cuda'):  # <- DATA_FLOW [9] OBSERVED WAVEFORMS RECEIVED
    """Sample from posterior p(params | data) using trained NPE model."""
    model.eval()
    model = model.to(device)
    
    # Convert to tensor
    if isinstance(observed_data, np.ndarray):
        data_tensor = torch.FloatTensor(observed_data)  # <- DATA_FLOW [10] WAVEFORMS CONVERTED TO TENSOR
    else:
        data_tensor = observed_data.clone()
    
    # Handle batch dimension
    single_sample = (data_tensor.dim() == 1)
    if single_sample:
        data_tensor = data_tensor.unsqueeze(0)
    
    data_tensor = data_tensor.to(device)  # <- DATA_FLOW [11] WAVEFORM TENSOR MOVED TO DEVICE
    
    # Generate samples
    with torch.no_grad():
        samples = model.sample_posterior(data_tensor, num_samples=num_samples)  # <- DATA_FLOW [12] CALL MODEL.SAMPLE_POSTERIOR()
        samples = samples.cpu().numpy()
    
    # Reshape based on input format
    if single_sample:
        samples = samples.reshape(num_samples, -1)
    else:
        batch_size = len(observed_data)
        param_dim = samples.shape[1]
        samples = samples.reshape(batch_size, num_samples, param_dim)
    
    return samples


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


def create_dingo_from_data(dataloader_result, param_dim=None, context_dim=64,
                           num_flow_layers=6, hidden_dim=128, multi_detector_mode='concatenate'):
    """Create DINGOModel with dimensions automatically inferred from dataloader metadata."""
    metadata = dataloader_result['metadata']
    
    # Infer dimensions from metadata
    if 'waveform_shape' in metadata:
        num_detectors, data_dim = metadata['waveform_shape']
    elif 'channels' in metadata and 'target_length' in metadata:
        num_detectors = len(metadata['channels'])
        data_dim = metadata['target_length']
    else:
        raise ValueError("Cannot infer data dimensions from metadata")

    if param_dim is None:
        if 'parameter_names' in metadata:
            param_dim = len(metadata['parameter_names'])
        else:
            raise ValueError("Cannot infer param_dim from metadata")

    print(f"Creating DINGOModel: data_dim={data_dim}, param_dim={param_dim}, "
          f"context_dim={context_dim}, num_flow_layers={num_flow_layers}")
    
    model = DINGOModel(data_dim=data_dim, param_dim=param_dim,
                       context_dim=context_dim, num_flow_layers=num_flow_layers,
                       hidden_dim=hidden_dim)
    return model


def infer_NPE(model, observed_data, num_samples=5000):  # <- DATA_FLOW [13] OBSERVED WAVEFORMS RECEIVED
    """Perform inference of NPE and return posterior samples with statistics."""
    model.eval()
    data_tensor = torch.FloatTensor(observed_data).unsqueeze(0)  # <- DATA_FLOW [14] WAVEFORMS CONVERTED & BATCHED
    
    with torch.no_grad():
        samples = model.sample_posterior(data_tensor, num_samples=num_samples)  # <- DATA_FLOW [15] CALL MODEL.SAMPLE_POSTERIOR()
        samples = samples.cpu().numpy()
    
    # Compute statistics if single parameter
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


# HYPERPARAMETER SEARCH 
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

#best_config, results = npe_hyperparameter_search(
#    param_grid, 
#    train_dloader, 
#    val_dloader=val_dloader,
#    n_epochs=30,
#    n_trials=2  # search through (500) ALL THE COMBINATIONS!!!!!
#)



# ============================================================================
# TEST MULTI-DETECTOR NPE WITH PYCBC DATA
# ============================================================================

# ============================================================================
# TEST MULTI-DETECTOR NPE WITH PYCBC DATA
# ============================================================================

# Create model with multi-detector support
model = DINGOModel(
    data_dim=6751,
    param_dim=3,
    context_dim=256,
    num_flow_layers=6,
    hidden_dim=128,
    embedding='conv1d',
    num_detectors=2
)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# Train the model
print(f"Training on device: {device}")
history = train_npe_pycbc(
    model,
    train_dloader,
    val_dloader=val_dloader,
    n_epochs=10,
    lr=1e-4,
    optimizer='adamw',
    device=device,
    model_path='best_multi_detector_npe.pt',
    verbose=True
)

# Save training history
import pickle
with open('training_history.pkl', 'wb') as f:
    pickle.dump(history, f)
print(f"Training complete! History saved to training_history.pkl")

# Forward pass test for visualization
batch_data, batch_params = next(iter(train_dloader))
batch_data = batch_data.to(device)
batch_params = batch_params.to(device)

log_prob = model(batch_params, batch_data)

# Sample test
samples = model.sample_posterior(batch_data[:1], num_samples=100)

# ============================================================================
# PLOT RESULTS
# ============================================================================

fig = plt.figure(figsize=(14, 10))

# Plot 1: Log probability (mock training curve)
ax1 = plt.subplot(2, 3, 1)
epochs = range(1, len(history['train_log_probs']) + 1)
ax1.plot(epochs, history['train_log_probs'], 'b-o', linewidth=2, label='Training')
if history['val_log_probs']:
    ax1.plot(epochs, history['val_log_probs'], 'r-s', linewidth=2, label='Validation')
ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('Log Probability', fontsize=11)
ax1.set_title('NPE Training Progress', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2-4: Posterior sample histograms
param_names = ['mass1', 'mass2', 'spin1z']
param_labels = ['Mass 1 ($M_\\odot$)', 'Mass 2 ($M_\\odot$)', 'Spin 1z']

for i, (param_name, param_label) in enumerate(zip(param_names, param_labels), 2):
    ax = plt.subplot(2, 3, i)
    samples_1d = samples[:, i-2].cpu().numpy() if torch.is_tensor(samples) else samples[:, i-2]
    
    # Create histogram
    counts, bins, patches = ax.hist(samples_1d, bins=30, density=True, 
                                     alpha=0.7, color='steelblue', edgecolor='black')
    
    # Add statistics
    mean_val = samples_1d.mean()
    std_val = samples_1d.std()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
    ax.axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=1.5, label=f'±1σ')
    ax.axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=1.5)
    
    ax.set_xlabel(param_label, fontsize=11)
    ax.set_ylabel('Probability Density', fontsize=11)
    ax.set_title(f'Posterior: {param_label}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=9)

# Plot 5: Data shape info
ax5 = plt.subplot(2, 3, 5)
ax5.axis('off')
info_text = f"""
Multi-Detector NPE Status

Data Shape: {batch_data.shape}
• Batch size: {batch_data.shape[0]}
• Detectors: {batch_data.shape[1]} (H1, L1)
• Samples per detector: {batch_data.shape[2]}

Model Config:
• Embedding: Conv1D
• Context dim: 256
• Flow layers: 6 (MultiDetector)
• Parameters: 3

Sampling:
• Posterior samples: {samples.shape[0]}
• Parameter dimensions: {samples.shape[1]}
"""
ax5.text(0.1, 0.5, info_text, fontsize=10, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', 
         facecolor='lightgreen', alpha=0.3))

plt.tight_layout()
plt.savefig('/workspace/multi_detector_npe_results.png', dpi=150, bbox_inches='tight')
plt.close()