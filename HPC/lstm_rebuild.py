from JHPY import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from JHPY import *

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

class LSTMEmbeddingNetwork(nn.Module):
    """LSTM-based embedding network for waveform data. Better at capturing temporal dependencies."""
    def __init__(self, context_dim=512, hidden_dim=256, num_layers=2, num_detectors=1):
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


class DINGOModelLSTM(nn.Module):
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
        self.embedding_net = LSTMEmbeddingNetwork(data_dim, context_dim, 256, 2, num_detectors=num_detectors)
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
    


################################## DATA STUFF ######################################


def create_dingolstm_from_data(dataloader_result, param_dim=None, context_dim=64,
                           num_flow_layers=6, hidden_dim=128):
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
    
    print(num_detectors, data_dim)

    if param_dim is None:
        if 'parameter_names' in metadata:
            param_dim = len(metadata['parameter_names'])
        else:
            raise ValueError("Cannot infer param_dim from metadata")
    
    print(f"Creating DINGOModelLSTM: data_dim={data_dim}, param_dim={param_dim}, "
          f"context_dim={context_dim}, num_flow_layers={num_flow_layers}, num_detectors={num_detectors}")

    model = DINGOModelLSTM(data_dim=data_dim, param_dim=param_dim,
                       context_dim=context_dim, num_flow_layers=num_flow_layers,
                       hidden_dim=hidden_dim, num_detectors=num_detectors)
    

    return model

output = load_dataloaders("data_noise.pt")

train_dloader = output["train_loader"]
val_dloader = output["val_loader"]
test_dloader = output["test_loader"]
metadata = output["metadata"]

model = create_dingo_from_data(output)

print(model.config)

optim = torch.optim.Adam(model.parameters(), lr=5e-4)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optim, mode='max', factor=0.5, patience=6
)
training_stuff = train_npe_model(model, optim, 1, train_dloader, val_dloader, patience=15, scheduler=sched, model_path='best_dingo_model_noise.pt')
