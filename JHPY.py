"""
Welcome to John-Hamza py.
This is a library containing the neural networks we are using for our project.
A lot of this is written by Claude.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist
import itertools
import random
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm

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
    
    # Vectorized sampling - generate all parameters at once
    frequencies = np.random.uniform(freq_low, freq_high, num_simulations)
    phases = np.random.uniform(phase_low, phase_high, num_simulations)
    amplitudes = np.random.uniform(amplitude_low, amplitude_high, num_simulations)
    
    # Create time array
    t = np.linspace(0, 6*np.pi, num_points)
    
    # Vectorized signal generation: shape [num_simulations, num_points]
    # Broadcasting: frequencies, phases, amplitudes are [num_simulations, 1] or [num_simulations]
    # t is [num_points]
    signal = amplitudes[:, np.newaxis] * np.sin(
        2*np.pi*frequencies[:, np.newaxis] * t + phases[:, np.newaxis]
    )
    
    # Generate noise for all signals at once
    noise = np.random.normal(0, noise_std, (num_simulations, num_points))
    
    # Add noise to signals
    X = torch.FloatTensor(signal + noise)  # [num_simulations, num_points]

    # Stack parameters as labels: [num_simulations, 3] with columns [amplitude, frequency, phase]
    y = torch.FloatTensor(np.column_stack([amplitudes, frequencies, phases]))  # [num_simulations, 3]

    # Convert to tensors for return
    frequencies = torch.FloatTensor(frequencies).unsqueeze(1)  # [N, 1]
    phases = torch.FloatTensor(phases).unsqueeze(1)  # [N, 1]
    amplitudes = torch.FloatTensor(amplitudes).unsqueeze(1)  # [N, 1]

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

        # Create mask (which dimensions to transform)
        # For 2D: 'even' = [1,0] (transform 2nd), 'odd' = [0,1] (transform 1st)
        # For 3D: 'even' = [1,0,1] (transform 2nd), 'odd' = [0,1,0] (transform 1st & 3rd)
        self.register_buffer('mask', torch.zeros(dim))
        if mask_type in ['half', 'even']:
            self.mask[::2] = 1  # Set every other dimension starting from 0: [1,0,1,0,...]
        elif mask_type == 'odd':
            self.mask[1::2] = 1  # Set every other dimension starting from 1: [0,1,0,1,...]

        # Networks for scale (s) and translation (t) - EXACT from working notebook
        self.scale_net = nn.Sequential(
            nn.Linear(dim + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
            nn.Tanh()  # Stabilize training
        )

        self.translation_net = nn.Sequential(
            nn.Linear(dim + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
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

################### Model Classes ###################

class ParameterPredictor(nn.Module):
    """
    LSTM-based neural network for predicting scalar parameters from time series.

    Configurable model with LSTM layers followed by fully connected layers.
    Config options: lstm_hidden_size (256), lstm_num_layers (1), fc_layer_sizes ([128, 64]),
    activation ('silu'/'relu'/'tanh'), dropout (0.0).
    """
    def __init__(self, config=None):
        """
        Initialize model with optional config overrides.

        Args:
            config (dict, optional): Config dict with lstm_hidden_size, lstm_num_layers, fc_layer_sizes, activation, dropout.
        """
        super().__init__()

        # Default configuration
        default_config = {
            'lstm_hidden_size': 256,
            'lstm_num_layers': 1,
            'fc_layer_sizes': [128, 64],  # Sizes of fully connected layers before output
            'activation': 'silu',  # 'silu', 'relu', 'tanh'
            'dropout': 0.0,  # Dropout probability
        }

        # Merge with provided config
        if config is None:
            config = {}
        self.config = {**default_config, **config}

        # Build LSTM
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=self.config['lstm_hidden_size'],
            num_layers=self.config['lstm_num_layers'],
            batch_first=True,
            dropout=self.config['dropout'] if self.config['lstm_num_layers'] > 1 else 0.0
        )

        # Build fully connected layers
        fc_layers = []
        input_size = self.config['lstm_hidden_size']

        for hidden_size in self.config['fc_layer_sizes']:
            fc_layers.append(nn.Linear(input_size, hidden_size))

            # Add activation
            if self.config['activation'] == 'silu':
                fc_layers.append(nn.SiLU())
            elif self.config['activation'] == 'relu':
                fc_layers.append(nn.ReLU())
            elif self.config['activation'] == 'tanh':
                fc_layers.append(nn.Tanh())

            # Add dropout if specified
            if self.config['dropout'] > 0:
                fc_layers.append(nn.Dropout(self.config['dropout']))

            input_size = hidden_size

        # Output layer (3 parameters: amplitude, frequency, phase)
        fc_layers.append(nn.Linear(input_size, 3))

        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        """
        Forward pass: process time series through LSTM and FC layers.

        Args:
            x (torch.Tensor): Input shape [batch, sequence_length]

        Returns:
            torch.Tensor: Output shape [batch, 3] with predictions for amplitude, frequency, phase
        """
        # Reshape input to [batch, sequence, features]
        x = x.unsqueeze(-1)  # Add feature dimension: [batch, sequence_length, 1]
        lstm_out, _ = self.lstm(x)
        # Use last output for prediction
        last_out = lstm_out[:, -1, :]
        return self.fc(last_out)

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

        # Repeat context for multiple samples
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
    """
    def __init__(self, data_dim=100, param_dim=1, context_dim=64,
                 num_flow_layers=6, hidden_dim=128, config=None):
        super().__init__()

        # Support both positional args and config dict
        if config is not None:
            data_dim = config.get('data_dim', data_dim)
            param_dim = config.get('param_dim', param_dim)
            context_dim = config.get('context_dim', context_dim)
            num_flow_layers = config.get('num_flow_layers', num_flow_layers)
            hidden_dim = config.get('hidden_dim', hidden_dim)

        # Store config for checkpointing
        self.config = {
            'data_dim': data_dim,
            'param_dim': param_dim,
            'context_dim': context_dim,
            'num_flow_layers': num_flow_layers,
            'hidden_dim': hidden_dim
        }

        self.embedding_net = EmbeddingNetwork(
            data_dim=data_dim,
            context_dim=context_dim,
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

################### Training Functions ###################
    
def train_predictor_model(model, optimizer, loss_fcn, n_epochs, train_dloader, val_dloader, start_epoch=0, patience=8, scheduler=None, save_best_model=False, model_path='best_predictor_model.pt', grad_clip_norm=5.0, dropout_rate=None):
    """
    Train model with early stopping, validation monitoring, gradient clipping, and optional checkpointing.

    Args:
        model (nn.Module): PyTorch model to train
        optimizer (torch.optim.Optimizer): Optimizer for training
        loss_fcn (callable): Loss function
        n_epochs (int): Number of epochs to train
        train_dloader (DataLoader): Training data loader
        val_dloader (DataLoader): Validation data loader
        start_epoch (int, optional): Starting epoch for resume. Defaults to 0.
        patience (int, optional): Epochs to wait before early stopping. Defaults to 8.
        scheduler (torch.optim.lr_scheduler, optional): LR scheduler. Defaults to None.
        save_best_model (bool, optional): Save best model checkpoint. Defaults to False.
        model_path (str, optional): Path for checkpoint. Defaults to 'best_predictor_model.pt'.
        grad_clip_norm (float, optional): Max gradient norm for clipping. Set to None to disable. Defaults to 5.0.
        dropout_rate (float, optional): Dropout probability (0.0-1.0). None uses model's default. Defaults to None.

    Returns:
        dict: Contains 'train_losses', 'val_losses', 'train_metrics', 'val_metrics', 'best_val_loss', 'best_val_epoch'.
    """
    train_losses, val_losses = [], []
    train_metrics, val_metrics = [], []
    best_val_loss = float('inf')
    best_val_epoch = 0

    # Set dropout rate if specified
    if dropout_rate is not None:
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = dropout_rate

    for epoch in range(start_epoch, start_epoch + n_epochs):
        model.train()
        tloss, vloss = 0, 0
        train_predictions = []
        train_targets = []

        for X_train, y_train in tqdm(train_dloader, desc='Epoch {}, training'.format(epoch+1)):
            optimizer.zero_grad()
            pred = model(X_train)
            loss = loss_fcn(pred, y_train)
            tloss += loss.item()
            loss.backward()

            # Gradient clipping
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

            optimizer.step()

            train_predictions.extend(pred.detach().numpy())
            train_targets.extend(y_train.numpy())

            
        model.eval()
        vloss = 0
        val_predictions = []
        val_targets = []

        with torch.no_grad():
            for X_valid, y_valid in tqdm(val_dloader, desc='Epoch {}, validation'.format(epoch+1)):
                pred = model(X_valid)
                loss = loss_fcn(pred, y_valid)
                vloss += loss.item()
                val_predictions.extend(pred.numpy())
                val_targets.extend(y_valid.numpy())

        # Calculate metrics
        train_metrics_dict = calculate_metrics(
            np.array(train_predictions), 
            np.array(train_targets)
        )
        val_metrics_dict = calculate_metrics(
            np.array(val_predictions), 
            np.array(val_targets)
        )

        # Store losses
        avg_train_loss = tloss / len(train_dloader)
        avg_val_loss = vloss / len(val_dloader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Store metrics
        train_metrics.append(train_metrics_dict)
        val_metrics.append(val_metrics_dict)

        # Print epoch results
        print(f"\n[Epoch {epoch+1:2d}]")
        print(f"Training - Loss: {avg_train_loss:.4f}, MAE: {train_metrics_dict['mae']:.4f}, "
              f"RMSE: {train_metrics_dict['rmse']:.4f}, R²: {train_metrics_dict['r2']:.4f}")
        print(f"Validation - Loss: {avg_val_loss:.4f}, MAE: {val_metrics_dict['mae']:.4f}, "
              f"RMSE: {val_metrics_dict['rmse']:.4f}, R²: {val_metrics_dict['r2']:.4f} \n")

        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step(avg_val_loss)
            current_lr = [param_group['lr'] for param_group in optimizer.param_groups]
            print(f"Current learning rates: {current_lr}")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            print("New best validation performance \n")
            best_val_loss = avg_val_loss
            best_val_epoch = epoch
            
            # Save the best model
            if save_best_model:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'model_config': model.config if hasattr(model, 'config') else None,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics
                }
                torch.save(checkpoint, model_path)
                print(f"Model checkpoint saved to {model_path}\n")
                
        elif best_val_epoch <= epoch - patience:
            print(f'No improvement in validation loss in last {patience} epochs \n')
            break

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'best_val_loss': best_val_loss,
        'best_val_epoch': best_val_epoch
    }
    
def train_npe_model(model, optimizer, n_epochs, train_dloader, val_dloader, start_epoch=0, patience=15, scheduler=None, save_best_model=False, model_path='best_npe_model.pt', grad_clip_norm=5.0, dropout_rate=None):
    """
    Train NPE model with log probability, early stopping, validation monitoring, and optional checkpointing.

    Args:
        model (nn.Module): PyTorch model to train
        optimizer (torch.optim.Optimizer): Optimizer for training
        n_epochs (int): Number of epochs to train
        train_dloader (DataLoader): Training data loader
        val_dloader (DataLoader): Validation data loader
        start_epoch (int, optional): Starting epoch for resume. Defaults to 0.
        patience (int, optional): Epochs to wait for improvement before early stopping. Defaults to 8.
        scheduler (torch.optim.lr_scheduler, optional): LR scheduler. Defaults to None.
        save_best_model (bool, optional): Save best model checkpoint. Defaults to False.
        model_path (str, optional): Path for checkpoint. Defaults to 'best_model.pt'.
        grad_clip_norm (float, optional): Max gradient norm for clipping. Set to None to disable. Defaults to 5.0.
        dropout_rate (float, optional): Dropout probability (0.0-1.0). None uses model's default. Defaults to None.

    Returns:
        dict: Contains 'train_log_probs', 'val_log_probs', 'best_val_log_prob', 'best_val_epoch'.
    """
    train_log_probs = []
    val_log_probs = []
    best_val_log_prob = float('-inf')  # Higher is better for log probability
    best_val_epoch = 0

    # Set dropout rate if specified
    if dropout_rate is not None:
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = dropout_rate

    for epoch in range(start_epoch, start_epoch + n_epochs):
        model.train()
        train_log_prob_sum = 0

        for X_train, y_train in tqdm(train_dloader, desc='Epoch {}, training'.format(epoch+1)):
            optimizer.zero_grad()
            log_prob = model(y_train, X_train)
            loss = -log_prob.mean()  # Negative log prob for gradient descent
            
            loss.backward()
            
            # Gradient clipping
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            
            optimizer.step()
            
            train_log_prob_sum += log_prob.mean().item()

        model.eval()
        val_log_prob_sum = 0

        with torch.no_grad():
            for X_valid, y_valid in tqdm(val_dloader, desc='Epoch {}, validation'.format(epoch+1)):
                log_prob = model(y_valid, X_valid)
                val_log_prob_sum += log_prob.mean().item()

        # Compute averages
        avg_train_log_prob = train_log_prob_sum / len(train_dloader)
        avg_val_log_prob = val_log_prob_sum / len(val_dloader)
        train_log_probs.append(avg_train_log_prob)
        val_log_probs.append(avg_val_log_prob)

        # Print epoch results
        print(f"\n[Epoch {epoch+1:2d}]")
        print(f"Training - Log Prob: {avg_train_log_prob:.4f}")
        print(f"Validation - Log Prob: {avg_val_log_prob:.4f}")

        # Learning rate scheduling (inverted: higher log prob is better)
        if scheduler is not None:
            scheduler.step(avg_val_log_prob)
            current_lr = [param_group['lr'] for param_group in optimizer.param_groups]
            print(f"Current learning rates: {current_lr}")

        # Early stopping check (higher log prob is better)
        if avg_val_log_prob > best_val_log_prob:
            print("New best validation performance \n")
            best_val_log_prob = avg_val_log_prob
            best_val_epoch = epoch

            # Save the best model
            if save_best_model:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_log_prob': best_val_log_prob,
                    'model_config': model.config if hasattr(model, 'config') else None,
                    'train_log_probs': train_log_probs,
                    'val_log_probs': val_log_probs
                }
                torch.save(checkpoint, model_path)
                print(f"Model checkpoint saved to {model_path}\n")

        elif best_val_epoch <= epoch - patience:
            print(f'No improvement in validation log prob in last {patience} epochs \n')
            break

    return {
        'train_log_probs': train_log_probs,
        'val_log_probs': val_log_probs,
        'best_val_log_prob': best_val_log_prob,
        'best_val_epoch': best_val_epoch
    }
    
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
    
    # R² score
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }

def load_predictor(model_path='best_predictor_model.pt'):
    """
    Load saved ParameterPredictor checkpoint.

    Args:
        model_path (str, optional): Path to checkpoint. Defaults to 'best_predictor_model.pt'.

    Returns:
        tuple: (model, checkpoint dict)
    """
    checkpoint = torch.load(model_path, weights_only=False)

    # Recreate model with saved config
    model = ParameterPredictor(checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded ParameterPredictor from {model_path}")
    print(f"  Best epoch: {checkpoint['epoch'] + 1}")
    print(f"  Best validation loss: {checkpoint['best_val_loss']:.4f}")

    return model, checkpoint

def load_npe(model_path='best_npe_model.pt', model_class=DINGOModel):
    """
    Load saved NPE model checkpoint (NormalizingFlow or DINGOModel).

    Args:
        model_path (str, optional): Path to checkpoint. Defaults to 'best_npe_model.pt'.
        model_class: NPE model class (DINGOModel or NormalizingFlow). Defaults to DINGOModel.

    Returns:
        tuple: (model, checkpoint dict)
    """
    checkpoint = torch.load(model_path, weights_only=False)

    # Recreate model with saved config - pass as keyword argument
    model = model_class(config=checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded {model_class.__name__} from {model_path}")
    print(f"  Best epoch: {checkpoint['epoch'] + 1}")
    print(f"  Best validation log prob: {checkpoint['best_val_log_prob']:.4f}")

    return model, checkpoint

def predictor_hyperparameter_search(param_grid, train_loader, val_loader, n_epochs=20, n_trials=None, model_path='best_predictor_model.pt'):
    """
    Search for best hyperparameter configuration.

    Args:
        param_grid: Dict of parameter names to value lists
        train_loader: Training data loader
        val_loader: Validation data loader
        n_epochs: Epochs per configuration. Defaults to 20.
        n_trials: Random trials to try; None = all combinations. Defaults to None.
        model_path: Path to save best model. Defaults to 'best_predictor_model.pt'.

    Returns:
        tuple: (best_config dict, results list)
    """

    results = []
    best_val_loss = float('inf')
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
    
    print(f"Testing {len(all_combinations)} configurations...\n")
    
    for i, combo in enumerate(all_combinations):
        # Create config dictionary
        config = dict(zip(param_names, combo))
        
        print(f"{'='*60}")
        print(f"Trial {i+1}/{len(all_combinations)}")
        print(f"Config: {config}")
        print(f"{'='*60}")
        
        # Create model with this configuration
        model = ParameterPredictor(config)
        
        # Create optimizer and scheduler
        lr = config.get('learning_rate', 0.01)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        )
        
        loss_fcn = nn.MSELoss()
        
        # Train the model
        try:
            outputs = train_predictor_model(
                model,
                optimizer,
                loss_fcn,
                n_epochs,
                train_loader,
                val_loader,
                patience=8,
                scheduler=scheduler,
                save_best_model=False
            )

            # Get best validation loss
            final_val_loss = min(outputs['val_losses'])
            final_val_metrics = outputs['val_metrics'][outputs['val_losses'].index(final_val_loss)]

            # Store results
            result = {
                'config': config.copy(),
                'best_val_loss': final_val_loss,
                'best_val_mae': final_val_metrics['mae'],
                'best_val_rmse': final_val_metrics['rmse'],
                'best_val_r2': final_val_metrics['r2'],
                'n_epochs_trained': len(outputs['val_losses'])
            }
            results.append(result)

            print(f"\nFinal validation loss: {final_val_loss:.4f}")
            print(f"Best validation R²: {final_val_metrics['r2']:.4f}\n")

            # Update best configuration and save best model
            if final_val_loss < best_val_loss:
                best_val_loss = final_val_loss
                best_config = config.copy()
                # Save the best model so far
                checkpoint = {
                    'epoch': len(outputs['val_losses']) - 1,
                    'model_state_dict': model.state_dict(),
                    'best_val_loss': best_val_loss,
                    'model_config': model.config if hasattr(model, 'config') else None,
                    'train_losses': outputs['train_losses'],
                    'val_losses': outputs['val_losses'],
                    'train_metrics': outputs['train_metrics'],
                    'val_metrics': outputs['val_metrics']
                }
                torch.save(checkpoint, model_path)
                print(f"*** New best configuration found! ***\n")
        
        except Exception as e:
            print(f"Error training with config {config}: {e}\n")
            continue
    
    # Print summary
    print(f"\n{'='*60}")
    print("HYPERPARAMETER SEARCH COMPLETE")
    print(f"{'='*60}")
    print(f"\nBest configuration:")
    for key, value in best_config.items():
        print(f"  {key}: {value}")
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    
    # Sort results by validation loss
    results.sort(key=lambda x: x['best_val_loss'])
    
    return best_config, results

def npe_hyperparameter_search(param_grid, train_loader, val_loader, model_class=DINGOModel, n_epochs=20, n_trials=None, model_path='best_npe_model.pt'):
    """
    Hyperparameter search optimized for NPE models (NormalizingFlow, DINGOModel).

    Args:
        param_grid: Dict of parameter names to value lists
        train_loader: Training data loader
        val_loader: Validation data loader
        model_class: NPE model class (DINGOModel or NormalizingFlow). Defaults to DINGOModel.
        n_epochs: Epochs per configuration. Defaults to 20.
        n_trials: Random trials to try; None = all combinations. Defaults to None.
        model_path: Path to save best model. Defaults to 'best_npe_model.pt'.

    Returns:
        tuple: (best_config dict, results list sorted by validation log prob)
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

    print(f"Testing {len(all_combinations)} configurations with {model_class.__name__}...\n")

    for i, combo in enumerate(all_combinations):
        # Create config dictionary
        config = dict(zip(param_names, combo))

        print(f"{'='*60}")
        print(f"Trial {i+1}/{len(all_combinations)}")
        print(f"Config: {config}")
        print(f"{'='*60}")

        # Create model with this configuration
        model = model_class(config=config)

        # Create optimizer and scheduler
        lr = config.get('learning_rate', 0.001)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',  # Higher log prob is better for NPE
            factor=0.5,
            patience=2,
            min_lr=1e-7,
        )

        # Extract grad_clip_norm and dropout_rate if present
        grad_clip_norm = config.get('grad_clip_norm', 5.0)
        dropout_rate = config.get('dropout_rate', None)

        # Train the model
        try:
            outputs = train_npe_model(
                model,
                optimizer,
                n_epochs,
                train_loader,
                val_loader,
                patience=8,
                scheduler=scheduler,
                grad_clip_norm=grad_clip_norm,
                dropout_rate=dropout_rate,
                save_best_model=False
            )

            # Get best validation log prob
            final_val_log_prob = max(outputs['val_log_probs'])

            # Store results
            result = {
                'config': config.copy(),
                'best_val_log_prob': final_val_log_prob,
                'n_epochs_trained': len(outputs['val_log_probs'])
            }
            results.append(result)

            print(f"\nFinal validation log prob: {final_val_log_prob:.4f}\n")

            # Update best configuration and save best model
            if final_val_log_prob > best_val_log_prob:
                best_val_log_prob = final_val_log_prob
                best_config = config.copy()
                # Save the best model so far
                checkpoint = {
                    'epoch': len(outputs['val_log_probs']) - 1,
                    'model_state_dict': model.state_dict(),
                    'best_val_log_prob': best_val_log_prob,
                    'model_config': model.config if hasattr(model, 'config') else None,
                    'train_log_probs': outputs['train_log_probs'],
                    'val_log_probs': outputs['val_log_probs']
                }
                torch.save(checkpoint, model_path)
                print(f"*** New best configuration found! ***\n")

        except Exception as e:
            print(f"Error training with config {config}: {e}\n")
            continue

    # Print summary
    print(f"\n{'='*60}")
    print("NPE HYPERPARAMETER SEARCH COMPLETE")
    print(f"{'='*60}")
    print(f"\nBest configuration:")
    for key, value in best_config.items():
        print(f"  {key}: {value}")
    print(f"\nBest validation log prob: {best_val_log_prob:.4f}")

    # Sort results by log prob (descending)
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
    
    # Prepare data
    data_tensor = torch.FloatTensor(observed_data).unsqueeze(0)  # [1, data_dim]
    
    # Sample from posterior
    with torch.no_grad():
        samples = model.sample_posterior(data_tensor, num_samples=num_samples)
        samples = samples.numpy()  # [num_samples, param_dim]
    
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
