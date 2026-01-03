# The lstm models are currently having issues, I am going to rebuild it in order to try and fix this

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm



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