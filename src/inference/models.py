# models.py

import torch
import torch.nn as nn

class MotionFeatureNet(nn.Module):
    def __init__(self, input_dim, num_classes, config):
        super(MotionFeatureNet, self).__init__()
        
        hidden_dims = config['model']['hidden_dims']
        dropout_rate = config['model']['dropout_rate']
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Final classification layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
