import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP_Tower(nn.Module):
    """
    MLP's General Structure: Embedding -> MLP -> Normalize
    """
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.1):
        
        super().__init__()
        layers = []
        curr_dim = input_dim

        # Construct multilayer perceptron
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            curr_dim = h_dim
        
        layers.append(nn.Linear(curr_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        out = self.mlp(x)
        # Normalize -> Cosine Similarity
        # x shape: [batch_size, input_total_dim]
        return F.normalize(out, p=2, dim=1)

