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

class SequenceEncoder(nn.Module):
    def __init__(self, item_count, emb_dim=32):
        super().__init__()

        # Item embedding
        self.item_emb = nn.Embedding(item_count, emb_dim, padding_idx=0)

        # Self-Attention layer
        self.attention = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=4, batch_first=True)
    
    def forward(self, history_seq):
        # history_seq: [batch, 20]
        # [batch, 20, emb_dim]
        seq_emb = self.item_emb(history_seq)

        # Generate Mask
        # key_padding_mask: [batch, 20], True stands for padding location
        mask = (history_seq == 0)

        # Calculate Self-Attention
        # attn_output: [batch, 20, emb_dim]
        attn_output, _ = self.attention(seq_emb, seq_emb, seq_emb, key_padding_mask=mask)

        # --- Last Valid Action ---

        valid_lengths = (~mask).long().sum(dim=1)
        valid_lengths = torch.where(valid_lengths == 0, torch.tensor(1, device=mask.device))

        last_valid_idx = valid_lengths - 1
        batch_indices = torch.arrange(attn_output.size(0), device=attn_output.device)

        # [batch, emb_dim]
        final_seq_vec = attn_output[batch_indices, last_valid_idx, :]

        return final_seq_vec
