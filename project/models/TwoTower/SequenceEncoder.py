import torch
import torch.nn as nn
import torch.nn.functional as F

class SequenceEncoder(nn.Module):
    def __init__(self, item_count, genre_count, emb_dim=32):
        super().__init__()
        
        # Genre Embedding
        self.genre_emb = nn.Embedding(genre_count, emb_dim, padding_idx=0) 

        # Movie embedding
        self.item_emb = nn.Embedding(item_count, emb_dim, padding_idx=0)

        self.pos_emb = nn.Embedding(20, emb_dim)

        # Self-Attention layer
        self.attention = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=4, batch_first=True)

    def forward(self, history_seq, genre_sep):
        
        # history movie seq: [batch, 20]
        batch_size, seq_len = history_seq.shape
        # --- Embedding Layer ---
        movie_vec = self.item_emb(history_seq)

        # history genre sep: [batch, 20, 3]
        genre_vec_raw = self.genre_emb(genre_sep)

        # Transform to [batch, 20]
        genre_vec_pooled = torch.sum(genre_vec_raw, dim=2)

        position = torch.arange(seq_len, device=history_seq.device).unsqueeze(0)
        pos_vec = self.pos_emb(position)
        
        # Combining
        seq_emb = movie_vec + genre_vec_pooled + pos_vec

        # --- Attention Layer ---
        # Generate Mask
        # key_padding_mask: [batch, 20], True stands for padding location
        mask = (history_seq == 0)

        # Calculate Self-Attention
        # attn_output: [batch, 20, emb_dim]
        attn_output, _ = self.attention(seq_emb, seq_emb, seq_emb, key_padding_mask=mask)

        # Residual connection
        seq_emb = self.layer_norm(seq_emb + attn_output)

        # --- Pooling (Last Valid Item) ---
        valid_lengths = (~mask).long().sum(dim=1)
        valid_lengths = torch.clamp(valid_lengths, min=1)

        last_valid_idx = valid_lengths - 1
        batch_indices = torch.arange(batch_size, device=attn_output.device)

        # [batch, emb_dim]
        final_seq_vec = seq_emb[batch_indices, last_valid_idx, :]
        final_seq_vec = final_seq_vec * (~mask[:, 0].unsqueeze(-1)) # make sure invalid user = 0

        return final_seq_vec