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
        # Self-Attention layer
        self.attention = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=4, batch_first=True)

    def forward(self, history_seq, genre_sep):

        # history movie seq: [batch, 20]
        movie_vec = self.item_emb(history_seq)

        # history genre sep: [batch, 20, 3]
        genre_vec_raw = self.genre_emb(genre_sep)
        # Transform to [batch, 20]
        genre_vec_pooled = torch.sum(genre_vec_raw, dim=2)

        
        # [batch, 20, emb_dim]
        seq_emb = movie_vec + genre_vec_pooled

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
        batch_indices = torch.arange(attn_output.size(0), device=attn_output.device)

        # [batch, emb_dim]
        final_seq_vec = attn_output[batch_indices, last_valid_idx, :]

        return final_seq_vec