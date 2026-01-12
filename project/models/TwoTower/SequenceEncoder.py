import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    def forward(self, x):
        # FFN(x) = max(0, xW1 + b1)W2 + b2
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class SequenceEncoder(nn.Module):
    def __init__(self, item_count: int, genre_count: int, emb_dim=32, dropout = 0.1, max_seq_len=20):

        '''
        :params item_count: Total number of items
        :params genre_count: Total number of genres
        :params emb_dim: Length of Embedding output vector
        :params dropout: Dropout probability rate
        :params max_seq_len: The length of the sequence
        '''
        super().__init__()
        
        # Genre Embedding
        self.genre_emb = nn.Embedding(genre_count, emb_dim, padding_idx=0) 

        # Movie embedding
        self.item_emb = nn.Embedding(item_count, emb_dim, padding_idx=0)

        self.pos_emb = nn.Embedding(max_seq_len, emb_dim)

        # Self-Attention layer
        self.attention = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=4, batch_first=True)
        # Normalize layer
        self.layer_norm = nn.LayerNorm(emb_dim)

        # FFN Layer
        self.ffn = PositionwiseFeedForward(d_model=emb_dim, d_ff=emb_dim * 4, dropout=0.1)
        self.norm2 = nn.LayerNorm(emb_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, history_seq, genre_sep):  
        
        # history movie seq: [batch, 20]
        batch_size, seq_len = history_seq.shape
        # --- Embedding Layer ---
        movie_vec = self.item_emb(history_seq)

        # history genre sep: [batch, 20, 3]
        genre_vec_raw = self.genre_emb(genre_sep)

        # Transform to [batch, 20]
        genre_vec_pooled = torch.mean(genre_vec_raw, dim=2)

        position = torch.arange(seq_len, device=history_seq.device).unsqueeze(0)
        pos_vec = self.pos_emb(position)
        
        # Combining
        seq_emb = movie_vec + genre_vec_pooled + pos_vec

        # --- Attention Layer ---
        # Generate Mask
        # key_padding_mask: [batch, 20], True stands for padding location
        mask = (history_seq == 0)

        all_padded_mask = mask.all(dim=1)
        if all_padded_mask.any():
            mask[all_padded_mask, 0] = False

        # Calculate Self-Attention
        # attn_output: [batch, 20, emb_dim]
        attn_output, _ = self.attention(seq_emb, seq_emb, seq_emb, key_padding_mask=mask)

        # Residual connection
        seq_emb = self.layer_norm(seq_emb + attn_output)

        ffn_output = self.ffn(seq_emb)
        seq_emb = self.norm2(seq_emb + self.dropout(ffn_output))

        # --- Pooling (Last Valid Item) ---
        valid_lengths = (~mask).long().sum(dim=1)
        is_valid_user = (valid_lengths > 0)
        valid_lengths_clamped = torch.clamp(valid_lengths, min=1)

        last_valid_idx = valid_lengths_clamped - 1
        batch_indices = torch.arange(batch_size, device=attn_output.device)

        # [batch, emb_dim]
        final_seq_vec = seq_emb[batch_indices, last_valid_idx, :]
        final_seq_vec = final_seq_vec * is_valid_user.unsqueeze(-1)# make sure invalid user = 0

        return final_seq_vec