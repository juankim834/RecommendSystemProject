import torch
import torch.nn as nn
from Tower import MLP_Tower


class ItemTower(nn.Module):
    def __init__(self, 
                 movie_count: int, year_vocab_size: int, year_emb_dim: int, 
                 genre_count: int, genre_emb_dim: int, 
                 item_emb_dim: int, mlp_hidden_dims: list[int], output_dim: int):
        super().__init__()

        self.id_emb = nn.Embedding(movie_count, item_emb_dim, padding_idx=0)
        self.year_emb = nn.Embedding(year_vocab_size, year_emb_dim)
        self.genre_emb = nn.Embedding(genre_count, genre_emb_dim, padding_idx=0)

        self.total_input_dim = item_emb_dim + year_emb_dim + genre_emb_dim

        self.mlp = MLP_Tower(self.total_input_dim, mlp_hidden_dims, output_dim)
    
    def forward(self, 
                movie_id, year, 
                genre):
        
        """
        Forward pass of the Item Tower.
        
        Generates a dense vector representation for movie items by combining ID embeddings, release year embeddings, and pooled genre embeddings.

        Args:
            movie_id (torch.LongTensor): Tensor of shape (batch_size,) containing unique movie indices.
            year (torch.LongTensor): Tensor of shape (batch_size,) containing encoded release year indices.
            genre (torch.LongTensor): Tensor of shape (batch_size, 3) containing genre indices. Since a movie can have multiple genres, this is handled as a fixed-length list with padding (index 0) where necessary. The embeddings of these genres are aggregated using Mean Pooling.

        Returns:
            torch.FloatTensor: Output tensor of shape (batch_size, output_dim).
            This represents the final embedding of the items.
        """
        
        movie_id_vec = self.id_emb(movie_id)
        year_vec = self.year_emb(year)

        genre_vec = self.genre_emb(genre)
        genre_vec_pooled = torch.mean(genre_vec, dim=1)

        combined_vec = torch.cat([movie_id_vec, year_vec, genre_vec_pooled], dim=1)

        return self.mlp(combined_vec)
    