import torch
import torch.nn as nn
import torch.nn.functional as F
from project.models.TwoTower.SequenceEncoder import SequenceEncoder
from project.models.TwoTower.Tower import MLP_Tower

class UserTower(nn.Module):
    def __init__(self, 
                 # Sequence parameters
                 item_count: int, genre_count: int, seq_emb_dim: int, max_seq_len: int, 
                 # User Sparse featrues
                 user_count: int, user_id_dim: int,  gender_count: int, gender_dim: int, age_group_count: int, age_group_dim: int,occup_group_count: int, occp_dim: int, zip_count: int, zip_dim:int, year_vocab_size: int, year_emb_dim: int, month_emb_dim: int, weekday_emb_dim: int, hour_emb_dim: int,
                 # User Dense features
                 user_activity_dim: int, 
                 # MLP Parameters
                 mlp_hidden_dims: list[int], output_dims: int,
                 dropout=0.1, 
                 weekday_vocab_size=8,  hour_vocab_size=25, month_vocab_size=13, dense_feat_emb_dim=16):
        super().__init__()

        self.seq_encoder = SequenceEncoder(
            item_count=item_count,
            genre_count=genre_count,
            emb_dim=seq_emb_dim,
            max_seq_len=max_seq_len,
            dropout=dropout
        )

        self.user_id_emb = nn.Embedding(user_count, user_id_dim, padding_idx=0)
        self.gender_emb = nn.Embedding(gender_count, gender_dim, padding_idx=0)
        self.age_emb = nn.Embedding(age_group_count, age_group_dim, padding_idx=0)
        self.occup_emb = nn.Embedding(occup_group_count, occp_dim, padding_idx=0)
        self.zip_emb = nn.Embedding(zip_count, zip_dim, padding_idx=0)
        self.year_emb = nn.Embedding(year_vocab_size, year_emb_dim, padding_idx=0)
        self.month_emb = nn.Embedding(month_vocab_size, month_emb_dim, padding_idx=0)
        self.weekday_emb = nn.Embedding(weekday_vocab_size, weekday_emb_dim, padding_idx=0)
        self.hour_emb = nn.Embedding(hour_vocab_size, hour_emb_dim, padding_idx=0)

        self.user_act_proj = nn.Linear(user_activity_dim, dense_feat_emb_dim)

        self.total_input_dim = seq_emb_dim + user_id_dim + gender_dim + age_group_dim + occp_dim + zip_dim + year_emb_dim + month_emb_dim + weekday_emb_dim + hour_emb_dim + dense_feat_emb_dim

        self.mlp = MLP_Tower(
            input_dim=self.total_input_dim,
            hidden_dims=mlp_hidden_dims,
            output_dim=output_dims,
            dropout=dropout
        )
    
    def forward(
            self, 
            # Sparse user features
            user_id, gender, age, occup, zip_code, year, month, hour, weekday, 
            # Dense user features
            user_activity, 
            # Sequence user features
            hist_movie_ids, hist_genre_ids):
        
        """
        Forward pass of the user tower.

        Encodes static user profile features, temporal context features, 
        and historical behavior sequences into a unified user representation.
        
        Args:
            user_id (torch.LongTensor): Tensor of shape (batch_size,) containing user ID indices.
            gender (torch.LongTensor): Tensor of shape (batch_size,) containing encoded gender features.
            age (torch.LongTensor): Tensor of shape (batch_size,) containing encoded age group indices.
            occup (torch.LongTensor): Tensor of shape (batch_size,) containing encoded occupation indices.
            zip_code (torch.LongTensor): Tensor of shape (batch_size,) containing encoded ZIP code indices.
            year (torch.LongTensor): Tensor of shape (batch_size,) representing the year of interaction.
            month (torch.LongTensor): Tensor of shape (batch_size,) representing the month of interaction.
            hour (torch.LongTensor): Tensor of shape (batch_size,) representing the hour of interaction.
            weekday (torch.LongTensor): Tensor of shape (batch_size,) representing the weekday of interaction.
            user_activity (torch.FloatTensor): Tensor of shape (batch_size, activity_dim) representing user-level activity statistics.
            hist_movie_ids (torch.LongTensor): Tensor of shape (batch_size, seq_len) containing historical movie IDs.
            hist_genre_ids (torch.LongTensor): Tensor of shape (batch_size, seq_len) containing genre IDs corresponding to the historical movies.

        Returns:
            torch.FloatTensor: Output tensor of shape (batch_size, output_dim).
            This represents the final embedding of the user. 
        """
                

        seq_vec = self.seq_encoder(hist_movie_ids, hist_genre_ids)
        user_vec = self.user_id_emb(user_id)
        gender_vec = self.gender_emb(gender)
        age_vec = self.age_emb(age)
        occup_vec = self.occup_emb(occup)
        zip_vec = self.zip_emb(zip_code)
        year_vec = self.year_emb(year)
        month_vec = self.month_emb(month)
        hour_vec = self.hour_emb(hour)
        weekday_vec = self.weekday_emb(weekday)


        user_act_vec = F.relu(self.user_act_proj(user_activity.view(-1, 1).float()))

        combined_vec = torch.cat([user_vec, gender_vec, age_vec, occup_vec, zip_vec, year_vec, month_vec, hour_vec, weekday_vec, user_act_vec, seq_vec], dim=1)
        output = self.mlp(combined_vec)
        return output






        





        