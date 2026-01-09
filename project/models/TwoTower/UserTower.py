import torch
import torch.nn as nn
import torch.nn.functional as F
from SequenceEncoder import SequenceEncoder
from Tower import MLP_Tower

class UserTower(nn.Module):
    def __init__(self, 
                 # Sequence parameters
                 item_count: int, genre_count: int, seq_emb_dim: int, max_seq_len: int, 
                 # User Sparse featrues
                 user_count: int, gender_count: int, user_feat_dim: int, age_group_count: int, occup_group_count: int, zip_count: int, year_vocab_size: int, year_emb_dim: int, month_emb_dim: int, weekday_emb_dim: int, hour_emb_dim: int,
                 # User Dense features
                 user_activity_dim: int, 
                 # MLP Parameters
                 mlp_hidden_dims: int, output_dims: int,
                 dropout=0.1, 
                 weekday_vocab_size=8,  hour_vocab_size=25, month_vocab_size=12, dense_feat_emb_dim=16):
        super().__init__()

        self.seq_encoder = SequenceEncoder(
            item_count=item_count,
            genre_count=genre_count,
            emb_dim=seq_emb_dim,
            max_seq_len=max_seq_len,
            dropout=dropout
        )

        self.user_id_emb = nn.Embedding(user_count, user_feat_dim)
        self.gender_emb = nn.Embedding(gender_count, user_feat_dim)
        self.age_emb = nn.Embedding(age_group_count, user_feat_dim)
        self.occup_emb = nn.Embedding(occup_group_count, user_feat_dim)
        self.zip_emb = nn.Embedding(zip_count, user_feat_dim)
        self.year_emb = nn.Embedding(year_vocab_size, year_emb_dim)
        self.month_emb = nn.Embedding(month_vocab_size, month_emb_dim)
        self.weekday_emb = nn.Embedding(weekday_vocab_size, weekday_emb_dim, padding_idx=0)
        self.hour_emb = nn.Embedding(hour_vocab_size, hour_emb_dim, padding_idx=0)

        self.user_act_proj = nn.Linear(user_activity_dim, dense_feat_emb_dim)

        self.total_input_dim = seq_emb_dim + 5 * user_feat_dim + year_emb_dim + month_emb_dim + weekday_emb_dim + hour_emb_dim + dense_feat_emb_dim

        self.mlp = MLP_Tower(
            input_dim=self.total_input_dim,
            hidden_dims=mlp_hidden_dims,
            output_dim=output_dims,
            dropout=dropout
        )
    
    def forward(
            self, 
            # Sparse user features
            user_id, gender, age, occup, zip, year, month, 
            # Dense user features
            hour, weekday, user_activity, 
            # Sequence user features
            hist_movie_ids, hist_genre_ids):
        '''
        Forward pass of the user tower.

        Encodes static user profile features, temporal context features,
        and historical behavior sequences into a unified user representation.
        
        :param user_id: torch.LongTensor
            Tensor of shape (batch_size,) containing user ID indices.
        :param gender: torch.LongTensor
            Tensor of shape (batch_size,) containing encoded gender features.
        :param age: torch.LongTensor
            Tensor of shape (batch_size,) containing encoded age group indices.
        :param occup: torch.LongTensor
            Tensor of shape (batch_size,) containing encoded occupation indices.
        :param zip: torch.LongTensor
            Tensor of shape (batch_size,) containing encoded ZIP code indices.
        :param year: torch.LongTensor
            Tensor of shape (batch_size,) representing the year of interaction.
        :param month: torch.LongTensor
            Tensor of shape (batch_size,) representing the month of interaction.
        :param hour: torch.LongTensor
            Tensor of shape (batch_size,) representing the hour of interaction.
        :param weekday: torch.LongTensor
            Tensor of shape (batch_size,) representing the weekday of interaction.
        :param user_activity: torch.LongTensor
            Tensor of shape (batch_size, activity_dim) representing user-level activity statistics (interaction frequency).
        :param hist_movie_ids: torch.LongTensor
            Tensor of shape (batch_size, seq_len) containing historical movie IDs.
        :param hist_genre_ids: torch.LongTensor
            Tensor of shape (batch_size, seq_len) containing genre IDs corresponding to the historical movies.
        '''
        seq_vec = self.seq_encoder(hist_movie_ids, hist_genre_ids)
        user_vec = self.user_id_emb(user_id)
        gender_vec = self.gender_emb(gender)
        age_vec = self.age_emb(age)
        occup_vec = self.occup_emb(occup)
        zip_vec = self.zip_emb(zip)
        year_vec = self.year_emb(year)
        month_vec = self.month_emb(month)
        hour_vec = self.hour_emb(hour)
        weekday_vec = self.weekday_emb(weekday)

        user_act_vec = F.relu(self.user_act_proj(user_activity))

        combined_vec = torch.cat([user_vec, gender_vec, age_vec, occup_vec, zip_vec, year_vec, month_vec, hour_vec, weekday_vec, seq_vec])
        output = self.mlp(combined_vec)
        return output






        





        