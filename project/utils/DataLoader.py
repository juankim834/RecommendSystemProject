import torch
from torch.utils.data import Dataset, DataLoader
import numpy as py
import pandas as pd

class RecommendationDataset(Dataset):
    def __init__(self, pkl_path, max_seq_len=20):
        """

        :param pkl_path: file path of pkl
        :param max_seq_len: max length of sequence
        """

        self.data = pd.read_pickle(pkl_path)
        self.max_seq_len = max_seq_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data.iloc[index]

        item_inputs = {
            'movie_id': torch.tensor(row["movie_id_enc"], dtype=torch.long),
            'year': torch.tensor(row['movei_year_enc'], dtype=torch.long),
            'genre': torch.tensor(row['genre_ids'], dtype=torch.long)
        }

        user_inputs = {
            'user_id': torch.tensor(row['user_id_enc'], dtype=torch.long),
            'gender': torch.tensor(row['gender_enc'], dtype=torch.long),
            'occup': torch.tensor(row['occupation_enc'], dtype=torch.long),
            'zip': torch.tensor(row['zip_enc'], dtype=torch.long),
            'year_rate': torch.tensor(row['year_enc'], dtype=torch.long),
            'rating_month':torch.tensor(row['rating_month'], dtype=torch.long),
            'rating_hour':torch.tensor(row['rating_hour'], dtype=torch.long),
            'rating_weekday': torch.tensor(row['rating_weekday'], dtype=torch.long),
            'rating_activiry_log': torch.tensor(row['user_activity_log'], dtype=torch.long),
            'hist_movie_ids': torch.tensor(row['hist_movie_ids'], dtype=torch.long),
            'hist_genre_ids': torch.tensor(row['hist_genre_ids'], dtype=torch.long)
        }

        return user_inputs, item_inputs
    
    


