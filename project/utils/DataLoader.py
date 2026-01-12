import torch
from torch.utils.data import Dataset, DataLoader
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
    
    def __getitem__(self, index: int):
        row = self.data.iloc[index]

        item_inputs = {
            'movie_id': torch.tensor(row["movie_id_enc"], dtype=torch.long),
            'year': torch.tensor(row['release_year_enc'], dtype=torch.long),
            'genre': torch.tensor(row['genre_ids'], dtype=torch.long)
        }

        user_inputs = {
            'user_id': torch.tensor(row['user_id_enc'], dtype=torch.long),
            'gender': torch.tensor(row['gender_enc'], dtype=torch.long),
            'age': torch.tensor(row['age_enc']),
            'occup': torch.tensor(row['occupation_enc'], dtype=torch.long),
            'zip_code': torch.tensor(row['zip_enc'], dtype=torch.long),
            'year': torch.tensor(row['year_enc'], dtype=torch.long),
            'month':torch.tensor(row['rating_month'], dtype=torch.long),
            'hour':torch.tensor(row['rating_hour'], dtype=torch.long),
            'weekday': torch.tensor(row['rating_weekday'], dtype=torch.long),
            'user_activity': torch.tensor(row['user_activity_log'], dtype=torch.float),
            'hist_movie_ids': torch.tensor(row['hist_movie_ids'], dtype=torch.long),
            'hist_genre_ids': torch.tensor(row['hist_genre_ids'], dtype=torch.long)
        }

        return user_inputs, item_inputs
    
    


