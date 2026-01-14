import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from project.utils.config_utils import file_loader

class RecommendationDataset(Dataset):
    def __init__(self, config_path, tower_name,pkl_path, max_seq_len=20):
        """

        :param config_path: file path of config
        :param tower_name: tower name
        :param pkl_path: file path of pkl
        :param max_seq_len: max length of sequence
        """
        raw_cfg = file_loader(config_path)
        cfg = raw_cfg.get("two_tower", {}).get(tower_name, {})
        if len(cfg) == 0:
            raise RuntimeError(f"Config loading failed, {tower_name} has no items in config")
        self.sparse_id_list = self.feature_id_loader("sparse_features", cfg)
        self.dense_id_list = self.feature_id_loader("dense_features", cfg)
        self.seq_id_list = self.feature_id_loader("sequence_features", cfg)
        data_raw = pd.read_pickle(pkl_path)
        data_raw = data_raw[data_raw['label']==1]
        self.data = data_raw.to_dict('records')
        self.max_seq_len = max_seq_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data[idx]
        data_dict = {}

        # 1. Sparse Data
        if self.sparse_id_list:
            feat_data_dict = {}
            for id in self.sparse_id_list:
                feat_data_dict[id] = torch.tensor(row[id], dtype=torch.long)
            data_dict["sparse_feature"] = feat_data_dict
        
        # 2. Dense Data
        if self.dense_id_list:
            feat_data_dict = {}
            for id in self.dense_id_list:
                val = row[id]
                feat_data_dict[id] = torch.tensor(val, dtype=torch.float32)
            data_dict["dense_feature"] = feat_data_dict
        
        # 3. Sequence Data
        if self.seq_id_list:
            feat_data_dict = {}
            for id in self.seq_id_list:
                raw_seq = row[id]
                seq = list(raw_seq)
                
                # Padding / Truncating Logic
                if len(seq) > self.max_seq_len:
                    seq = seq[-self.max_seq_len:]
                else:
                    seq = seq + [0] * (self.max_seq_len - len(seq))
                feat_data_dict[id] = torch.tensor(seq, dtype=torch.long)
            data_dict["seq_feature"] = feat_data_dict

        return data_dict
    
    def feature_id_loader(self, feature_type, cfg):
        feature_list = cfg.get(feature_type, [])
        
        if not feature_list:
            return []
        
        feature_id_list = []
        for feature in feature_list:
            feature_id = feature.get("name")
            feature_id_list.append(feature_id)
        return feature_id_list
        

    
    


