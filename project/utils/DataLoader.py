import torch
from torch.utils.data import Dataset
import pandas as pd
from project.utils.config_utils import file_loader

class RecommendationDataset(Dataset):
    def __init__(self, config_path, pkl_path):
        """

        :param config_path: file path of config
        :param tower_name: tower name
        :param pkl_path: file path of pkl
        :param max_seq_len: max length of sequence
        """
        raw_cfg = file_loader(config_path)
        cfg = raw_cfg.get("two_tower", {})
        self.cfg = cfg
        if len(cfg) == 0:
            raise RuntimeError("Config loading failed, no items in config, do you need to run config_utils.py to dump sample config?")
        self.hard_neg_cfg = raw_cfg.get("hard_negatives", {})
        self.use_hard_negatives = self.hard_neg_cfg.get("enabled", False)
        
        self.num_hard_negatives = self.hard_neg_cfg.get("num_negatives", 5)
        
        self.data = pd.read_pickle(pkl_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        data_dict = {}
        for tower_name, tower_config in self.cfg.items():
            tower_seq_len = tower_config.get("transformer_parameters", {}).get("max_seq_len", 20)
            data_dict[tower_name] = self._get_tower_meta(row, tower_config, tower_seq_len)
        
        if self.use_hard_negatives:
            data_dict['hard_negatives'] = self._get_hard_negatives(row)

        return data_dict

    
    def _get_tower_meta(self, row, tower_cfg, max_seq_len):
        data_dict = {}
        
        sparse_id_list = self.feature_id_loader("sparse_features", tower_cfg)
        dense_id_list = self.feature_id_loader("dense_features", tower_cfg)
        seq_id_list = self.feature_id_loader("sequence_features", tower_cfg)

        # 1. Sparse Data
        if sparse_id_list:
            feat_data_dict = {}
            for feat_name in sparse_id_list:
                val = row.get(feat_name, 0)
                feat_data_dict[feat_name] = torch.tensor(val, dtype=torch.long)
            data_dict["sparse_feature"] = feat_data_dict
        
        # 2. Dense Data
        if dense_id_list:
            feat_data_dict = {}
            for feat_name in dense_id_list:
                val = row.get(feat_name, 0.0)
                tensor_val = torch.tensor(val, dtype=torch.float32)
                if tensor_val.ndim == 0:
                    tensor_val = tensor_val.unsqueeze(0)
                feat_data_dict[feat_name] = tensor_val
            data_dict["dense_feature"] = feat_data_dict
        
        # 3. Sequence Data
        if seq_id_list:
            feat_data_dict = {}
            for feat_name in seq_id_list:
                raw_seq = row.get(feat_name, [])
                seq = list(raw_seq)
                if len(seq) > max_seq_len:
                    seq = seq[-max_seq_len:]
                else:
                    seq = seq + [0] * (max_seq_len - len(seq))
                feat_data_dict[feat_name] = torch.tensor(seq, dtype=torch.long)
            data_dict["seq_feature"] = feat_data_dict
            
        return data_dict
    
    def _get_hard_negatives(self, row):
        """
        Load hard negative samples from the row
        
        Returns:
            dict with hard negative data based on config
        """
        hard_neg_dict = {}
        
        # Get the field name from config (default: 'hard_neg_ids')
        field_name = self.hard_neg_cfg.get("field_name", "hard_neg_ids")
        
        # Load hard negative IDs
        hard_neg_ids = row.get(field_name, [0] * self.num_hard_negatives)
        hard_neg_ids = list(hard_neg_ids)
        
        # Ensure correct length
        if len(hard_neg_ids) > self.num_hard_negatives:
            hard_neg_ids = hard_neg_ids[:self.num_hard_negatives]
        elif len(hard_neg_ids) < self.num_hard_negatives:
            hard_neg_ids = hard_neg_ids + [0] * (self.num_hard_negatives - len(hard_neg_ids))
        
        hard_neg_dict['ids'] = torch.tensor(hard_neg_ids, dtype=torch.long)
        
        # Load additional features for hard negatives if configured
        additional_features = self.hard_neg_cfg.get("additional_features", [])
        for feature_name in additional_features:
            feature_field = f"hard_neg_{feature_name}"
            if feature_field in row:
                feature_data = row[feature_field]
                # Assuming these are also lists/sequences
                if isinstance(feature_data, list):
                    hard_neg_dict[feature_name] = torch.tensor(feature_data, dtype=torch.long)
        
        return hard_neg_dict
    
    def feature_id_loader(self, feature_type, cfg):
        feature_list = cfg.get(feature_type, [])
        
        if not feature_list:
            return []
        
        feature_id_list = []
        for feature in feature_list:
            feature_id = feature.get("name")
            feature_id_list.append(feature_id)
        return feature_id_list
        

    
    


