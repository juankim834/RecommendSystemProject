import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from project.utils.config_utils import file_loader

class RecommendationDataset(Dataset):
    def __init__(self, config_path, pkl_path):
        """
        :param config_path: file path of config
        :param pkl_path: file path of pkl
        """
        raw_cfg = file_loader(config_path)
        cfg = raw_cfg.get("two_tower", {})
        self.cfg = cfg
        if len(cfg) == 0:
            raise RuntimeError("Config loading failed, no items in config")
        
        self.hard_neg_cfg = raw_cfg.get("hard_negatives", {})
        self.use_hard_negatives = self.hard_neg_cfg.get("enabled", False)
        self.num_hard_negatives = self.hard_neg_cfg.get("num_negatives", 5)
        
        # New: Get mapping config for hard negatives
        # Example: {"negative_field": "genre_id"} means the negative_ids correspond to genre_id
        self.hard_neg_mapping = self.hard_neg_cfg.get("mapping", {})
        # Example: "hard_negative_ids" - the column containing the list of negative IDs
        self.hard_neg_column = self.hard_neg_cfg.get("negative_column", "hard_negative_ids")
        
        # Convert DataFrame to dict of NumPy arrays for faster access
        df = pd.read_pickle(pkl_path)
        self.data = {col: df[col].values for col in df.columns}
        self.data_len = len(df)
        
        # Build reverse index for hard negative lookup
        if self.use_hard_negatives and self.hard_neg_mapping:
            self._build_negative_index(df)
        
        del df  # Free memory
        
        self.item_tower_cfg = self.cfg.get("item_tower", {})
        
        # Pre-cache feature lists to avoid repeated lookups
        self._cache_feature_lists()

    def _build_negative_index(self, df):
        """
        Build a reverse index to quickly find rows by target field value
        
        Example: if mapping is {"negative_field": "genre_id"}
        Creates index: {genre_id_value: [row_idx1, row_idx2, ...]}
        """
        self.negative_index = {}
        
        for neg_field, target_field in self.hard_neg_mapping.items():
            if target_field not in df.columns:
                print(f"Warning: target field '{target_field}' not found in data")
                continue
            
            # Build index: target_value -> list of row indices
            field_index = {}
            for idx, value in enumerate(df[target_field].values):
                # Handle different value types (could be int, str, etc.)
                key = value
                if key not in field_index:
                    field_index[key] = []
                field_index[key].append(idx)
            
            self.negative_index[target_field] = field_index
    
    def _cache_feature_lists(self):
        """Cache feature lists for each tower to avoid repeated parsing"""

        self.tower_features = {}
        for tower_name, tower_config in self.cfg.items():
            self.tower_features[tower_name] = {
                'sparse_features': self.feature_id_loader("sparse_features", tower_config),
                'dense_features': self.feature_id_loader("dense_features", tower_config),
                'sequence_features': self.feature_id_loader("sequence_features", tower_config),
                'max_seq_len': tower_config.get("transformer_parameters", {}).get("max_seq_len", 20)
            }

    def __len__(self):
        return self.data_len
    
    def __getitem__(self, idx):
        row = {col: self.data[col][idx] for col in self.data.keys()}

        data_dict = {}
        for tower_name, tower_config in self.cfg.items():
            features = self.tower_features[tower_name]

            data_dict[tower_name] = self._get_tower_meta(
                row,
                features.get('sparse_features', []),
                features.get('dense_features', []),
                features.get('sequence_features', []),
                features.get('max_seq_len', 20)
            )
        
        if self.use_hard_negatives:
            data_dict['hard_negatives'] = self._get_hard_negatives(row, idx)

        return data_dict

    
    def _get_tower_meta(self, row, sparse_id_list, dense_id_list, seq_id_list, max_seq_len):

        """
        Returns dict in format:
        {
            'sparse_feature': {feature_id: torch.Tensor, ...},
            'dense_feature': {feature_id: torch.Tensor, ...},
            'seq_feature': {feature_id: torch.Tensor, ...}
        }
        """

        data_dict = {}

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
    
    def _get_hard_negatives(self, row, idx):
        """
        Load hard negative samples using configurable mapping
        
        Example config:
        {
            "hard_negatives": {
                "enabled": true,
                "num_negatives": 5,
                "negative_column": "hard_negative_ids",
                "mapping": {
                    "negative_field": "genre_id"  # neg IDs correspond to genre_id values
                }
            }
        }
        
        Returns:
            list of dicts, each with same structure as item tower output
        """
        hard_negatives = []
        item_features = self.tower_features.get('item_tower', {})
        
        # Get the negative ID list from the configured column
        neg_ids = row.get(self.hard_neg_column, [])
        
        if not isinstance(neg_ids, (list, np.ndarray)):
            neg_ids = [neg_ids] if neg_ids else []
        else:
            neg_ids = list(neg_ids)
        
        # Limit to configured number
        neg_ids = neg_ids[:self.num_hard_negatives]
        
        # Get target field from mapping
        if not self.hard_neg_mapping:
            print("Warning: hard_neg_mapping is empty, cannot retrieve hard negatives")
            return self._get_empty_negatives(item_features)
        
        neg_field, target_field = list(self.hard_neg_mapping.items())[0]
        
        if target_field not in self.negative_index:
            print(f"Warning: target field '{target_field}' not in negative index")
            return self._get_empty_negatives(item_features)
        
        field_index = self.negative_index[target_field]
        
        # For each negative ID, find corresponding rows
        for neg_id in neg_ids:
            # Find rows where target_field == neg_id
            candidate_indices = field_index.get(neg_id, [])
            
            if not candidate_indices:
                # No match found, add empty negative
                hard_negatives.append(self._get_empty_item_features(item_features))
                continue
            
            # Take first candidate (or could randomize)
            # Ensure we don't pick the current row itself
            valid_candidates = [i for i in candidate_indices if i != idx]
            
            if not valid_candidates:
                hard_negatives.append(self._get_empty_item_features(item_features))
                continue
            
            neg_idx = valid_candidates[0]  # Or random.choice(valid_candidates)
            
            # Get row for this hard negative
            neg_row = {col: self.data[col][neg_idx] for col in self.data.keys()}
            
            # Get item tower features for this negative
            neg_features = self._get_tower_meta(
                neg_row,
                item_features.get('sparse_features', []),
                item_features.get('dense_features', []),
                item_features.get('sequence_features', []),
                item_features.get('max_seq_len', 20)
            )
            hard_negatives.append(neg_features)
        
        # Pad with empty samples if we don't have enough hard negatives
        while len(hard_negatives) < self.num_hard_negatives:
            hard_negatives.append(self._get_empty_item_features(item_features))
        
        return hard_negatives
    
    def _get_empty_negatives(self, item_features):
        """Return list of empty negatives"""
        return [self._get_empty_item_features(item_features) 
                for _ in range(self.num_hard_negatives)]
    
    def _get_empty_item_features(self, item_features):
        """Create empty/zero features matching item tower structure"""
        empty_dict = {}
        
        if item_features.get('sparse'):
            empty_dict['sparse_feature'] = {
                feat: torch.tensor(0, dtype=torch.long) 
                for feat in item_features['sparse']
            }
        
        if item_features.get('dense'):
            empty_dict['dense_feature'] = {
                feat: torch.tensor([0.0], dtype=torch.float32)
                for feat in item_features['dense']
            }
        
        if item_features.get('sequence'):
            max_seq_len = item_features.get('max_seq_len', 20)
            empty_dict['seq_feature'] = {
                feat: torch.zeros(max_seq_len, dtype=torch.long)
                for feat in item_features['sequence']
            }
        
        return empty_dict


    def feature_id_loader(self, feature_type, cfg):

        """
        Helper function autimatically load feature id from config.
        
        :param feature_type: String 
        Specify the feature that is going to load
        :param cfg: dict
        Config dictionary load from the config file, it has to be the tower config dictionary, not all of the config dictionary.
        """

        feature_list = cfg.get(feature_type, [])
        
        if not feature_list:
            return []
        
        feature_id_list = []
        for feature in feature_list:
            feature_id = feature.get("name")
            feature_id_list.append(feature_id)
        return feature_id_list
        
    def _pad_or_truncate(self, data_list, target_len, pad_val=0):
        if not isinstance(data_list, list):
            if hasattr(data_list, 'tolist'):
                data_list = data_list.tolist()
            else:
                data_list = list(data_list)
                
        curr_len = len(data_list)
        if curr_len < target_len:
            return data_list + [pad_val] * (target_len - curr_len)
        elif curr_len > target_len:
            return data_list[:target_len]
        return data_list

    
    


