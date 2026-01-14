import torch
from torch.utils.data import Dataset, DataLoader
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
            raise RuntimeError("Config loading failed, no items in config")
        data_raw = pd.read_pickle(pkl_path)
        data_raw = data_raw[data_raw['label']==1]
        self.data = data_raw.to_dict('records')
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data[idx]
        data_dict = {}
        for tower_name, tower_config in self.cfg.items():
            # 动态获取当前塔设定的 max_seq_len
            # 比如 user_tower 是 20，item_tower 是 3
            tower_seq_len = tower_config.get("transformer_parameters", {}).get("max_seq_len", 20)
            
            # 将当前塔的配置传给处理函数
            data_dict[tower_name] = self._get_tower_meta(row, tower_config, tower_seq_len)
            
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
                feat_data_dict[feat_name] = torch.tensor(val, dtype=torch.float32)
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
    
    def feature_id_loader(self, feature_type, cfg):
        feature_list = cfg.get(feature_type, [])
        
        if not feature_list:
            return []
        
        feature_id_list = []
        for feature in feature_list:
            feature_id = feature.get("name")
            feature_id_list.append(feature_id)
        return feature_id_list
        

    
    


