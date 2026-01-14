import torch
import torch.nn as nn
import torch.nn.functional as F
from project.models.TwoTower.SequenceEncoder import SequenceEncoder
from project.models.TwoTower.Tower import MLP_Tower

class UserTower(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        cfg_tower = cfg.get('two_tower', {})
        if (len(cfg_tower.get('user_tower', {}))==0):
            raise RuntimeError(f'TwoTower Model initializing failed, User Tower has no features')
        

        user_tower_cfg = cfg_tower.get('user_tower')
        mlp_hidden_dims = user_tower_cfg["mlp_hidden_dim"]
        output_dims = user_tower_cfg["output_dims"]
        dropout_cfg = user_tower_cfg['dropout']
        self.embeddings = nn.ModuleDict()

        # Initializing Sparse features
        user_sparse_features = user_tower_cfg.get("sparse_features", [])
        sparse_total_dim = 0
        for feat in user_sparse_features:
            name = feat["name"]
            vocab_size = feat["vocab_size"]
            embedding_dim = feat["embedding_dim"]
            padding_idx = feat.get("padding_idx", None)

            self.embeddings[name] = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embedding_dim,
                padding_idx=padding_idx
            )
            sparse_total_dim = sparse_total_dim + embedding_dim
        
        # Initializing Dense features
        user_dense_feature = user_tower_cfg.get('dense_features', [])
        dense_total_dim = 0
        for feat in user_dense_feature:
            name = feat["name"]
            origin_dim = feat["dim"]
            embedding_dim = feat["embedding_dim"]

            self.embeddings[name] = nn.Linear(origin_dim, embedding_dim)
            dense_total_dim = dense_total_dim + embedding_dim
        
        # Initializing Sequence features
        user_sequence_feature = user_tower_cfg.get('sequence_features', [])
        seq_total_dim = 0
        for feat in user_sequence_feature:
            seq_total_dim = seq_total_dim + feat["embedding_dim"]
        if len(user_sequence_feature) != 0:
            self.seq_encoder = SequenceEncoder(
                featrue_config_list=user_sequence_feature,
                model_dim=64,
                max_seq_len=20
            )
        
        self.total_embed_dim = sparse_total_dim + dense_total_dim + seq_total_dim

        self.mlp = MLP_Tower(
            input_dim=self.total_embed_dim, 
            hidden_dims=mlp_hidden_dims,
            output_dim=output_dims,
            dropout=dropout_cfg
        )
    
    def forward(self, input_dict):
        """
        :param input_dict: Feature dictionary. 
            Input example:{sparse_feature: {feature_id:torch.Tensor, ...}, dense_feature:{feature_id:torch.Tensor, ...}, seq_feature:{feature_id:torch.Tensor, ...}}
        """
        # Sparse features
        feature_embs = []
        sparse_feature = input_dict.get('sparse_feature', None)
        if sparse_feature is not None:
            for feature_id, data in sparse_feature.items():
                if feature_id in self.embeddings:
                    emb = self.embeddings[feature_id](data)
                    feature_embs.append(emb)
        # Dense features
        dense_feature = input_dict.get('dense_feature', None)
        if dense_feature is not None:
            for feature_name, data in dense_feature.items():
                if feature_name in self.embeddings:
                    emb = self.embeddings[feature_name](data)
                    feature_embs.append(emb)
        
        seq_feature_dict = input_dict.get('seq_feature', None)
        if self.seq_encoder is not None and seq_feature_dict is not None:
            seq_emb = self.seq_encoder(seq_feature_dict)
            feature_embs.append(seq_emb)
        
        if not feature_embs:
            raise RuntimeError("UserTower received an empty valid feature. Please check if the Input Dictionary and Config match")
        
        concat_emb = torch.cat(feature_embs, dim=1)
        output = self.mlp(concat_emb)
        return output





        





        