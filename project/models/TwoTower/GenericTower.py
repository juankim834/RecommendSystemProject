import torch
import torch.nn as nn
import torch.nn.functional as F
from project.models.TwoTower.SequenceEncoder import SequenceEncoder
from project.models.TwoTower.Tower import MLP_Tower

class GenericTower(nn.Module):

    def __init__(self, cfg, tower_name):
        super().__init__()
        # model_cfg: 2 side of towers
        # tower_cfg: only tower_name side tower
        model_cfg = cfg.get('two_tower', {})
        if (len(model_cfg.get(tower_name, {}))==0):
            raise RuntimeError(f'TwoTower Model initializing failed, {tower_name} has no features')
        
        tower_cfg = model_cfg.get(tower_name)
        mlp_hidden_dims = tower_cfg["mlp_hidden_dim"]
        output_dims = tower_cfg["output_dims"]
        dropout_cfg = tower_cfg['dropout']
        self.tower_embedding_dim = tower_cfg["embedding_dim"]
        self.embeddings = nn.ModuleDict()
        self.pooling_config = {}

        # Initializing Sparse features
        self.sparse_features = tower_cfg.get("sparse_features", [])
        sparse_total_dim = 0
        if self.sparse_features is not None:
            for feat in self.sparse_features:
                name = feat["name"]
                vocab_size = feat["vocab_size"]
                embedding_dim = feat["embedding_dim"]
                padding_idx = feat.get("padding_idx", 0)

                self.embeddings[name] = nn.Embedding(
                    num_embeddings=vocab_size,
                    embedding_dim=embedding_dim,
                    padding_idx=padding_idx
                )

                if 'pooling' in feat:
                    self.pooling_config[name] = feat['pooling']

                sparse_total_dim = sparse_total_dim + embedding_dim
        
        # Initializing Dense features
        dense_feature = tower_cfg.get('dense_features', [])
        dense_total_dim = 0
        if dense_feature is not None:
            for feat in dense_feature:
                name = feat["name"]
                origin_dim = feat["dim"]
                embedding_dim = feat["embedding_dim"]

                self.embeddings[name] = nn.Sequential(
                    nn.BatchNorm1d(origin_dim),
                    nn.Linear(origin_dim, embedding_dim),
                    nn.ReLU()
                )
                dense_total_dim = dense_total_dim + embedding_dim
        
        # Initializing Sequence features
        sequence_feature = tower_cfg.get('sequence_features', [])
        seq_total_dim = 0
        if sequence_feature is not None:
            if len(sequence_feature) > 0:
                model_dim = tower_cfg.get("embedding_dim", 32)
                transformer_cfg = tower_cfg.get("transformer_parameters", {})
                max_seq_len = transformer_cfg.get("max_seq_len", 20)
                trans_dropout = transformer_cfg.get("dropout", 0.1)
                ffN_dim = transformer_cfg.get("FFN_dim", 4*model_dim)
                n_head = transformer_cfg.get("n_head", 4)
                n_layers = transformer_cfg.get("n_layers", 1)
                if model_dim % n_head != 0:
                    raise ValueError(f"Embedding dim {model_dim} must be divisible by n_head {n_head}")

                self.seq_encoder = SequenceEncoder(
                    feature_config_list=sequence_feature,
                    model_dim=model_dim,
                    dim_feedforward=ffN_dim,
                    max_seq_len=max_seq_len,
                    n_head=n_head,
                    n_layers=n_layers,
                    dropout=trans_dropout
                )
                seq_total_dim = model_dim
            else:
                self.seq_encoder = None

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
                    feat_cfg = next((f for f in self.sparse_features if f['name'] == feature_id), None)
                    if feature_id in self.pooling_config and emb.dim() > 2:
                        pooling_type = self.pooling_config[feature_id]
                        if pooling_type == 'mean':
                            emb = torch.mean(emb, dim=1)
                        elif pooling_type == 'sum':
                            emb = torch.sum(emb, dim=1)
                    feature_embs.append(emb)
        # Dense features
        dense_feature = input_dict.get('dense_feature', None)
        if dense_feature is not None:
            for feature_name, data in dense_feature.items():
                if feature_name in self.embeddings:
                    emb = self.embeddings[feature_name](data)
                    feature_embs.append(emb)
        
        # Sequence features
        seq_feature_dict = input_dict.get('seq_feature', None)
        if self.seq_encoder is not None:
            seq_emb = self.seq_encoder(seq_feature_dict)
            feature_embs.append(seq_emb)
        
        if not feature_embs:
            raise RuntimeError("Tower received an empty valid feature. Please check if the Input Dictionary and Config match")
        
        concat_emb = torch.cat(feature_embs, dim=1)
        output = self.mlp(concat_emb)
        return output
    





        





        