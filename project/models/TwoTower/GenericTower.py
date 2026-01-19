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
        self.dense_features = tower_cfg.get("dense_features", [])
        self.seq_features = tower_cfg.get("sequence_features", [])
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

                nn.init.xavier_uniform_(self.embeddings[name].weight)

                if 'pooling' in feat:
                    self.pooling_config[name] = feat['pooling']

                sparse_total_dim = sparse_total_dim + embedding_dim
        
        # Initializing Dense features
        dense_total_dim = 0
        if self.dense_features is not None:
            for feat in self.dense_features:
                name = feat["name"]
                origin_dim = feat["dim"]
                embedding_dim = feat["embedding_dim"]

                self.embeddings[name] = nn.Sequential(
                    # nn.BatchNorm1d(origin_dim),
                    nn.Linear(origin_dim, embedding_dim),
                )
                dense_total_dim = dense_total_dim + embedding_dim
        
        # Initializing Sequence features
        seq_total_dim = 0
        if self.seq_features is not None:
            if len(self.seq_features) > 0:
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
                    feature_config_list=self.seq_features,
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

        self.feature_bn = nn.BatchNorm1d(self.total_embed_dim)

        self.mlp = MLP_Tower(
            input_dim=self.total_embed_dim, 
            hidden_dims=mlp_hidden_dims,
            output_dim=output_dims,
            dropout=dropout_cfg
        )
    
    def forward(self, input_dict, feature_column_mapping = None):
        """

        :param input_dict: Feature dictionary with batched matrices
            Format: {
                'sparse': torch.Tensor of shape (batch_size, n_sparse_features),
                'dense': torch.Tensor of shape (batch_size, n_dense_features),
                'sequence': {feature_id: torch.Tensor of shape (batch_size, seq_len)}
            }
        """
        
        feature_embs = []
        # Sparse features
        if self.sparse_features and 'sparse' in input_dict:
            sparse_matrix = input_dict['sparse'] # Shape: (batch_size, n_sparse_features)
            sequence_dict = input_dict.get('sequence', {})

            col_idx = 0
            for feat_cfg in self.sparse_features:
                feature_id = feat_cfg["name"] # Shape: (batch_size,)
                has_pooling = 'pooling' in feat_cfg
                if has_pooling:
                    if feature_id not in sequence_dict:
                        print(f"Warning: Pooled feature {feature_id} missing from sequence dict")
                        continue
                    feature_data = sequence_dict[feature_id]
                    pooling_type = self.pooling_config[feature_id]
                    pooling_type = self.pooling_config[feature_id]

                    # Ensure Sequence Dimension exists [Batch, Seq]
                    if feature_data.dim() == 1:
                        feature_data = feature_data.unsqueeze(1)

                    emb = self.embeddings[feature_id](feature_data)
                    # Apply pooling if configured
                    if pooling_type == 'mean':
                        emb = torch.mean(emb, dim=1) # Shape: (batch_size, embed_dim)
                    elif pooling_type == 'sum':
                        emb = torch.sum(emb, dim=1)
                    elif pooling_type == 'max':
                        emb = torch.max(emb, dim=1)[0]
                    
                    feature_embs.append(emb)
                
                else:
                    # Single-value sparse feature (in sparse matrix)
                    # Use mapping if provided, otherwise assume config order
                    if sparse_matrix is None: continue

                    if feature_column_mapping and 'sparse' in feature_column_mapping:
                        col_idx = feature_column_mapping['sparse'].get(feature_id)
                        if col_idx is None:
                            raise ValueError(f"Feature '{feature_id}' not found in column mapping")
                    else:
                        # Fallback: use order in config (less safe!)
                        col_idx = [f['name'] for f in self.sparse_features if 'pooling' not in f].index(feature_id)
                    
                    feature_col = sparse_matrix[:, col_idx]  # Shape: (batch_size,)

                    # Embed entire column at once!
                    if feature_id in self.embeddings:
                        try:
                            emb = self.embeddings[feature_id](feature_col)  # Shape: (batch_size, embed_dim)
                            feature_embs.append(emb)
                        except IndexError: # If out of index in CUDA, switch to run on cpu to debug
                            name = feat_cfg.get("name", "Unknown")
                            vocab_size = feat_cfg.get("vocab_size", "Unknown")
                            print(f"Config item {name}")
                            print(f"{feature_id} is out of index, and the col_idx {col_idx}")
                            print(f"Config input of Vocab_size: {vocab_size}")
                            print(f"Vocab_size input: {self.embeddings[feature_id].num_embeddings}")
                            print(f"[DEBUG] feature_id = {feature_id}")
                            print(f"[DEBUG] used col_idx = {col_idx}")
                            print(f"[DEBUG] feature_col max = {feature_col.max().item()}")
                            print(f"[DEBUG] feature_col min = {feature_col.min().item()}")
                            print(f"[DEBUG] vocab_size = {self.embeddings[feature_id].num_embeddings}")
                            raise
        
        if self.dense_features and 'dense' in input_dict:
            dense_matrix = input_dict['dense']

            col_idx = 0
            for feat_cfg in self.dense_features:
                feature_id = feat_cfg["name"]

                # Extract this feature's column
                if feature_column_mapping and 'dense' in feature_column_mapping:
                    col_idx = feature_column_mapping['dense'].get(feature_id)
                    if col_idx is None:
                        raise ValueError(f"Dense feature '{feature_id}' not found in column mapping")
                else:
                    # Fallback: use order in config
                    col_idx = [f['name'] for f in self.dense_features].index(feature_id)
                
                # Extract this feature's column
                feature_col = dense_matrix[:, col_idx:col_idx+1] # Shape: (batch_size, 1)

                # Process through linear layer
                if feature_id in self.embeddings:
                    if feature_col.dtype != torch.float32:
                        feature_col = feature_col.float()
                    emb = self.embeddings[feature_id](feature_col)
                    feature_embs.append(emb)
        
        if self.seq_encoder is not None and 'sequence' in input_dict:
            seq_feature_dict = input_dict['sequence']
            if seq_feature_dict:
                seq_emb = self.seq_encoder(seq_feature_dict)
                feature_embs.append(seq_emb)
        
        if not feature_embs:
            raise RuntimeError("Tower received no valid features. Check if input_dict matches config")
        
        concat_emb = torch.cat(feature_embs, dim=1)
        concat_emb = self.feature_bn(concat_emb)
        output = self.mlp(concat_emb)

        return output




        





        