import torch
import torch.nn as nn
import torch.nn.functional as F

class SequenceFeatureProcessor(nn.Module):
    def __init__(self, feature_config_list, target_dim, max_seq_len, dropout=0.1):
        """
        :param feature_config_list: sequence_features (List of Dicts)
        :param target_dim: Dimensions of model (d_model)
        :param max_seq_len: Max sequence length
        """
        super().__init__()
        self.feature_config_list = feature_config_list
        self.target_dim = target_dim
        self.dropout = dropout
        self.embeddings = nn.ModuleDict()
        total_concat_dim = 0
        for feat_cfg in feature_config_list:
            feat_name = feat_cfg['name']
            vocab_size = feat_cfg['vocab_size']
            emb_dim = feat_cfg['embedding_dim']
            padding_idx = feat_cfg.get('padding_index', 0)
            
            # Create Embedding
            self.embeddings[feat_name] = nn.Embedding(
                num_embeddings=vocab_size, 
                embedding_dim=emb_dim, 
                padding_idx=padding_idx
            )
            total_concat_dim += emb_dim
        
        self.feature_projection = nn.Sequential(
            nn.Linear(total_concat_dim, target_dim),
            nn.Dropout(dropout)
        )
        self.pos_emb = nn.Embedding(max_seq_len, target_dim)

    def forward(self, input_dict):
        """
        :params input_dict:  {'hist_item_id': tensor(...), 'hist_genre_id': tensor(...)}
        """
        # batch_size = next(iter(input_dict.values())).shape[0]
        device = next(iter(input_dict.values())).device
        
        emb_list = []

        for feat_cfg in self.feature_config_list:
            name = feat_cfg['name']
            pooling_type = feat_cfg.get('pooling', None)
            
            # Data Input
            if name not in input_dict:
                print(f"Configuration Error: Unable to find {name} in the input dictionary, {name} has skipped")
                continue
            x = input_dict[name]
            
            # Possible x shape:
            # [Batch, Seq](Normal Sequence) -> emb -> [Batch, Seq, Dim]
            # [Batch, Seq, Tags] (Multi number sequence) -> emb -> [Batch, Seq, Tags, Dim]
            emb = self.embeddings[name](x)

            # Pooling Processing for multi-value features
            # Transform [Batch, Seq, Tags, Dim] into [Batch, Seq, Dim
            if x.dim() == 3:
                if pooling_type == 'mean':
                    emb = torch.mean(emb, dim=2)
                elif pooling_type == 'sum':
                    emb = torch.sum(emb, dim=2)

            emb_list.append(emb)

        # total_emb shape: [Batch, Seq, Dim]
        if not emb_list:
            raise ValueError("Configuration Error: No valid features were processed!")
        
        concat_emb = torch.cat(emb_list, dim=-1)
        total_emb = self.feature_projection(concat_emb)

        seq_len = total_emb.shape[1]
        positions = torch.arange(seq_len, device=device).unsqueeze(0) # [1, Seq]

        total_emb = total_emb + self.pos_emb(positions)
        total_emb = F.dropout(total_emb, p=self.dropout, training=self.training)
        
        return total_emb
    