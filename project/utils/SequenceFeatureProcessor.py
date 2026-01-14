import torch
import torch.nn as nn

class SequenceFeatureProcessor(nn.Module):
    def __init__(self, feature_config_list, target_dim, max_seq_len):
        """
        :param feature_config_list: sequence_features (List of Dicts)
        :param target_dim: Dimensions of model (d_model)
        :param max_seq_len: Max sequence length
        """
        super().__init__()
        self.feature_config_list = feature_config_list
        self.target_dim = target_dim
        
        self.embeddings = nn.ModuleDict()
        
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
            if emb_dim != target_dim:
                self.embeddings[feat_name + "_proj"] = nn.Linear(emb_dim, target_dim)
        self.pos_emb = nn.Embedding(max_seq_len, target_dim)

    def forward(self, input_dict):
        """
        :params input_dict:  {'hist_item_id': tensor(...), 'hist_genre_id': tensor(...)}
        """
        # batch_size = next(iter(input_dict.values())).shape[0]
        device = next(iter(input_dict.values())).device
        
        total_emb = None

        for feat_cfg in self.feature_config_list:
            name = feat_cfg['name']
            pooling_type = feat_cfg.get('pooling', None)
            
            # Data Input
            if name not in input_dict:
                print(f"Unable to find {name} in the input dictionary, {name} has skipped")
                continue
            x = input_dict[name]
            
            # Possible x shape:
            # [Batch, Seq](Normal Sequence) -> emb -> [Batch, Seq, Dim]
            # [Batch, Seq, Tags] (Multi number sequence) -> emb -> [Batch, Seq, Tags, Dim]
            emb = self.embeddings[name](x)
            
            if name + "_proj" in self.embeddings:
                emb = self.embeddings[name + "_proj"](emb)

            # Pooling Processing
            # Transform [Batch, Seq, Tags, Dim] into [Batch, Seq, Dim
            if x.dim() == 3:
                if pooling_type == 'mean':
                    emb = torch.mean(emb, dim=2)
                elif pooling_type == 'sum':
                    emb = torch.sum(emb, dim=2)

            # Mixing all the feature
            if total_emb is None:
                total_emb = emb
            else:
                total_emb = total_emb + emb

        # total_emb shape: [Batch, Seq, Dim]
        if total_emb is None:
            raise ValueError("Configuration Error: No valid features were processed!")
        
        seq_len = total_emb.shape[1]
        positions = torch.arange(seq_len, device=device).unsqueeze(0) # [1, Seq]

        total_emb = total_emb + self.pos_emb(positions)
        
        return total_emb