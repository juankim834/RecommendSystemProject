import torch
import torch.nn as nn
import torch.nn.functional as F
from project.utils.SequenceFeatureProcessor import SequenceFeatureProcessor as pr
class SequenceEncoder(nn.Module):
    def __init__(self, 
                 feature_config_list, 
                 
                 model_dim=64,
                 dim_feedforward=4*64,
                 max_seq_len=20,
                 n_head=4,
                 n_layers=1,
                 dropout=0.1):
        super().__init__()
        self.feature_embedder = pr(feature_config_list, model_dim, max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.transformer_backbone = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=n_layers,
            enable_nested_tensor=False
        )

        self.output_norm = nn.LayerNorm(model_dim)
    
    def forward(self, input_dict):
        """
        :param input_dict: {'seq1_id': ..., 'seq2_id': ...}
        """
        main_feat_name = self.feature_embedder.feature_config_list[0]['name']
        main_seq = input_dict[main_feat_name]

        padding_mask = (main_seq == 0)

        seq_emb = self.feature_embedder(input_dict)

        context_emb = self.transformer_backbone(
            src = seq_emb, 
            src_key_padding_mask = padding_mask
        )

        context_emb = self.output_norm(context_emb)
        final_vec = self._gater_last_valid(context_emb, padding_mask)

        return final_vec

    def _gater_last_valid(self, seq_output, padding_mask):
        """
        :param seq_output: [B, L, D]
        :param padding_mask: [B, L]
        """
        valid_lengths = (~padding_mask).long().sum(dim=1)
        valid_lengths = torch.clamp(valid_lengths, min=1)

        last_valid_idx = valid_lengths - 1
        batch_indices = torch.arange(seq_output.size(0), device=seq_output.device)

        return seq_output[batch_indices, last_valid_idx, :]
    