import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoTowerModel(nn.Module):
    def __init__(self, user_tower, item_tower):
        super().__init__()
        self.user_tower = user_tower
        self.item_tower = item_tower

    def forward(self, batch_data):
        """
        Args:
            user_inputs (dict): All parameters of UserTower forward
            item_inputs (dict): All parameters of ItemTower forward
        """
        user_inputs = batch_data["user_tower"] 
        item_inputs = batch_data["item_tower"]

        user_emb = self.user_tower(user_inputs)
        item_emb = self.item_tower(item_inputs)

        return user_emb, item_emb
    
    def predict(self, batch_data):
        """
        For the reasoning/evaluation phase: Returns a similarity score (Cosine Similarity).
        """
        user_emb, item_emb = self.forward(batch_data)
        
        # shape: [batch_size]
        scores = (user_emb * item_emb).sum(dim=1) 
        return scores
    
    def get_item_embeddings(self, item_ids):

        """
        Get item embeddings for given item IDs (for hard negatives)
        
        Args:
            item_ids: tensor of shape [N] containing item IDs
            
        Returns:
            item_emb: tensor of shape [N, emb_dim]
        """

        item_inputs = {
            'sparse_feature': {
                'movie_id_enc': item_ids
            },
            'dense_feature': {},
            'seq_feature': {}
        }
        item_emb = self.item_tower(item_inputs)
        return item_emb

    def compute_loss(self, user_emb, item_emb, hard_neg_emb=None, temperature = 0.1):
        """
        Use In-batch negatives to compute loss
        
        For ith user, ith item is positive sample, the other (Batch - 1) items are negative sample is negative
        """

        if torch.isnan(user_emb).any():
            raise RuntimeError("Found NaN in User Embedding")
        if torch.isnan(item_emb).any():
            raise RuntimeError("Found NaN in Item Embedding")
        
        batch_size = user_emb.shape[0]
        
        if hard_neg_emb is not None:
            if torch.isnan(hard_neg_emb).any():
                raise RuntimeError("Found NaN in Hard Negative Embedding")
            
            assert hard_neg_emb.dim() == 3, f"Expected shape [B, N, D], got {hard_neg_emb.shape}"
            assert hard_neg_emb.size(0) == batch_size, "Batch size mismatch"
            
            in_batch_logits = torch.matmul(user_emb, item_emb.transpose(0, 1))
            
            hard_neg_logits = torch.bmm(
                user_emb.unsqueeze(1),
                hard_neg_emb.transpose(1, 2)
            ).squeeze(1)
            logits = torch.cat([in_batch_logits, hard_neg_logits], dim=1)
        else:
            logits = torch.matmul(user_emb, item_emb.transpose(0, 1))

        logits = logits / temperature

        batch_size = user_emb.shape[0]
        labels = torch.arange(batch_size).to(user_emb.device)
        loss = F.cross_entropy(logits, labels)

        # with torch.no_grad():
        #     # Check if positive scores are higher than negative scores
        #     pos_scores = logits[torch.arange(batch_size), labels]  # Diagonal
        #     neg_scores_mean = (logits.sum(dim=1) - pos_scores) / (logits.size(1) - 1)
            
        #     # Helpful for debugging - you can log these
        #     # print(f"Pos score mean: {pos_scores.mean():.3f}, Neg score mean: {neg_scores_mean.mean():.3f}")

        return loss
    
        
