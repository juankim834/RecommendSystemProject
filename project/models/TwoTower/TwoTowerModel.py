import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoTowerModel(nn.Module):
    def __init__(self, user_tower, item_tower, temperature=0.1):
        super().__init__()
        self.user_tower = user_tower
        self.item_tower = item_tower
        self.temperature = temperature

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

        user_emb = F.normalize(user_emb, p=2, dim=1)
        item_emb = F.normalize(item_emb, p=2, dim=1)

        return user_emb, item_emb
    
    def predict(self, batch_data):
        """
        For the reasoning/evaluation phase: Returns a similarity score (Cosine Similarity).
        """
        user_emb, item_emb = self.forward(batch_data)
        
        # shape: [batch_size]
        scores = (user_emb * item_emb).sum(dim=1) 
        return scores

    def compute_loss(self, user_emb, item_emb):
        """
        Use In-batch negatives to compute loss
        
        For ith user, ith item is positive sample, the other (Batch - 1) items are negative sample is negative
        """

        if torch.isnan(user_emb).any():
            raise RuntimeError("Found NaN in User Embedding")
        if torch.isnan(item_emb).any():
            raise RuntimeError("Found NaN in Item Embedding")

        logits = torch.matmul(user_emb, item_emb.transpose(0, 1))

        logits = logits / self.temperature

        batch_size = user_emb.shape[0]
        labels = torch.arange(batch_size).to(user_emb.device)
        loss = F.cross_entropy(logits, labels)

        return loss
    
        
