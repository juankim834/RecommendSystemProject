import torch
import torch.nn as nn
import torch.nn.functional as F
from Tower import MLP_Tower
from SequenceEncoder import SequenceEncoder

class TwoTowerModel(nn.Module):
    def __init__(self, user_tower, item_tower, temperature=0.1):
        super().__init__()
        self.user_tower = user_tower
        self.item_tower = item_tower
        self.temperature = temperature

    def forward(self, user_input, item_inputs):
        """
        Args:
            user_inputs (dict): All parameters of UserTower forward
            item_inputs (dict): All parameters of ItemTower forward
        """
        user_emb = self.user_tower(**user_input)
        item_emb = self.item_tower(**item_inputs)

        user_emb = F.normalize(user_emb, p=2, dim=1)
        item_emb = F.normalize(item_emb, p=2, dim=1)

        return user_emb, item_emb

    def comput_loss(self, user_emb, item_emb):
        """
        Use In-batch negatives to compute loss
        
        For ith user, ith item is positive sample, the other (Batch - 1) items are negative sample is negative
        """

        logits = torch.matmul(user_emb, item_emb.transpose(0, 1))

        logits = logits / self.temperature

        batch_size = user_emb.shape[0]
        labels = torch.arange(batch_size).to(user_emb.device)
        loss = F.cross_entropy(logits, labels)

        return loss
    
        
