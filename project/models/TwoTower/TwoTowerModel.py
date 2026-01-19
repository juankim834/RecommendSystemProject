import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoTowerModel(nn.Module):
    def __init__(self, user_tower, item_tower, 
                 user_feature_mapping=None, 
                 item_feature_mapping=None):
        """
        Args:
            user_tower: GenericTower for user side
            item_tower: GenericTower for item side
            user_feature_mapping: Feature column mapping for user tower (optional)
            item_feature_mapping: Feature column mapping for item tower (optional)
        """
        super().__init__()
        self.user_tower = user_tower
        self.item_tower = item_tower
        
        # Store feature mappings (can be set later if not provided)
        self.user_feature_mapping = user_feature_mapping
        self.item_feature_mapping = item_feature_mapping

    def set_feature_mappings(self, user_mapping, item_mapping):
        """
        Set feature mappings after initialization (useful if dataloaders created after model)
        
        Args:
            user_mapping: User feature column mapping from dataloader
            item_mapping: Item feature column mapping from dataloader
        """
        self.user_feature_mapping = user_mapping
        self.item_feature_mapping = item_mapping

    def forward(self, batch_data):
        """
        Args:
            batch_data (dict): Contains 'user_tower' and 'item_tower' inputs
                Format: {
                    'user_tower': {'sparse': tensor, 'dense': tensor, 'sequence': {...}},
                    'item_tower': {'sparse': tensor, 'dense': tensor, 'sequence': {...}},
                    'hard_negatives': [list of item batches] (optional)
                }
        """
        user_inputs = batch_data["user_tower"] 
        item_inputs = batch_data["item_tower"]

        # Pass mappings to towers
        user_emb = self.user_tower(user_inputs, self.user_feature_mapping)
        item_emb = self.item_tower(item_inputs, self.item_feature_mapping)
        
        # Handle hard negatives
        hard_neg_emb = None
        if 'hard_negatives' in batch_data and batch_data['hard_negatives']:
            hard_neg_embs_list = []
            for neg_sample in batch_data['hard_negatives']:
                # Each hard negative also needs the item mapping
                neg_emb = self.item_tower(neg_sample, self.item_feature_mapping)
                hard_neg_embs_list.append(neg_emb)
            hard_neg_emb = torch.stack(hard_neg_embs_list, dim=1)

        return user_emb, item_emb, hard_neg_emb
    
    def predict(self, batch_data):
        """
        For the reasoning/evaluation phase: Returns a similarity score (Cosine Similarity).
        """
        user_emb, item_emb, _ = self.forward(batch_data)
        
        # shape: [batch_size]
        scores = (user_emb * item_emb).sum(dim=1) 
        return scores
    
    def get_item_embeddings(self, item_inputs):
        """
        Get embeddings for item inputs (used in validation for indexing all items)
        """
        item_emb = self.item_tower(item_inputs, self.item_feature_mapping)
        return item_emb

    def compute_loss(self, user_emb, item_emb, item_ids=None, hard_neg_emb=None, temperature=0.1):
        """
        Use In-batch negatives to compute loss
        
        For ith user, ith item is positive sample, the other (Batch - 1) items are negative sample
        """

        if torch.isnan(user_emb).any():
            raise RuntimeError("Found NaN in User Embedding")
        if torch.isnan(item_emb).any():
            raise RuntimeError("Found NaN in Item Embedding")
        
        batch_size = user_emb.shape[0]

        in_batch_logits = torch.matmul(user_emb, item_emb.transpose(0, 1))
        in_batch_logits = in_batch_logits / temperature
        
        if item_ids is not None:
            # Make sure ids tensor dim=1
            item_ids = item_ids.view(-1) 
            
            # Find same id pair in batch
            # [B, 1] == [1, B] -> [B, B]
            collision_mask = (item_ids.unsqueeze(1) == item_ids.unsqueeze(0))
            
            # Exclude self from masking
            # eye matrix: diagonal is True
            eye = torch.eye(batch_size, device=user_emb.device).bool()
            
            # What we need to mask out are: entries with the same ID but that are not themselves (off-diagonal collisions)
            collision_mask = collision_mask & (~eye)
            
            # Set the logits at the conflicting positions to -inf (resulting in a probability of 0 after Softmax)
            in_batch_logits = in_batch_logits.masked_fill(collision_mask, -1e9)

        if hard_neg_emb is not None:
            if torch.isnan(hard_neg_emb).any():
                raise RuntimeError("Found NaN in Hard Negative Embedding")
            
            assert hard_neg_emb.dim() == 3, f"Expected shape [B, N, D], got {hard_neg_emb.shape}"
            assert hard_neg_emb.size(0) == batch_size, "Batch size mismatch"
            
            # hard_neg_emb: [B, N, D]
            # user_emb:     [B, D] -> [B, 1, D]
            # result:       [B, 1, N] -> [B, N]
            # Hard negative logits
            hard_neg_logits = torch.bmm(
                user_emb.unsqueeze(1),
                hard_neg_emb.transpose(1, 2)
            ).squeeze(1)
            hard_neg_logits = hard_neg_logits / temperature

            # Combine all logits
            logits = torch.cat([in_batch_logits, hard_neg_logits], dim=1)
        else:
            logits = in_batch_logits

        # Labels: positive item is at index i for user i
        labels = torch.arange(batch_size, device=user_emb.device)
        loss = F.cross_entropy(logits, labels)

        with torch.no_grad():
            # Diagnostic: Check if positive scores are higher than negative scores
            pos_scores = logits[torch.arange(batch_size), labels]  # Diagonal
            neg_scores_mean = (logits.sum(dim=1) - pos_scores) / (logits.size(1) - 1)
            
            # Optionally log these for debugging
            # print(f"Pos score mean: {pos_scores.mean():.3f}, Neg score mean: {neg_scores_mean.mean():.3f}")

        return loss
    
        
