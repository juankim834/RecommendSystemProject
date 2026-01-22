"""
DataLoader that creates combined user-item batches from a single DataFrame
containing both user and item features.
"""

import torch
from torch.utils.data import DataLoader as TorchDataLoader
from typing import Optional
from project.utils.DataLoader import RecommendationDataset, collate_fn
import pandas as pd


class CombinedTwoTowerDataLoader:
    """
    Creates batches containing both user and item features from a single DataFrame.
    This is for Option A where each row has user features + item features.
    """
    
    def __init__(self, config_path: str, pickle_path: str, 
                 batch_size: int = 512, shuffle: bool = True, 
                 num_workers: int = 0, hard_negatives_enabled: bool = False):
        """
        Args:
            config_path: Path to config.yaml
            pickle_path: Path to pickle file with both user and item features
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            hard_negatives_enabled: Whether hard negatives are in the data
        """

        pkl_df = pd.read_pickle(pickle_path)

        # Create two datasets - one for user features, one for item features
        self.user_dataset = RecommendationDataset(
            config_path, pkl_df, tower_type='user_tower'
        )
        self.item_dataset = RecommendationDataset(
            config_path, pkl_df, tower_type='item_tower'
        )
        
        # Verify they have the same length
        assert len(self.user_dataset) == len(self.item_dataset), \
            "User and item datasets must have same length"
        
        # Store feature mappings
        self.user_mapping = self.user_dataset.get_feature_column_mapping()
        self.item_mapping = self.item_dataset.get_feature_column_mapping()
        
        # Create underlying dataloader (we'll use user_dataset as base)
        self.dataloader = TorchDataLoader(
            dataset=list(range(len(self.user_dataset))),  # Just indices
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._combined_collate_fn,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        self.hard_negatives_enabled = hard_negatives_enabled
    
    def _combined_collate_fn(self, indices):
        """
        Custom collate function that combines user and item batches.
        
        Args:
            indices: List of sample indices
        
        Returns:
            Combined batch dict with 'user_tower' and 'item_tower' keys
        """
        # Get user samples
        user_samples = [self.user_dataset[idx] for idx in indices]
        user_batch = collate_fn(user_samples)
        
        # Get item samples
        item_samples = [self.item_dataset[idx] for idx in indices]
        item_batch = collate_fn(item_samples)
        
        # Combine into single batch
        combined_batch = {
            'user_tower': user_batch,
            'item_tower': item_batch
        }
        
        # TODO: Add hard negatives if enabled
        if self.hard_negatives_enabled:
            # You'll need to implement this based on your hard negative format
            # combined_batch['hard_negatives'] = ...
            pass
        
        return combined_batch
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)
    
    def get_feature_mappings(self):
        """Get both user and item feature mappings."""
        return {
            'user': self.user_mapping,
            'item': self.item_mapping
        }


def create_combined_dataloader(config_path: str, 
                               pickle_path: str,
                               batch_size: Optional[int] = None,
                               shuffle: bool = True,
                               num_workers: int = 0,
                               hard_negatives_enabled: bool = False):
    """
    Factory function to create a combined user-item dataloader.
    
    Args:
        config_path: Path to config.yaml
        pickle_path: Path to pickle file with both user and item features
        batch_size: Batch size (if None, read from config)
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        hard_negatives_enabled: Whether hard negatives are enabled
    
    Returns:
        CombinedTwoTowerDataLoader instance
    """
    if batch_size is None:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        batch_size = config['train']['batch_size']
    
    return CombinedTwoTowerDataLoader(
        config_path=config_path,
        pickle_path=pickle_path,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        hard_negatives_enabled=hard_negatives_enabled
    )


# Example usage
# if __name__ == "__main__":
#     # Create combined dataloader
#     train_loader = create_combined_dataloader(
#         config_path='config.yaml',
#         pickle_path='./data/cleaned/train_set.pkl',
#         batch_size=512,
#         shuffle=True
#     )
    
#     # Get feature mappings
#     mappings = train_loader.get_feature_mappings()
#     print("User feature mapping:", mappings['user']['sparse'])
#     print("Item feature mapping:", mappings['item']['sparse'])
    
#     # Iterate through batches
#     for batch in train_loader:
#         print("\nBatch structure:")
#         print("  User tower keys:", batch['user_tower'].keys())
#         print("  Item tower keys:", batch['item_tower'].keys())
        
#         if 'sparse' in batch['user_tower']:
#             print(f"  User sparse shape: {batch['user_tower']['sparse'].shape}")
#         if 'sparse' in batch['item_tower']:
#             print(f"  Item sparse shape: {batch['item_tower']['sparse'].shape}")
        
#         break  # Just show first batch