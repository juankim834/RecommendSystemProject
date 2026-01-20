from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
import pandas as pd
import numpy as np
from project.utils.config_utils import file_loader

class RecommendationDataset(Dataset):

    """
    Vectorized dataset that processes entire feature columns at once
    instead of row-by-row operations.
    """

    def __init__(self, config_path, pkl_path, tower_type):
        """
        :param config_path: file path of config
        :param pkl_path: file path of pkl
        :param tower_type: String
        """
        self.config = file_loader(config_path)
        self.df = pd.read_pickle(pkl_path)
        self.tower_type = tower_type
        self.tower_config = self.config['two_tower'][tower_type]

        # Parse feature metadata
        self.feature_metadata = self._parse_feature_metadata()
        # Build feature matrices (vectorized approach)
        # This returns both matrices and the feature order used
        self.feature_matrices, self.feature_column_mapping = self._build_feature_matrices()

    def _parse_feature_metadata(self) -> Dict[str, List[Tuple[str, List[str], int, int]]]:
        """
        Parse feature configuration and return metadata for each feature type.
        
        Returns:
            Dict with keys 'sparse', 'dense', 'sequence', each containing:
            List of tuples: (feature_id, column_names, start_idx, end_idx)
        """
        metadata = {
            'sparse': [],
            'dense': [],
            'sequence': []
        }
        
        current_idx = 0
        
        # Parse sparse features
        if 'sparse_features' in self.tower_config:
            REQUIRED_SPARSE_KEYS = ['name', 'embedding_dim']
            for feat in self.tower_config['sparse_features']:
                missing = [k for k in REQUIRED_SPARSE_KEYS if k not in feat]
                if missing:
                    raise ValueError(
                        f"Sparse feature config missing keys {missing}: {feat}"
                    )
                feature_id = feat['name']
                embed_dim = feat['embedding_dim']
                
                
                # Assume column name matches feature name (can be customized)
                columns = [feature_id]
                
                metadata['sparse'].append((
                    feature_id,
                    columns,
                    current_idx,
                    current_idx + embed_dim
                ))
                current_idx += embed_dim
        
        # Parse dense features
        if 'dense_features' in self.tower_config:
            REQUIRED_DENSE_KEYS = ['name', 'embedding_dim']
            for feat in self.tower_config['dense_features']:
                missing = [k for k in REQUIRED_DENSE_KEYS if k not in feat]
                if missing:
                    raise ValueError(
                        f"Dense feature config missing keys {missing}: {feat}"
                    )

                feature_id = feat['name']
                output_dim = feat['embedding_dim']
                
                columns = [feature_id]
                
                metadata['dense'].append((
                    feature_id,
                    columns,
                    current_idx,
                    current_idx + output_dim
                ))
                current_idx += output_dim
        
        # Parse sequence features
        if 'sequence_features' in self.tower_config:
            REQUIRED_SEQ_KEYS = ['name', 'embedding_dim']
            for feat in self.tower_config['sequence_features']:
                missing = [k for k in REQUIRED_SEQ_KEYS if k not in feat]
                if missing:
                    raise ValueError(
                        f"Dequence feature config missing keys {missing}: {feat}"
                    )

                feature_id = feat['name']
                embed_dim = feat['embedding_dim']
                
                columns = [feature_id]
                
                metadata['sequence'].append((
                    feature_id,
                    columns,
                    current_idx,
                    current_idx + embed_dim
                ))
                current_idx += embed_dim
        
        return metadata

    def _build_feature_matrices(self):
        """
        Build numpy matrices for each feature type.
        This is the vectorization step - entire columns at once!
        
        Returns:
            Tuple of (matrices dict, column_mapping dict)
        """
        matrices = {}
        column_mapping = {
            'sparse': {},
            'dense': {},
            'sequence': {}
        }
        
        # Sparse features
        if self.feature_metadata['sparse']:
            sparse_data = []
            sparse_col_idx = 0
            
            for feature_id, columns, _, _ in self.feature_metadata['sparse']:
                if columns[0] not in self.df.columns:
                    raise ValueError(f"Feature '{feature_id}' column '{columns[0]}' not found in DataFrame. "
                                f"Available columns: {list(self.df.columns)}")
                
                feat_cfg = next((f for f in self.tower_config.get('sparse_features', []) 
                                if f['name'] == feature_id), None)
                
                if feat_cfg and 'pooling' in feat_cfg:
                    if 'sequence' not in matrices:
                        matrices['sequence'] = {}
                    raw_data = self.df[columns[0]].tolist()
                    # Check the first non-null element to see if it's already a list
                    if len(raw_data) > 0:
                        matrices['sequence'][feature_id] = raw_data
                    else:
                        raise AttributeError(f"No data was provided in the feature: {feature_id}")

                    column_mapping['sequence'][feature_id] = feature_id
                else:
                    col_data = self.df[columns[0]].values
                    sparse_data.append(col_data.reshape(-1, 1))
                    column_mapping['sparse'][feature_id] = sparse_col_idx
                    sparse_col_idx += 1
            
            if sparse_data:
                matrices['sparse'] = np.hstack(sparse_data)
        
        # Dense features
        if self.feature_metadata['dense']:
            dense_data = []
            dense_col_idx = 0
            
            for feature_id, columns, _, _ in self.feature_metadata['dense']:
                if columns[0] not in self.df.columns:
                    raise ValueError(f"Feature '{feature_id}' column '{columns[0]}' not found in DataFrame")
                
                col_data = self.df[columns[0]].values.astype(np.float32)
                dense_data.append(col_data.reshape(-1, 1))
                
                column_mapping['dense'][feature_id] = dense_col_idx
                dense_col_idx += 1
            
            matrices['dense'] = np.hstack(dense_data)
        
        # Sequence features
        if self.feature_metadata['sequence']:
            if 'sequence' not in matrices:
                matrices['sequence'] = {}
            
            for feature_id, columns, _, _ in self.feature_metadata['sequence']:
                if columns[0] not in self.df.columns:
                    raise ValueError(f"Feature '{feature_id}' column '{columns[0]}' not found in DataFrame")
                
                matrices['sequence'][feature_id] = self.df[columns[0]].tolist()
                # FIX: Track feature mapping
                column_mapping['sequence'][feature_id] = feature_id
        
        return matrices, column_mapping
    
    def get_feature_column_mapping(self) -> Dict[str, Dict[str, int]]:
        """
        Get the mapping of feature names to their column indices in matrices.
        
        Returns:
            Dict mapping feature type to feature_name->column_index
            Format: {
                'sparse': {'user_id_enc': 0, 'gender_enc': 1, ...},
                'dense': {'user_activity_log': 0, ...},
                'sequence': {'hist_movie_ids': 'hist_movie_ids', ...}  # Keys for dict lookup
            }
        """
        return self.feature_column_mapping
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample. Note: This still returns individual samples,
        but the embedding will be done in batches in the model.
        """
        sample = {}
        
        # Sparse features
        if 'sparse' in self.feature_matrices:
            sample['sparse'] = self.feature_matrices['sparse'][idx]
        
        # Dense features
        if 'dense' in self.feature_matrices:
            sample['dense'] = self.feature_matrices['dense'][idx]
        
        # Sequence features
        if 'sequence' in self.feature_matrices:
            sample['sequence'] = {
                feat_id: self.feature_matrices['sequence'][feat_id][idx]
                for feat_id in self.feature_matrices['sequence']
            }
        
        return sample
    
def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to batch samples efficiently.
    This is where the real vectorization happens!
    """
    batched = {}
    # Batch sparse features - stack into single tensor
    if "sparse" in batch[0]:
        sparse_list = [sample['sparse'] for sample in batch]
        batched["sparse"] = torch.tensor(np.stack(sparse_list), dtype=torch.long)

    # Batch dense features - stack into single tensor
    if "dense" in batch[0]:
        dense_list = [sample["dense"] for sample in batch]
        batched["dense"] = torch.tensor(np.stack(dense_list), dtype=torch.float32)
    
    # Pad sequences to max length in batch
    if "sequence" in batch[0]:
        batched["sequence"] = {}
        for feat_id in batch[0]["sequence"]:
            sequences = [sample["sequence"][feat_id] for sample in batch]

            max_len = max(len(seq) if isinstance(seq, list) else seq.shape[0] for seq in sequences)
            padded_seqs = []
            for seq in sequences:
                if isinstance(seq, list):
                    seq = np.array(seq)
                    # Pad if needed
                    if len(seq) < max_len:
                        if seq.ndim == 1:
                            padded = np.pad(seq, (0, max_len -  len(seq)), constant_values=0)
                        else:
                            padded = np.pad(seq, ((0, max_len - len(seq)), (0, 0)), constant_values=0)
                    else:
                        padded = seq
                    
                    padded_seqs.append(padded)
            batched['sequence'][feat_id] = torch.tensor(np.stack(padded_seqs), dtype=torch.long)
    return batched

def create_loader(config_path: str, 
                  pickle_path: str,
                  tower_type: str = 'user',
                  batch_size: Optional[int] = None,
                  shuffle: bool = True,
                  num_workers: int = 0) -> TorchDataLoader:
    """
    Factory function to create a vectorized dataloader.
    
    Args:
        config_path: Path to config.yaml
        pickle_path: Path to pickle file with dataframe
        tower_type: 'user' or 'item'
        batch_size: Batch size (if None, read from config)
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
    
    Returns:
        TorchDataLoader instance
    """
    dataset = RecommendationDataset(config_path, pickle_path, tower_type)

    if batch_size is None:
        config = file_loader(config_path)
        batch_size = config["train"]["batch_size"]

    dataloader = TorchDataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False
    )
    return dataloader

# Example usage
# if __name__ == "__main__":
#     # Create dataloader
#     dataloader = create_loader(
#         config_path='config.yaml',
#         pickle_path='data.pkl',
#         tower_type='user',
#         batch_size=512
#     )
    
#     # Get feature mapping
#     dataset = dataloader.dataset
#     feature_mapping = dataset.get_feature_index_mapping()
#     print("Feature Index Mapping:")
#     for feat_id, idx_range in feature_mapping:
#         print(f"  {feat_id}: columns {idx_range[0]} to {idx_range[1]}")
    
#     # Iterate through batches (vectorized!)
#     for batch in dataloader:
#         print("\nBatch shapes:")
#         if 'sparse' in batch:
#             print(f"  Sparse: {batch['sparse'].shape}")
#         if 'dense' in batch:
#             print(f"  Dense: {batch['dense'].shape}")
#         if 'sequence' in batch:
#             for feat_id, tensor in batch['sequence'].items():
#                 print(f"  Sequence {feat_id}: {tensor.shape}")
#         break    
