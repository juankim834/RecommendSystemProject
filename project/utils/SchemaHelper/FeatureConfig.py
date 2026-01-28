from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Literal

import yaml

@dataclass
class FeatureConfig:
    # Inherent parameters
    name: str
    type: Literal["sparse", "dense", "sequence"]
    vocab_size: int = 0

    group: str = "default"

    # Model Hyperparameters
    embedding_dim: int = 32
    padding_idx: int = 0
    pooling: Any = None
    extra_params: Dict[str, Any] = field(default_factory=dict)


    def __repr__(self):
        return f"<Feature '{self.name}': Dim={self.embedding_dim} -> [{self.target_module}]>"