from typing import List, Dict
from project.utils.SchemaHelper.FeatureConfig import FeatureConfig


class SchemaHelper:
    def __init__(self, feature_configs):
        self.features = feature_configs
        self._name_index = {f.name: f for f in feature_configs}
    
    def get_all(self) -> List[FeatureConfig]:
        return self.features
    
    def get_by_name(self, name: str) -> FeatureConfig:
        return self._name_index.get(name)
    
    def filter_by_group(self, group_names: List[str]) -> List[FeatureConfig]:
        return [f for f in self.features if f.group in group_names]
    
    def filter_by_type(self, feat_type: str) -> List[FeatureConfig]:
        return [f for f in self.features if f.type == feat_type]
    
    def filter_by_type(self, feat_type: str) -> List[FeatureConfig]:
        return [f for f in self.features if f.type == feat_type]
    
