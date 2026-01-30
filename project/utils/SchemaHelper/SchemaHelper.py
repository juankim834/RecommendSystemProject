from typing import List, Dict, Any
from project.utils.SchemaHelper.FeatureConfig import FeatureConfig
from copy import deepcopy
from dataclasses import fields

class SchemaHelper:
    def __init__(self, feature_configs):
        self.features = feature_configs
        self._name_index = {f.name: f for f in feature_configs}
    
    @property
    def get_all(self) -> List[FeatureConfig]:
        return self.features
    
    def get_by_name(self, name: str) -> FeatureConfig:
        return self._name_index.get(name)
    
    def filter_by_group(self, group_names: List[str]) -> List[FeatureConfig]:
        return [f for f in self.features if f.group in group_names]
    
    def filter_by_type(self, feat_type: str) -> List[FeatureConfig]:
        return [f for f in self.features if f.type == feat_type]
    
    @property
    def name_map(self) -> Dict[str, FeatureConfig]:
        """
        Used to expose name map
        """
        return self._name_index
    
    def get_customized_features(self, name:str, override_params:Dict[str, Any]=None) -> FeatureConfig:
        """
        get_customized_features:
        Get features from the pool and safely apply override params, return the FeatureConfig after override.

        :param name: Feature name
        :type name: str
        :param override_params: Params which is going to override, key is param name.
        :type override_params: Dict[str, Any]
        :return: Return FeatureConfig Object after override
        :rtype: FeatureConfig
        """
        original_config = self.get_by_name(name)
        if not original_config:
            raise KeyError(f"Feature '{name}' not found in global pool")
        config_copy = deepcopy(original_config)

        if not override_params:
            return config_copy

        # Apply override logic
        # Get dataclass valid fields set
        valid_fields = {f.name for f in fields(config_copy)}

        for k, v in override_params.items():
            if k in valid_fields:
                # Use setattr to plug in if param has been defined by dataclass
                setattr(original_config, k, v)
            else:
                # If it has not been in dataclass, plug into extra_params
                original_config.extra_params[k] = v

        return config_copy