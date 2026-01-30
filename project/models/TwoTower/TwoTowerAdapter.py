from typing import Dict, List, Any
from project.utils.SchemaHelper.FeatureConfig import FeatureConfig
from project.utils.SchemaHelper.SchemaHelper import SchemaHelper

class TwoTowerSchemaHelper:
    """
    Use to adapt the shemahelper to twotower model
    """
    def __init__(self, schema_helper: SchemaHelper):
        """
        :param SchemaHelper: 
        """
        self.schema = schema_helper

    

    def build_config(self, recipe:Dict[str, Any]) -> Dict[str, List[FeatureConfig]]:
        """
        Use Recipe to construct FeatureConfig dict

        :params recipe: Example:
        {
            "user_tower": ["user_id", {"name": "gender", "embedding_dim": 8}],
            "item_tower": ["item_id", "category"]
        }
        
        """
        final_config = {}
        for tower_name, feature_list in recipe.items():
            tower_configs = []

            for item in feature_list:
                # Read Recipe
                if isinstance(item, str):
                    feat_name = item
                    overrides = {}
                elif isinstance(item, dict):
                    feat_name = item.get("name")
                    if not feat_name:
                        raise ValueError(f"Recipe item in {tower_name} missing 'name'.")
                    # Get all the overrides parameters, except for 'name' field in the dictionary
                    overrides = {k: v for k, v in item.items() if k != "name"}
                else:
                    raise TypeError(f"Invalid format in {tower_name}: {item}")
                
                # Get the FeatureConfig Object after overriding
                try:
                    feature_obj = self.schema.get_customized_features(feat_name, overrides)
                    tower_configs.append(feature_obj)
                except KeyError as e:
                    raise KeyError(f"Error building {tower_name}: {str(e)}")
                
            final_config[tower_name] = tower_configs

        return final_config