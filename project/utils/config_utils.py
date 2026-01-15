
import yaml
import os

def load_config(path):
    """
    load YAML config
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found at: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_config(config_dict, path):
    """
    Save dictionary as YAML

    :param config_dict: Config dictionary
    :param path: Save path
    """
    dir_name = os.path.dirname(path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"Configuration saved to {path}")

def generate_default_config(save_path='config/default_config.yaml'):
    """
    Generate a standard configuration template that adapts to GenericTower and TwoTowerModel.
    """
    default_config = {
        "two_tower": {
            # === User Tower Config ===
            "user_tower": {
                "mlp_hidden_dim": [256, 128],
                "output_dims": 32,
                "dropout": 0.1,
                "embedding_dim": 32,
                
                # Transformer
                "transformer_parameters": {
                    "max_seq_len": 20,
                    "n_head": 4,
                    "n_layers": 2,
                    "FFN_dim": 256,
                    "dropout": 0.1
                },

                "sparse_features": [
                    {"name": "user_id", "vocab_size": 10000, "embedding_dim": 64, "padding_index": 0},
                    {"name": "gender", "vocab_size": 3, "embedding_dim": 64},
                    {"name": "age_bucket", "vocab_size": 10, "embedding_dim": 64}
                ],
                "dense_features": [
                    {"name": "age_norm", "dim": 1, "embedding_dim": 64}
                ],
                "sequence_features": [
                    {"name": "click_history_seq", "vocab_size": 5000, "embedding_dim": 64, "padding_idx": 0}
                ]
            },

            # === Item Tower Config ===
            "item_tower": {
                "mlp_hidden_dim": [256, 128],
                "output_dims": 64,
                "dropout": 0.1,
                "embedding_dim": 64,


                "sparse_features": [
                    {"name": "item_id", "vocab_size": 5000, "embedding_dim": 64},
                    {"name": "category_id", "vocab_size": 100, "embedding_dim": 64}
                ],
                "dense_features": [
                    {"name": "price_norm", "dim": 1, "embedding_dim": 64}
                ],
                "sequence_features": []
            }
        },
        
        # === Training parameters ===
        "train": {
            "batch_size": 1024,
            "epochs": 10,
            "learning_rate": 1e-3,
            "device": "cuda",
            "temperature": 0.1
        }
    }

    save_config(default_config, save_path)

file_loader = load_config

if __name__ == '__main__':
    generate_default_config('config.yaml')
    