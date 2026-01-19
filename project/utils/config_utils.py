
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
                    "output_dims": 64,
                    "dropout": 0.1,
                    "embedding_dim": 32,
                    
                    # Transformer
                    "transformer_parameters": {
                        "max_seq_len": 20,
                        "n_head": 4,
                        "n_layers": 3,
                        "FFN_dim": 256,
                        "dropout": 0.15
                    },

                    "sparse_features": [
                        {"name": "user_id_enc", "vocab_size": 6060, "embedding_dim": 32},
                        {"name": "gender_enc", "vocab_size": 3, "embedding_dim": 4},
                        {"name": "age_enc", "vocab_size": 9, "embedding_dim": 8},
                        {"name": "occupation_enc", "vocab_size": 22, "embedding_dim": 8},
                        {"name": "zip_enc", "vocab_size": 685, "embedding_dim": 16},
                        {"name": "year_enc", "vocab_size": 152, "embedding_dim": 8},
                        {"name": "rating_month", "vocab_size": 13, "embedding_dim": 4},
                        {"name": "rating_weekday", "vocab_size": 8, "embedding_dim": 4},
                        {"name": "rating_hour", "vocab_size": 25, "embedding_dim": 4}
                    ],
                    "dense_features": [
                        {"name": "user_activity_log", "dim": 1, "embedding_dim": 8}
                    ],
                    "sequence_features": [
                        {"name": "hist_movie_ids", "vocab_size": 3500, "embedding_dim": 32, "padding_idx": 0},
                        {"name": "hist_genre_ids", "vocab_size": 25, "embedding_dim": 8, "padding_idx": 0, "pooling": "mean"}
                    ]
                },

                # === Item Tower Config ===
                "item_tower": {
                    "mlp_hidden_dim": [256, 128],
                    "output_dims": 64,
                    "dropout": 0.1,
                    "embedding_dim": 32,
                    
                    # Transformer (Added based on your YAML)
                    "transformer_parameters": {
                        "max_seq_len": 3,
                        "FFN_dim": 128,
                        "n_head": 2,
                        "n_layers": 2,
                        "dropout": 0.1
                    },

                    "sparse_features": [
                        {"name": "movie_id_enc", "vocab_size": 3500, "embedding_dim": 32}
                    ],
                    "dense_features": [
                        {"name": "movie_pop_log", "dim": 1, "embedding_dim": 8},
                        {"name": "movie_avg_rate_log", "dim": 1, "embedding_dim": 8}
                    ],
                    "sequence_features": [
                        {"name": "genre_ids", "vocab_size": 25, "embedding_dim": 8, "padding_idx": 0}
                    ]
                }
            },

            # === Hard Negatives (Added based on your YAML) ===
            "hard_negatives": {
                "enabled": True,
                "num_negatives": 5,
                "field_name": "hard_neg_ids",
                "additional_features": []
            },
            
            # === Training parameters ===
            "train": {
                "batch_size": 768,
                "epochs": 50,
                "learning_rate": 0.005,
                "device": "cuda",
                "temperature": 0.1
            }
        }

    save_config(default_config, save_path)

file_loader = load_config

if __name__ == '__main__':
    generate_default_config('config.yaml')
    