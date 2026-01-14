import torch.optim as optim
from tqdm import tqdm
import torch
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from project.utils.DataLoader import RecommendationDataset
from project.utils.training_utils import to_device, train_one_epoch, validate
from project.models.TwoTower.GenericTower import GenericTower
from project.models.TwoTower.TwoTowerModel import TwoTowerModel
from project.utils.config_utils import load_config
import os


CONFIG_PATH = "config.yaml"
PKL_PATH_TRAIN = "./data/cleaned/train_set.pkl"
PKL_PATH_VAL = "./data/cleaned/val_set.pkl"

cfg = load_config(CONFIG_PATH)
epochs = cfg.get("train", {}).get("epochs", 10)
learning_rate = cfg.get("train", {}).get("learning_rate", 5e-4)
batch_size = cfg.get("train", {}).get("batch_size", 1024)
temperature = cfg.get("train", {}).get("temperature", 0.07)
DEVICE = cfg.get("train", {}).get("device", "cuda")
if torch.cuda.is_available() == False and DEVICE == "cuda":
    raise RuntimeError("Cuda is not available")
user_tower = GenericTower(cfg, "user_tower")
item_tower = GenericTower(cfg, "item_tower")

model = TwoTowerModel(user_tower, item_tower, temperature=temperature).to(DEVICE)

train_dataset = RecommendationDataset(CONFIG_PATH, PKL_PATH_TRAIN)
val_dataset = RecommendationDataset(CONFIG_PATH, PKL_PATH_VAL)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1, persistent_workers=True)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

total_step = len(train_loader) * epochs
warmup_step = int(total_step * 0.1)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_training_steps=total_step, num_warmup_steps=warmup_step)

print(f"Start training on {DEVICE}...")
if __name__ == '__main__':
    patience = 5
    min_delta = 1e-4
    best_val_loss = float('inf')
    patience_counter = 0
    save_dir = './project/models/TwoTower/'
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE, scheduler=scheduler)
        val_loss = validate(model, val_loader, DEVICE)

        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss
            }
            torch.save(checkpoint, os.path.join(save_dir, "best_model.pth"))
            print(f"  Validation loss improved. Model saved.")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    torch.save(model.state_dict(), os.path.join(save_dir, "final_full_model.pth"))