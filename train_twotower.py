import torch.optim as optim
from tqdm import tqdm
import pickle as pk
import torch
from torch.utils.data import DataLoader
from project.utils.DataLoader import RecommendationDataset
from project.models.TwoTower.ItemTower import ItemTower
from project.models.TwoTower.UserTower import UserTower
from project.models.TwoTower.TwoTowerModel import TwoTowerModel


BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
EPOCHS = 10
DEVICE = torch.device("cuda")

train_dataset = RecommendationDataset('./data/cleaned/train_set.pkl')
val_dataset = RecommendationDataset('./data/cleaned/val_set.pkl')

with open('./data/cleaned/encoders.pkl', 'rb') as f:
    encoders = pk.load(f)

raw_movie_count = len(encoders['movie_encoder'].classes_) + 1
BASE_YEAR = 1900
TARGET_MAX_YEAR = 2030
raw_year_vocab_size = (TARGET_MAX_YEAR - BASE_YEAR) + 2
raw_genre_count = encoders['genre_vocab_size'] + 1


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

item_emb_config = {
    'item_emb_dim': 64,
    'user_feat_dim': 64, 
    'year_emb_dim': 16,
    'month_emb_dim': 8,
    'weekday_emb_dim': 8,
    'hour_emb_dim': 8,
    'genre_emb_dim': 16,
    'output_dim': 64
}

item_tower = ItemTower(raw_movie_count, raw_year_vocab_size, item_emb_config['year_emb_dim'], raw_genre_count, item_emb_config['genre_emb_dim'], item_emb_config['item_emb_dim'], [256, 128], output_dim=64)

user_emb_config = {
    'max_seq_len': 20,
    'seq_emb_dim': 16,
    'user_id_dim': 64,
    'gender_dim': 4,
    'age_group_dim': 8,
    'occup_dim': 8,
    'zip_dim':16,
    'year_vocab_size': raw_year_vocab_size,
    'year_emb_dim': 16,
    'month_emb_dim': 8,
    'weekday_emb_dim': 8,
    'hour_emb_dim': 8,
    'user_activity_dim': 1
}

raw_user_count = len(encoders['user_encoder'].classes_) + 1
user_tower = UserTower(raw_movie_count, raw_genre_count, 64, 20, raw_user_count, user_emb_config['user_id_dim'], 3, user_emb_config['gender_dim'], len(encoders['age_encoder'].classes_)+1, user_emb_config['age_group_dim'], len(encoders['occupation_encoder'].classes_)+1, user_emb_config['occup_dim'], len(encoders['zip_encoder'].classes_)+1, user_emb_config['zip_dim'], raw_year_vocab_size, user_emb_config['year_emb_dim'], user_emb_config['month_emb_dim'], user_emb_config['weekday_emb_dim'], user_emb_config['hour_emb_dim'], user_emb_config['user_activity_dim'], [256, 128], 64, dropout=0.3)

model = TwoTowerModel(user_tower, item_tower, temperature=0.1).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    pbar = tqdm(loader, desc="Training")

    for user_inputs, item_inputs in pbar:
        user_inputs = {k: v.to(device) for k, v in user_inputs.items()}
        item_inputs = {k: v.to(device) for k, v in item_inputs.items()}

        optimizer.zero_grad()
        user_emb, item_emb = model(user_inputs, item_inputs)
        loss = model.compute_loss(user_emb, item_emb)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})

    return total_loss / len(loader)

def validate(model, loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for user_inputs, item_inputs in loader:
            user_inputs = {k: v.to(device) for k, v in user_inputs.items()}
            item_inputs = {k: v.to(device) for k, v in item_inputs.items()}

            user_emb, item_emb = model(user_inputs, item_inputs)
            loss = model.compute_loss(user_emb, item_emb)
            total_loss += loss.item()
        
        return total_loss / len(loader)
    
print(f"Start training on {DEVICE}...")

import os

for epoch in range(EPOCHS):
    train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
    val_loss = validate(model, val_loader, DEVICE)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': val_loss
    }

    torch.save(checkpoint, os.path.join('./project/models/TwoTower/', f"model_epoch_{epoch}.pth"))

    print(f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

torch.save(model.state_dict(), "final_user_tower.pth")





