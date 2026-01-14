import torch
from tqdm import tqdm

def to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_device(v, device) for v in data]
    else:
        return data


def train_one_epoch(model, loader, optimizer, device, scheduler = None):
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc="Training")
    
    for batch_data in pbar:
        batch_data = to_device(batch_data, device)

        optimizer.zero_grad()
        user_emb, item_emb = model(batch_data)
        
        loss = model.compute_loss(user_emb, item_emb)

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(loader)

def validate(model, loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_data in loader:
            batch_data = to_device(batch_data, device)

            user_emb, item_emb = model(batch_data)
            loss = model.compute_loss(user_emb, item_emb)
            total_loss += loss.item()
            
    return total_loss / len(loader)