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

def _tensor_getter(item_batch, dir):
    raw_dict = item_batch
    for p in dir:
        raw_dict = raw_dict.get(p, {})
    if len(raw_dict) == 0:
        raise ValueError("Item id direction wrong")
    return raw_dict

def validate(model, loader, item_loader, device, k_list=[10, 20], 
             item_id_in_tower_batch_dir=["item_tower", "sparse_feature", "movie_id_enc"],
             item_id_direction=["sparse_feature", "movie_id_enc"]):
    model.eval()
    total_loss = 0
    total_recall = {k: 0.0 for k in k_list}
    num_batches = 0
    pbar_validating = tqdm(loader, desc="Validating and calculating Recall@K")

    with torch.no_grad():
        all_item_embs_list = []
        all_item_ids_list = []
        for item_batch in item_loader:
            item_batch = to_device(item_batch, device)
            item_emb = model.item_tower(item_batch)
            all_item_embs_list.append(item_emb)
            all_item_ids_list.append(_tensor_getter(item_batch, item_id_direction))
        all_item_embs = torch.cat(all_item_embs_list, dim=0)
        all_item_ids = torch.cat(all_item_ids_list, dim=0)

        for batch_data in pbar_validating:
            batch_data = to_device(batch_data, device)

            user_emb, item_emb = model(batch_data)
            loss = model.compute_loss(user_emb, item_emb)
            total_loss += loss.item()
            targets = _tensor_getter(batch_data, item_id_in_tower_batch_dir)
            scores = torch.matmul(user_emb, all_item_embs.t())

            assert all_item_embs.size(0) == all_item_ids.size(0)
            assert targets.ndim == 1
            assert all_item_ids.dtype == targets.dtype

            for k in k_list:
                _, topk_indices = torch.topk(scores, k=k, dim=1)
                pred_ids = all_item_ids[topk_indices]
                hits = (pred_ids == targets.view(-1, 1)).any(dim=1)
                total_recall[k] += hits.float().sum().item()
            
            num_batches += len(targets)
        
        acc_dict = {}
        for k in k_list:
            acc = total_recall[k] / num_batches
            acc_dict[k] = acc
    return total_loss / len(loader), acc_dict