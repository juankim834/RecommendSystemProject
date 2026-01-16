import torch
from tqdm import tqdm
import torch.nn.functional as F

def to_device(data, device):
    """
    Recursively moves data to the specified PyTorch device.

    This utility function handles complex nested data structures commonly found 
    in batches (e.g., dictionaries of tensors, lists of tensors). It leaves 
    non-Tensor data types (integers, strings, None) unchanged.

    :param data: The input data. Can be a torch.Tensor, a dictionary, a list, 
                 or a nested combination of these.
    :param device: The target device (e.g., 'cpu', 'cuda', 'cuda:0', or a torch.device object).
    
    :return: The same data structure as the input, with all contained Tensors moved 
             to the specified device.
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_device(v, device) for v in data]
    else:
        return data


def train_one_epoch(model, loader, optimizer, device, scheduler=None, 
                   item_loader=None, log_every_n_batches=100, epoch=None,
                   item_id_direction=["sparse_feature", "movie_id_enc"],
                   max_grad_norm=1.0, temperature=0.1):
    """
    :param item_loader: DataLoader for all items (needed for hard negative mining)
    :param log_every_n_batches: Log embedding stats every N batches
    :param epoch: Current epoch number (for logging)
    :param item_id_direction: Path to item IDs in batch
    :param max_grad_norm: Maximum gradient norm for clipping
    """
    model.train()
    total_loss = 0
    
    # Pre-compute all item embeddings for hard negative mining
    all_item_embs = None
    all_item_ids = None
    id_to_index_map = None
    
    if item_loader is not None:
        print("Pre-computing item embeddings for hard negative mining...")
        with torch.no_grad():
            all_item_embs_list = []
            all_item_ids_list = []
            for item_batch in tqdm(item_loader, desc="Computing item embeddings"):
                item_batch = to_device(item_batch, device)
                item_emb = model.item_tower(item_batch)
                all_item_embs_list.append(item_emb)
                all_item_ids_list.append(_tensor_getter(item_batch, item_id_direction))
            
            all_item_embs = torch.cat(all_item_embs_list, dim=0)
            all_item_ids = torch.cat(all_item_ids_list, dim=0)
            
            cpu_ids = all_item_ids.detach().cpu().tolist()
            id_to_index_map = {id_val: idx for idx, id_val in enumerate(cpu_ids)}
    
    pbar = tqdm(loader, desc="Training")
    
    for batch_idx, batch_data in enumerate(pbar):
        batch_data = to_device(batch_data, device)

        optimizer.zero_grad()
        user_emb, item_emb = model(batch_data)
        
        # Add hard negatives if available
        hard_neg_emb = None
        if all_item_embs is not None and 'hard_negatives' in batch_data:
            if 'ids' in batch_data['hard_negatives']:
                hard_neg_ids = batch_data['hard_negatives']['ids']
                if hard_neg_ids.sum() > 0:
                    batch_size, num_negs = hard_neg_ids.shape
                    hard_neg_ids_flat = hard_neg_ids.view(-1).cpu().tolist()
                    indices = [id_to_index_map.get(mid, 0) for mid in hard_neg_ids_flat]
                    indices_tensor = torch.tensor(indices, device=device)

                    hard_neg_emb_flat = F.embedding(indices_tensor, all_item_embs)
                    hard_neg_emb = hard_neg_emb_flat.view(batch_size, num_negs, -1)
        
        loss = model.compute_loss(user_emb, item_emb, hard_neg_emb, temperature=temperature)

        loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        
        # Log embedding statistics periodically
        if batch_idx % log_every_n_batches == 0 and batch_idx > 0:
            with torch.no_grad():
                user_std = user_emb.std(dim=0).mean().item()
                item_std = item_emb.std(dim=0).mean().item()
                user_mean_norm = user_emb.mean(dim=0).norm().item()
                item_mean_norm = item_emb.mean(dim=0).norm().item()
                
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "u_std": f"{user_std:.3f}",
                    "i_std": f"{item_std:.3f}",
                    "grad": f"{grad_norm:.2f}"
                })
        else:
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(loader)
    
    # Log epoch summary
    if epoch is not None:
        with torch.no_grad():
            print(f"\nEpoch {epoch} Training Summary:")
            print(f"  Average Loss: {avg_loss:.4f}")
            print(f"  Temperature: {model.temperature}")
    
    return avg_loss


def _tensor_getter(item_batch, dir):
    raw_dict = item_batch
    for p in dir:
        raw_dict = raw_dict.get(p, {})
    if len(raw_dict) == 0:
        raise ValueError("Item id direction wrong")
    return raw_dict


def validate(model, loader, item_loader, device, epoch, k_list=[10, 20], 
             item_id_in_tower_batch_dir=["item_tower", "sparse_feature", "movie_id_enc"],
             item_id_direction=["sparse_feature", "movie_id_enc"], log_embeddings=True):
    
    model.eval()
    total_loss = 0
    total_recall = {k: 0.0 for k in k_list}
    num_samples = 0
    pbar_validating = tqdm(loader, desc="Validating")

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

        cpu_ids = all_item_ids.detach().cpu().tolist()
        id_to_index_map = {id_val: idx for idx, id_val in enumerate(cpu_ids)}

        if log_embeddings and epoch is not None:
            emb_std = all_item_embs.std(dim=0).mean().item()
            emb_mean_norm = all_item_embs.mean(dim=0).norm().item()
            
            num_items = all_item_embs.shape[0]
            if num_items > 1000:
                sample_indices = torch.randperm(num_items)[:1000]
                sample_embs = all_item_embs[sample_indices]
                dists = torch.cdist(sample_embs, sample_embs)
            else:
                dists = torch.cdist(all_item_embs, all_item_embs)
            
            mask = ~torch.eye(dists.shape[0], dtype=torch.bool, device=dists.device)
            avg_dist = dists[mask].mean().item()
            min_dist = dists[mask].min().item()
            max_dist = dists[mask].max().item()

            del dists, mask
            if num_items > 1000:
                del sample_embs
                
            torch.cuda.empty_cache()
            
            print(f"\n{'='*70}")
            print(f"Epoch {epoch} - Item Embedding Diagnostics:")
            print(f"{'='*70}")
            print(f"  Item Embedding Std:       {emb_std:.6f}")
            print(f"  Item Embedding Mean Norm: {emb_mean_norm:.6f}")
            print(f"  Avg Pairwise Distance:    {avg_dist:.6f}")
            print(f"  Min Pairwise Distance:    {min_dist:.6f}")
            print(f"  Max Pairwise Distance:    {max_dist:.6f}")
            print(f"  Total Items:              {num_items}")
            
            # Enhanced health check
            if emb_std < 0.10:
                print(f"  CRITICAL: Embedding std is very low ({emb_std:.6f})")
                print(f"     Embeddings are collapsing! Immediate actions:")
                print(f"     1. INCREASE temperature to 0.5 or 1.0")
                print(f"     2. Ensure hard negatives are being used")
                print(f"     3. Check if LayerNorm was removed from embeddings")
                print(f"     4. Reduce dropout if > 0.3")
            elif emb_std < 0.15:
                print(f"   CAUTION: Embedding std is low ({emb_std:.6f})")
                print(f"     Monitor - should increase. If not improving:")
                print(f"     1. Increase temperature slightly")
                print(f"     2. Add more hard negatives")
            elif emb_std > 0.25:
                print(f"  HEALTHY: Embedding std is good ({emb_std:.6f})")
            else:
                print(f"    MODERATE: Embedding std is acceptable ({emb_std:.6f})")
            
            # Distance health check
            if avg_dist < 0.5:
                print(f"  CRITICAL: Avg pairwise distance too low ({avg_dist:.6f})")
                print(f"     Embeddings are too similar - model not discriminating!")
            elif avg_dist < 1.0:
                print(f"   CAUTION: Avg pairwise distance is low ({avg_dist:.6f})")
            else:
                print(f"  GOOD: Avg pairwise distance is healthy ({avg_dist:.6f})")
            
            print(f"{'='*70}\n")

        for batch_data in pbar_validating:
            batch_data = to_device(batch_data, device)

            user_emb, item_emb = model(batch_data)

            hard_neg_emb = None
            if 'hard_negatives' in batch_data and 'ids' in batch_data['hard_negatives']:
                hard_neg_ids = batch_data['hard_negatives']['ids']
                if hard_neg_ids.sum() > 0:
                    batch_size, num_negs = hard_neg_ids.shape
                    hard_neg_ids_flat = hard_neg_ids.view(-1).cpu().tolist()
                    indices = [id_to_index_map.get(mid, 0) for mid in hard_neg_ids_flat]
                    indices_tensor = torch.tensor(indices, device=device)

                    hard_neg_emb_flat = F.embedding(indices_tensor, all_item_embs)
                    hard_neg_emb = hard_neg_emb_flat.view(batch_size, num_negs, -1)
            
            loss = model.compute_loss(user_emb, item_emb, hard_neg_emb)
            total_loss += loss.item()
            targets = _tensor_getter(batch_data, item_id_in_tower_batch_dir)
            scores = torch.matmul(user_emb, all_item_embs.t())

            for k in k_list:
                _, topk_indices = torch.topk(scores, k=k, dim=1)
                pred_ids = all_item_ids[topk_indices]
                hits = (pred_ids == targets.view(-1, 1)).any(dim=1)
                total_recall[k] += hits.float().sum().item()
            
            num_samples += len(targets)
        
        acc_dict = {}
        for k in k_list:
            acc = total_recall[k] / num_samples
            acc_dict[k] = acc
    
    return total_loss / len(loader), acc_dict
