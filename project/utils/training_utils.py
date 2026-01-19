import torch
from tqdm import tqdm
import torch.nn.functional as F

def to_device(data, device):
    """
    Recursively move data to device, handling nested structures
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {key: to_device(value, device) for key, value in data.items()}
    elif isinstance(data, list):  # NEW: Handle list of dicts
        return [to_device(item, device) for item in data]
    else:
        return data


def train_one_epoch(model, loader, optimizer, device, scheduler=None, 
                   log_every_n_batches=100, epoch=None,
                   max_grad_norm=1.0, temperature=0.1, 
                   item_id_feature='movie_id_enc', item_id_type='sparse'):
    
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc=f"Training Epoch {epoch}")
    
    for batch_idx, batch_data in enumerate(pbar):
        batch_data = to_device(batch_data, device)  # Make sure to_device handles lists!

        optimizer.zero_grad()
        
        # Forward pass - now returns 3 values including hard_neg_emb
        user_emb, pos_item_emb, hard_neg_emb = model(batch_data)

        current_batch_item_ids  = extract_item_id(
            batch_data['item_tower'], 
            feature_name=item_id_feature, 
            feature_type=item_id_type
        )
        
        # Compute loss
        loss = model.compute_loss(
            user_emb, 
            pos_item_emb, 
            hard_neg_emb=hard_neg_emb,  
            item_ids=current_batch_item_ids, 
            temperature=temperature
            )

        loss.backward()
        
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        
        if batch_idx % log_every_n_batches == 0:
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.6f}"
            })

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch} finished. Avg Loss: {avg_loss:.4f}")
    return avg_loss

def extract_item_id(item_batch, feature_name = 'movie_id_enc', feature_type = 'sparse', item_id_col = 0):
    """
    Extract item IDs from the new vectorized batch format.
    
    Args:
        item_batch: Batch dict with 'sparse', 'dense', 'sequence' keys
        feature_name: Name of the feature containing item IDs (e.g., 'movie_id_enc')
        feature_type: Type of feature - 'sparse', 'dense', or 'sequence'
    
    Returns:
        Tensor of item IDs with shape (batch_size,) or (batch_size, 1)
    """
    if feature_type == 'sparse':
        # For sparse features, we need to know which column index
        # This assumes movie_id_enc is the first sparse feature (index 0)
        sparse_matrix = item_batch.get('sparse')
        if sparse_matrix is not None:
            # Assuming item ID is the first column (index 0)
            # Adjust this based on your actural feature ordering
            return sparse_matrix[:, item_id_col]
        elif feature_type == 'dense':
            dense_matrix = item_batch.get('dense')
            if dense_matrix is not None:
                return dense_matrix[:, 0]
        
        elif feature_type == 'sequence':
            seq_dict = item_batch.get('sequence', {})
            if feature_name in seq_dict:
                return seq_dict[feature_name][:, 0]
        raise ValueError(f"Could not extract item ID '{feature_name}' from batch")

def validate(model, loader, item_loader, device, epoch, k_list=[10, 20], 
             item_id_feature='movie_id_enc',
             item_id_type='sparse',
             item_id_col_idx=0, 
             log_embeddings=True):
    """
    Validation with new vectorized format.
    
    Args:
        model: Two-tower model
        loader: Validation dataloader (returns user + item batches)
        item_loader: Item-only dataloader for building item index
        device: torch device
        epoch: Current epoch number
        k_list: List of K values for Recall@K
        item_id_feature: Name of the item ID feature
        item_id_type: 'sparse', 'dense', or 'sequence'
        item_id_col_idx: Column index of item ID in the feature matrix
        log_embeddings: Whether to log embedding statistics
    """
    model.eval()
    total_loss = 0
    total_recall = {k: 0.0 for k in k_list}
    num_samples = 0
    print("Pre-computing all item embeddings for Validation...")
    all_item_embs_list = []
    all_item_ids_list = []
    
    with torch.no_grad():
        for item_batch in tqdm(item_loader, desc="Indexing Items"):
            item_batch = to_device(item_batch, device)
            # Here, `item_batch` is a standard Tower input.
            item_emb = model.get_item_embeddings(item_batch) 
            
            all_item_embs_list.append(item_emb)
            
            # Extract the ID for subsequent matching.
            if item_id_type == 'sparse' and 'sparse' in item_batch:
                ids = item_batch['sparse'][:, item_id_col_idx]
            elif item_id_type == 'dense' and 'dense' in item_batch:
                ids = item_batch['dense'][:, item_id_col_idx]
            elif item_id_type == 'sequence' and 'sequence' in item_batch:
                ids = item_batch['sequence'][item_id_feature][:, 0]
            else:
                raise ValueError(f"Cannot extract item IDs from batch with type'{item_id_type}'")
            all_item_ids_list.append(ids)
            
        all_item_embs = torch.cat(all_item_embs_list, dim=0) # [Total_Items, Emb_Dim]
        all_item_ids = torch.cat(all_item_ids_list, dim=0)   # [Total_Items, 1]
    
    pbar = tqdm(loader, desc="Validating")
    
    with torch.no_grad():
        if log_embeddings and epoch is not None:
            _log_embedding_stats(all_item_embs, epoch)

        for batch_data in pbar:
            batch_data = to_device(batch_data, device)
            # Forward pass
            user_emb, pos_item_emb, hard_neg_emb = model(batch_data)

            # Extract target item IDs from the batch
            # Assuming batch_data has both user and item features combined
            # You'll need to adapt this based on how your model structures the batch
            item_batch = batch_data.get('item_tower', {})

            if not item_batch:
                raise ValueError("batch_data does not contain 'item_tower' key")

            if item_id_type == 'sparse' and 'sparse' in item_batch:
                targets = item_batch['sparse'][:, item_id_col_idx]
            elif item_id_type == 'dense' and 'dense' in item_batch:
                targets = item_batch['dense'][:, item_id_col_idx]
            elif item_id_type == 'sequence' and 'sequence' in item_batch:
                targets = item_batch['sequence'][item_id_feature][:, 0]
            else:
                raise ValueError(f"Cannot extract target item IDs from batch. "
                               f"item_batch keys: {item_batch.keys()}, "
                               f"looking for type: {item_id_type}")

            

            # Compute loss
            loss = model.compute_loss(
                user_emb, 
                pos_item_emb, 
                hard_neg_emb=hard_neg_emb, 
                item_ids=targets
            )
            total_loss += loss.item()
            # Recall Rate
            scores = torch.matmul(user_emb, all_item_embs.t())
            for k in k_list:
                _, topk_indices = torch.topk(scores, k=k, dim=1)
                # Retrieve Item ID by Index
                pred_ids = all_item_ids[topk_indices] # [Batch, K, 1]
                
                # Simple Hit Detection
                # targets: [Batch, 1] -> [Batch, 1, 1]
                hits = (pred_ids == targets.unsqueeze(1)).any(dim=1)
                total_recall[k] += hits.float().sum().item()
            
            
            num_samples += len(targets)

    avg_loss = total_loss / len(loader)
    acc_dict = {k: total_recall[k] / num_samples for k in k_list}
    
    print(f"\nValidation Result - Loss: {avg_loss:.4f}")
    for k, acc in acc_dict.items():
        print(f"Recall@{k}: {acc:.4f}")
        
    return avg_loss, acc_dict

def _log_embedding_stats(all_item_embs, epoch):

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
