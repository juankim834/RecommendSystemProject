import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import pandas as pd

from project.utils.DataLoader import create_loader
from project.utils.CombineTwoTower import CombinedTwoTowerDataLoader
from project.utils.CombineTwoTower import create_combined_dataloader
from project.models.TwoTower.GenericTower import GenericTower
from project.models.TwoTower.TwoTowerModel import TwoTowerModel
from project.utils.training_utils import train_one_epoch, validate, build_user_history
from project.utils.config_utils import file_loader
# from model_diagnostics import run_full_diagnostics


def main():
    """
    Main training function for Option A: 
    Single DataFrame with both user and item features in each row.
    """
    # Configuration
    config_path = 'config.yaml'
    train_data_path = './data/cleaned/train_set.pkl'
    val_data_path = './data/cleaned/val_set.pkl'
    item_data_path = './data/cleaned/item_set.pkl'
    
    config = file_loader(config_path)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup dataloaders for TRAINING
    print("Setting up training dataloader...")
    train_loader = CombinedTwoTowerDataLoader(
        config_path=config_path,
        pickle_path=train_data_path,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=1
    )
    
    # Setup dataloaders for VALIDATION
    print("Setting up validation dataloader...")
    val_loader = CombinedTwoTowerDataLoader(
        config_path=config_path,
        pickle_path=val_data_path,
        batch_size=config['train']['batch_size'],
        shuffle=False,
        num_workers=1
    )
    
    # Set up user's history for computing recall@K
    train = pd.read_pickle(train_data_path)

    
    # Setup ITEM INDEX dataloader (for computing Recall@K)
    # This extracts only item features for building the item catalog
    print("Setting up item index dataloader...")
    item_loader = create_loader(
        config_path=config_path,
        pickle_path=item_data_path,
        tower_type='item_tower',
        batch_size=config['train']['batch_size'],
        shuffle=False,
        num_workers=1
    )
    
    val_metadata_loader = create_combined_dataloader(
        config_path='metadata_config.yaml',  # Minimal config
        pickle_path=val_data_path,  # Same pickle file
        batch_size=config['train']['batch_size'],  # MUST match val_loader
        shuffle=False
    )

    metadata_config = file_loader('metadata_config.yaml').get('two_tower')
    meta_item = metadata_config.get('item_tower').get('metadata_fields')
    meta_user = metadata_config.get('user_tower').get('metadata_fields')

    user_history = build_user_history(
        train,
        user_col=meta_user,
        item_col=meta_item
    )

    # Get feature mappings
    mappings = train_loader.get_feature_mappings()
    user_mapping = mappings['user']
    item_mapping = mappings['item']
    
    print(f"\nFeature mappings:")
    print(f"  User sparse features: {list(user_mapping['sparse'].keys())}")
    print(f"  Item sparse features: {list(item_mapping['sparse'].keys())}")
    
    # Create model
    print("\nCreating model...")
    user_tower = GenericTower(config, 'user_tower')
    item_tower = GenericTower(config, 'item_tower')
    
    model = TwoTowerModel(
        user_tower=user_tower,
        item_tower=item_tower,
        user_feature_mapping=user_mapping,
        item_feature_mapping=item_mapping
    ).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=config['train']['learning_rate'])
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, 
    #     T_max=config['train']['epochs']
    # )
    
    # Training parameters
    num_epochs = config['train']['epochs']
    temperature = config['train']['temperature']
    patience = config['train'].get('patience', 8)
    
    best_recall = 0.0
    patience_counter = 0

    # run_full_diagnostics(
    #     model=model,
    #     train_loader=train_loader,
    #     config=config,
    #     device=device
    # )
    
    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Temperature: {temperature}, Patience: {patience}")
    print(f"{'='*70}\n")
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*70}")
        
        # TRAINING
        avg_train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            # scheduler=scheduler,
            log_every_n_batches=100,
            epoch=epoch,
            temperature=temperature
        )
        
        # VALIDATION
        # Get the column index for movie_id_enc from the mapping
        movie_id_col_idx = item_mapping['sparse'].get('movie_id_enc', 0)
        
        avg_val_loss, metrics = validate(
            model=model,
            loader=val_loader,
            item_loader=item_loader,
            meta_data_loader=val_metadata_loader,
            device=device,
            epoch=epoch,
            k_list=[10, 20, 50],
            item_id_feature='movie_id_enc',
            item_id_type='sparse',
            item_id_col_idx=movie_id_col_idx,
            log_embeddings=True, 
            user_history=user_history
        )
        
        # Check for improvement
        current_recall = metrics[10]  # Use Recall@10 as primary metric
        
        if current_recall > best_recall:
            best_recall = current_recall
            patience_counter = 0
            
            # Save best model
            save_path = Path('./checkpoints') / f'best_model_epoch_{epoch}.pt'
            save_path.parent.mkdir(exist_ok=True, parents=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # 'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'metrics': metrics,
                'user_mapping': user_mapping,
                'item_mapping': item_mapping,
                'config': config
            }, save_path)
            
            print(f"\n New best model saved! Recall@10: {best_recall:.4f}")
        else:
            patience_counter += 1
            print(f"\n No improvement. Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"\n Early stopping triggered after {epoch} epochs")
                break
        
        # Step scheduler
        # scheduler.step()
        
        # Log current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.6f}")
    
    print(f"\n{'='*70}")
    print(f"Training completed!")
    print(f"Best Recall@10: {best_recall:.4f}")
    print(f"{'='*70}")
    
    return model, best_recall


if __name__ == '__main__':
    model, best_recall = main()