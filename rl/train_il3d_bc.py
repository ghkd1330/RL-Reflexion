#!/usr/bin/env python3
"""
Train IL3D Behavior Cloning Policy

Trains a vision-only BC policy on the IL3D dataset (data/il3d_bc/).
"""

import sys
import os
import json
import pickle
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Add project to path
sys.path.insert(0, '/home/lilmeow/RL-project')

from rl.models.il3d_bc_policy import IL3DBCPolicy


def load_il3d_dataset(data_dir='data/il3d_bc'):
    """
    Load IL3D BC dataset and flatten into (features, actions) arrays.
    
    Args:
        data_dir: Directory containing train/valid pkl files
        
    Returns:
        X_train, y_train, X_valid, y_valid: Numpy arrays
    """
    data_path = Path(data_dir)
    
    # Load train episodes
    with open(data_path / 'train_episodes.pkl', 'rb') as f:
        train_episodes = pickle.load(f)
    
    # Load valid episodes
    with open(data_path / 'valid_episodes.pkl', 'rb') as f:
        valid_episodes = pickle.load(f)
    
    # Flatten train episodes
    X_train_list = []
    y_train_list = []
    
    for ep in train_episodes:
        X_train_list.append(ep['features'])  # (T, 25088)
        y_train_list.append(ep['actions'])    # (T,)
    
    X_train = np.concatenate(X_train_list, axis=0)  # (N_train, 25088)
    y_train = np.concatenate(y_train_list, axis=0)  # (N_train,)
    
    # Flatten valid episodes
    X_valid_list = []
    y_valid_list = []
    
    for ep in valid_episodes:
        X_valid_list.append(ep['features'])
        y_valid_list.append(ep['actions'])
    
    X_valid = np.concatenate(X_valid_list, axis=0)
    y_valid = np.concatenate(y_valid_list, axis=0)
    
    print(f"Dataset loaded:")
    print(f"  Train: {X_train.shape[0]} transitions")
    print(f"  Valid: {X_valid.shape[0]} transitions")
    
    return X_train, y_train, X_valid, y_valid


def train_il3d_bc(epochs=10, batch_size=64, lr=1e-3, device='cuda'):
    """
    Train IL3D BC policy.
    
    Args:
        epochs: Number of training epochs
        batch_size: Training batch size
        lr: Learning rate
        device: Device to train on
        
    Returns:
        metrics: Training metrics dict
    """
    print("="*70)
    print("IL3D Behavior Cloning Training")
    print("="*70)
    
    # Device setup
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load dataset
    print("\nLoading dataset...")
    X_train, y_train, X_valid, y_valid = load_il3d_dataset()
    
    # Create PyTorch datasets
    train_dataset = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).long()
    )
    
    valid_dataset = TensorDataset(
        torch.from_numpy(X_valid).float(),
        torch.from_numpy(y_valid).long()
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False)
    
    print(f"✓ Dataloaders created")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Valid batches: {len(valid_loader)}")
    
    # Create model
    print("\nCreating model...")
    model = IL3DBCPolicy().to(device)
    print(f"✓ Model created")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    
    best_valid_acc = 0.0
    best_epoch = 0
    history = []
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for features, actions in pbar:
            features = features.to(device)
            actions = actions.to(device)
            
            # Forward
            logits = model(features)
            loss = criterion(logits, actions)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Stats
            train_loss += loss.item() * features.size(0)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == actions).sum().item()
            train_total += features.size(0)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # Validate
        model.eval()
        valid_loss = 0.0
        valid_correct = 0
        valid_total = 0
        
        with torch.no_grad():
            for features, actions in valid_loader:
                features = features.to(device)
                actions = actions.to(device)
                
                logits = model(features)
                loss = criterion(logits, actions)
                
                valid_loss += loss.item() * features.size(0)
                preds = torch.argmax(logits, dim=1)
                valid_correct += (preds == actions).sum().item()
                valid_total += features.size(0)
        
        valid_loss /= valid_total
        valid_acc = valid_correct / valid_total
        
        # Log
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'valid_loss': valid_loss,
            'valid_acc': valid_acc
        }
        history.append(epoch_metrics)
        
        print(f"  Epoch {epoch+1}: "
              f"train_loss={train_loss:.4f}, train_acc={train_acc:.2%}, "
              f"valid_loss={valid_loss:.4f}, valid_acc={valid_acc:.2%}")
        
        # Save best model
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_epoch = epoch + 1
            
            # Save checkpoint
            checkpoint_dir = Path('models/il3d_bc')
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            torch.save(model.state_dict(), checkpoint_dir / 'bc_3d_policy_best.pth')
            print(f"  ✓ New best model saved (valid_acc={valid_acc:.2%})")
    
    # Save final model
    torch.save(model.state_dict(), checkpoint_dir / 'bc_3d_policy_last.pth')
    
    # Prepare metrics
    metrics = {
        'epochs': epochs,
        'best_valid_acc': best_valid_acc,
        'best_epoch': best_epoch,
        'final_train_acc': history[-1]['train_acc'],
        'final_valid_acc': history[-1]['valid_acc'],
        'num_train_transitions': len(X_train),
        'num_valid_transitions': len(X_valid),
        'history': history
    }
    
    # Save metrics
    metrics_path = Path('data/logs/il3d_bc_train_metrics.json')
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}")
    print(f"\nBest validation accuracy: {best_valid_acc:.2%} (epoch {best_epoch})")
    print(f"Train transitions: {len(X_train)}")
    print(f"Valid transitions: {len(X_valid)}")
    print(f"\nCheckpoints saved:")
    print(f"  - models/il3d_bc/bc_3d_policy_best.pth")
    print(f"  - models/il3d_bc/bc_3d_policy_last.pth")
    print(f"\nMetrics saved:")
    print(f"  - data/logs/il3d_bc_train_metrics.json")
    
    return metrics


def eval_checkpoint(checkpoint_path='models/il3d_bc/bc_3d_policy_best.pth'):
    """
    Evaluate BC policy checkpoint on validation set.
    
    Args:
        checkpoint_path: Path to checkpoint
    """
    print("="*70)
    print("Evaluating BC Policy Checkpoint")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = IL3DBCPolicy().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"✓ Model loaded from {checkpoint_path}")
    
    # Load validation data
    _, _, X_valid, y_valid = load_il3d_dataset()
    
    valid_dataset = TensorDataset(
        torch.from_numpy(X_valid).float(),
        torch.from_numpy(y_valid).long()
    )
    valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False)
    
    # Evaluate
    correct = 0
    total = 0
    
    with torch.no_grad():
        for features, actions in tqdm(valid_loader, desc="Evaluating"):
            features = features.to(device)
            actions = actions.to(device)
            
            logits = model(features)
            preds = torch.argmax(logits, dim=1)
            
            correct += (preds == actions).sum().item()
            total += features.size(0)
    
    accuracy = correct / total
    
    print(f"\nValidation Accuracy: {accuracy:.2%} ({correct}/{total})")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train IL3D BC policy')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs (default: 10)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate (default: 1e-3)')
    parser.add_argument('--eval-only', action='store_true',
                       help='Only evaluate existing checkpoint')
    
    args = parser.parse_args()
    
    if args.eval_only:
        eval_checkpoint()
    else:
        train_il3d_bc(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr
        )


if __name__ == '__main__':
    main()
