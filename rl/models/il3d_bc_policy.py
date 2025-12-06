#!/usr/bin/env python3
"""
IL3D Behavior Cloning Policy

Lightweight vision-only policy for ALFRED 3D tasks.
Maps ResNet features (25,088-dim) to 7 discrete actions.
"""

import torch
import torch.nn as nn


class IL3DBCPolicy(nn.Module):
    """
    Behavior cloning policy for IL3D dataset.
    
    Architecture:
    - Input: ResNet features (25,088-dim)
    - FC layers with dropout
    - Output: 7 action logits
    """
    
    def __init__(self, feature_dim=25088, hidden_dim1=512, hidden_dim2=256, 
                 num_actions=7, dropout=0.2):
        """
        Initialize BC policy.
        
        Args:
            feature_dim: Input feature dimension (default: 25088)
            hidden_dim1: First hidden layer size (default: 512)
            hidden_dim2: Second hidden layer size (default: 256)
            num_actions: Number of discrete actions (default: 7)
            dropout: Dropout rate (default: 0.2)
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_actions = num_actions
        
        # Network layers
        self.fc1 = nn.Linear(feature_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.head = nn.Linear(hidden_dim2, num_actions)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Feature tensor (batch_size, feature_dim)
            
        Returns:
            logits: Action logits (batch_size, num_actions)
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        logits = self.head(x)
        
        return logits


def load_il3d_bc_policy(checkpoint_path, device='cuda'):
    """
    Load trained IL3D BC policy from checkpoint.
    
    Args:
        checkpoint_path: Path to state_dict checkpoint
        device: Device to load model on
        
    Returns:
        model: Loaded policy model
    """
    model = IL3DBCPolicy()
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
    
    return model


# Test
if __name__ == '__main__':
    print("Testing IL3DBCPolicy...")
    
    # Create model
    model = IL3DBCPolicy()
    print(f"✓ Model created")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, 25088)
    
    logits = model(x)
    print(f"✓ Forward pass")
    print(f"  Input: {x.shape}")
    print(f"  Output: {logits.shape}")
    
    # Test prediction
    probs = torch.softmax(logits, dim=1)
    actions = torch.argmax(probs, dim=1)
    print(f"✓ Action prediction")
    print(f"  Actions: {actions[:5].tolist()}")
    
    print("\n✓ All tests passed")
