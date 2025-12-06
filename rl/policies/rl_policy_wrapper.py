#!/usr/bin/env python3
"""
RL Policy Wrapper for ALFRED Evaluation.

Loads the CQL offline RL policy and provides action prediction interface.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add d3rlpy if needed
try:
    import d3rlpy
except ImportError:
    print("Warning: d3rlpy not installed")


class RLPolicyWrapper:
    """
    Wrapper for offline RL (CQL) policy.
    
    Provides action prediction from observations.
    """
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda:0'):
        """
        Initialize RL policy.
        
        Args:
            checkpoint_path: Path to CQL checkpoint (.d3 file)
            device: Device to run inference on
        """
        self.device = device
        self.checkpoint_path = checkpoint_path
        
        # Action mapping (same as IL)
        self.action_to_idx = {
            'LookDown_15': 0,
            'LookUp_15': 1,
            'MoveAhead_25': 2,
            'PickupObject': 3,
            'RotateLeft_90': 4,
            'RotateRight_90': 5,
            'ToggleObjectOn': 6
        }
        self.idx_to_action = {v: k for k, v in self.action_to_idx.items()}
        
        # Load model
        print(f"Loading RL policy from {checkpoint_path}...")
        self.policy = self._load_policy()
        print("✓ RL policy loaded")
        
    def _load_policy(self):
        """Load CQL policy from d3rlpy checkpoint."""
        try:
            # d3rlpy expects "cuda" or "cpu", not "cuda:0"
            device = "cuda" if "cuda" in self.device else "cpu"
            policy = d3rlpy.load_learnable(self.checkpoint_path, device=device)
            return policy
        except Exception as e:
            print(f"Error loading policy: {e}")
            raise
    
    def predict(self, features: np.ndarray) -> int:
        """
        Predict action from features.
        
        Args:
            features: ResNet features, shape (25088,)
            
        Returns:
            Action index [0-6]
        """
        # Ensure correct shape
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Predict with d3rlpy
        action = self.policy.predict(features)
        
        # Extract action index
        if isinstance(action, np.ndarray):
            action_idx = int(action[0][0]) if action.ndim > 1 else int(action[0])
        else:
            action_idx = int(action)
        
        return action_idx
    
    def get_action_name(self, action_idx: int) -> str:
        """Convert action index to name."""
        return self.idx_to_action.get(action_idx, 'MoveAhead_25')


def test_rl_wrapper():
    """Test RL policy wrapper."""
    print("Testing RL Policy Wrapper...")
    
    checkpoint_path = 'models/offline_rl_cql/cql_policy.d3'
    
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    try:
        policy = RLPolicyWrapper(checkpoint_path)
        
        # Test prediction with dummy features
        features = np.random.randn(25088).astype(np.float32)
        action = policy.predict(features)
        print(f"Predicted action: {action} ({policy.get_action_name(action)})")
        
        print("✓ RL wrapper test passed")
    except Exception as e:
        print(f"❌ RL wrapper test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    test_rl_wrapper()
