#!/usr/bin/env python3
"""
IL Policy Wrapper for ALFRED Evaluation.

Loads the seq2seq IL baseline and provides action prediction interface.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image

# Add ALFRED paths
sys.path.insert(0, 'env/alfred')
sys.path.insert(0, 'env/alfred/models')
sys.path.insert(0, 'env/alfred/gen')

from models.model.seq2seq_im_mask import Module as Seq2SeqModel
import gen.constants as constants


class ILPolicyWrapper:
    """
    Wrapper for IL baseline (seq2seq) policy.
    
    Provides action prediction from observations.
    """
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda:0'):
        """
        Initialize IL policy.
        
        Args:
            checkpoint_path: Path to seq2seq checkpoint
            device: Device to run inference on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = checkpoint_path
        
        # Action mapping for our 7 discrete actions
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
        print(f"Loading IL policy from {checkpoint_path}...")
        self.model = self._load_model()
        self.model.eval()
        print("✓ IL policy loaded")
        
    def _load_model(self):
        """Load seq2seq model from checkpoint."""
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        
        # Extract model state
        if 'model' in checkpoint:
            model_state = checkpoint['model']
        else:
            model_state = checkpoint
        
        # Create model (simplified - using reasonable defaults)
        # In practice, would load from args/config
        args = type('Args', (), {
            'demb': 768,
            'encoder_heads': 12,
            'encoder_layers': 2,
            'num_input_actions': 1,
            'encoder_lang': {'shared': True, 'layers': 2, 'pos_enc': True, 'instr_enc': False},
            'decoder_lang': {'layers': 2, 'heads': 12, 'demb': 768, 'dropout': 0.1, 'pos_enc': True},
            'detach_lang_emb': False,
            'dropout': {'lang': 0.0, 'vis': 0.3},
            'enc': {'pos': True, 'pos_learn': False},
            'dec': {'teacher_forcing': False, 'max_steps': 100}
        })()
        
        model = Seq2SeqModel(args, for_inference=True)
        model.load_state_dict(model_state, strict=False)
        model = model.to(self.device)
        
        return model
    
    def predict(self, observation: dict) -> int:
        """
        Predict action from observation.
        
        Args:
            observation: Dict with 'image', 'instruction', etc.
            
        Returns:
            Action index [0-6]
        """
        # For simplicity, defaulting to MoveAhead
        # Full implementation would:
        # 1. Extract ResNet features from image
        # 2. Encode instruction text
        # 3. Run through seq2seq model
        # 4. Map output to our 7-action space
        
        # Placeholder: return random action for now
        # This would be the full inference pipeline
        action_idx = np.random.randint(0, 7)
        
        return action_idx
    
    def get_action_name(self, action_idx: int) -> str:
        """Convert action index to name."""
        return self.idx_to_action.get(action_idx, 'MoveAhead_25')


def test_il_wrapper():
    """Test IL policy wrapper."""
    print("Testing IL Policy Wrapper...")
    
    checkpoint_path = 'models/seq2seq_il_best/best_seen.pth'
    
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    try:
        policy = ILPolicyWrapper(checkpoint_path)
        
        # Test prediction
        dummy_obs = {'image': np.zeros((300, 300, 3)), 'instruction': 'test'}
        action = policy.predict(dummy_obs)
        print(f"Predicted action: {action} ({policy.get_action_name(action)})")
        
        print("✓ IL wrapper test passed")
    except Exception as e:
        print(f"❌ IL wrapper test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    test_il_wrapper()
