#!/usr/bin/env python3
"""
RL (CQL) Policy Wrapper for Real 3D Simulator - UPGRADED

NO FALLBACK TO RANDOM - Loads real CQL policy or fails explicitly.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path
from PIL import Image

# d3rlpy for CQL
try:
    import d3rlpy
    D3RLPY_AVAILABLE = True
except ImportError:
    D3RLPY_AVAILABLE = False
    print("ERROR: d3rlpy is required for RL policy")


class RLSimPolicyWrapper:
    """
    Wrapper for RL (CQL) policy - NO FALLBACK VERSION.
    
    Takes RGB frames, extracts features, and predicts actions using CQL.
    """
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        """
        Initialize RL policy.
        
        Args:
            checkpoint_path: Path to CQL checkpoint (.d3 file)
            device: Device to use (cuda/cpu)
        """
        if not D3RLPY_AVAILABLE:
            raise ImportError("d3rlpy is required but not installed. Install with: pip install d3rlpy==2.5.0")
        
        # d3rlpy 2.8.1 expects device in format "cuda:0" or "cpu:0"
        if 'cuda' in device and torch.cuda.is_available():
            self.device_str = 'cuda:0'
        else:
            self.device_str = 'cpu:0'
        
        self.device = torch.device(self.device_str.split(':')[0])  # For PyTorch
        
        print(f"Loading RL (CQL) policy from {checkpoint_path}...")
        
        # Load CQL policy - NO FALLBACK
        try:
            self.policy = d3rlpy.load_learnable(checkpoint_path, device=self.device_str)
            print(f"✓ CQL policy loaded successfully on {self.device_str}")
        except Exception as e:
            error_msg = f"Failed to load CQL policy: {e}\n"
            error_msg += "This is a CRITICAL error. The RL policy must load correctly.\n"
            error_msg += "Check:\n"
            error_msg += f"  1. Checkpoint exists: {Path(checkpoint_path).exists()}\n"
            error_msg += f"  2. d3rlpy version: {d3rlpy.__version__}\n"
            error_msg += "  3. Device format: Should be 'cuda:0' or 'cpu:0'\n"
            raise RuntimeError(error_msg) from e
        
        # Initialize ResNet feature extractor (same as offline dataset)
        self.resnet = models.resnet18(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        self.resnet = self.resnet.to(self.device)
        self.resnet.eval()
        
        # Image preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✓ Feature extractor ready")
    
    def extract_features(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract ResNet features from RGB frame.
        
        Args:
            frame: RGB frame (H, W, 3) numpy array
            
        Returns:
            features: Feature vector (25088,) flattened
        """
        # Convert to PIL Image
        img = Image.fromarray(frame.astype('uint8'), 'RGB')
        
        # Preprocess
        img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.resnet(img_tensor)  # (1, 512, 7, 7)
            features = features.view(1, -1)  # (1, 25088)
        
        return features.cpu().numpy()[0]
    
    def predict(self, frame: np.ndarray, instruction: str = None) -> int:
        """
        Predict action from frame using CQL policy.
        
        Args:
            frame: RGB frame (H, W, 3)
            instruction: Instruction text (not used by CQL)
            
        Returns:
            action_idx: Predicted action index (0-6)
        """
        # Extract features
        features = self.extract_features(frame)
        
        # Reshape for d3rlpy (expects 2D: batch_size x features)
        features_batch = features.reshape(1, -1)
        
        # Predict with CQL - NO FALLBACK
        try:
            action = self.policy.predict(features_batch)
            
            # Extract action index
            if isinstance(action, np.ndarray):
                action_idx = int(action[0])
            else:
                action_idx = int(action)
            
            # Ensure valid range
            action_idx = max(0, min(6, action_idx))
            
            return action_idx
        except Exception as e:
            raise RuntimeError(f"CQL prediction failed: {e}") from e


def test_rl_sim_policy():
    """Test RL policy wrapper."""
    print("Testing RL Simulator Policy...")
    
    # Create dummy frame
    frame = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    instruction = "Pick up the apple"
    
    # Load policy - will raise error if fails
    try:
        policy = RLSimPolicyWrapper('models/offline_rl_cql/cql_policy.d3')
        
        # Predict
        action = policy.predict(frame, instruction)
        print(f"✓ Predicted action: {action}")
        
        print("✓ Test complete - RL policy working!")
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    test_rl_sim_policy()
