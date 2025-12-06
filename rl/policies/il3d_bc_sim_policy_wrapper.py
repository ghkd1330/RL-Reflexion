#!/usr/bin/env python3
"""
IL3D-BC Simulator Policy Wrapper

Wraps the trained behavior cloning policy for evaluation in the real 3D AI2-THOR simulator.
Uses ResNet feature extraction to map RGB frames to discrete actions (0-6).
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

import sys
sys.path.insert(0, '/home/lilmeow/RL-project')

from rl.models.il3d_bc_policy import IL3DBCPolicy


class IL3DBC3DSimPolicy:
    """
    IL3D-BC policy wrapper for 3D simulator evaluation.
    
    Loads trained BC model and performs vision-only action prediction.
    """
    
    def __init__(self, checkpoint_path='models/il3d_bc/bc_3d_policy_best.pth', device='cuda'):
        """
        Initialize IL3D-BC policy.
        
        Args:
            checkpoint_path: Path to trained BC checkpoint
            device: Device to use
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading IL3D-BC policy from {checkpoint_path}...")
        
        # Load BC model
        self.model = IL3DBCPolicy().to(self.device)
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        print(f"✓ BC model loaded on {self.device}")
        
        # Initialize ResNet feature extractor (same as training)
        self.resnet = models.resnet18(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        self.resnet = self.resnet.to(self.device)
        self.resnet.eval()
        
        # Image preprocessing (same as training)
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✓ ResNet feature extractor ready")
        print(f"  Policy: IL3D-BC (vision-only)")
        print(f"  Actions: 7 discrete (0-6)")
    
    def extract_features(self, frame: np.ndarray) -> torch.Tensor:
        """
        Extract ResNet features from RGB frame.
        
        Args:
            frame: RGB frame (H, W, 3) numpy array, uint8
            
        Returns:
            features: Feature tensor (1, 25088)
        """
        # Convert to PIL Image
        img = Image.fromarray(frame.astype('uint8'), 'RGB')
        
        # Preprocess
        img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.resnet(img_tensor)  # (1, 512, 7, 7)
            features = features.view(1, -1)  # (1, 25088)
        
        return features
    
    def predict(self, frame: np.ndarray, instruction: str = None) -> int:
        """
        Predict action from RGB frame.
        
        Vision-only prediction (instruction is ignored).
        
        Args:
            frame: RGB frame (H, W, 3)
            instruction: Instruction text (ignored, for API compatibility)
            
        Returns:
            action_id: Discrete action index (0-6)
        """
        # Extract features
        features = self.extract_features(frame)
        
        # Predict action
        with torch.no_grad():
            logits = self.model(features)  # (1, 7)
            action_id = torch.argmax(logits, dim=1).item()
        
        return action_id


# Test
if __name__ == '__main__':
    print("Testing IL3DBC3DSimPolicy...")
    
    # Create policy
    policy = IL3DBC3DSimPolicy()
    
    # Test prediction
    frame = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    action = policy.predict(frame)
    
    print(f"✓ Predicted action: {action}")
    print(f"✓ Test complete")
