#!/usr/bin/env python3
"""
RL + Reflexion Policy Wrapper for 3D Simulator

Wraps CQL RL policy with Reflexion meta-controller for improved performance.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

import sys
sys.path.insert(0, '/home/lilmeow/RL-project')

from reflection.reflexion_controller_3d import ReflexionController3D


class RLReflexionSimPolicyWrapper:
    """
    RL+Reflexion policy wrapper for 3D simulator.
    
    Combines CQL RL policy with Reflexion meta-controller.
    """
    
    def __init__(self, rl_checkpoint_path='models/offline_rl_cql/cql_policy.d3',
                 rule_db_path='data/rules/rule_database_real_3d.json',
                 device='cuda'):
        """
        Initialize RL+Reflexion policy.
        
        Args:
            rl_checkpoint_path: Path to CQL checkpoint
            rule_db_path: Path to rule database
            device: Device to use
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading RL+Reflexion policy...")
        
        # Load RL policy (CQL)
        try:
            self.rl_policy = torch.load(rl_checkpoint_path, map_location=self.device)
            self.rl_policy.eval()
            print(f"✓ RL policy loaded: {rl_checkpoint_path}")
        except Exception as e:
            print(f"⚠ Could not load RL policy: {e}")
            print(f"  Falling back to random policy for demonstration")
            self.rl_policy = None
        
        # Initialize Reflexion controller
        self.reflexion = ReflexionController3D(rule_db_path)
        print(f"✓ Reflexion controller loaded")
        
        # ResNet feature extractor (same as RL training)
        self.resnet = models.resnet18(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        self.resnet = self.resnet.to(self.device)
        self.resnet.eval()
        
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✓ RL+Reflexion policy ready on {self.device}")
        print(f"  Base: CQL (vision-only)")
        print(f"  Meta-controller: Reflexion (rule-based)")
        
        # Track last reward
        self.last_reward = 0.0
    
    def start_episode(self, task_type=None, episode_id=None):
        """Start new episode and load Reflexion rules."""
        self.reflexion.start_episode(task_type, episode_id)
        self.last_reward = 0.0
    
    def extract_features(self, frame: np.ndarray) -> torch.Tensor:
        """
        Extract ResNet features from RGB frame.
        
        Args:
            frame: RGB frame (H, W, 3)
            
        Returns:
            features: (1, 25088) feature tensor
        """
        img = Image.fromarray(frame.astype('uint8'), 'RGB')
        img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.resnet(img_tensor)  # (1, 512, 7, 7)
            features = features.view(1, -1)  # (1, 25088)
        
        return features
    
    def predict(self, frame: np.ndarray, reward: float = 0.0) -> int:
        """
        Predict action using RL + Reflexion.
        
        Args:
            frame: RGB frame (H, W, 3)
            reward: Reward from previous step
            
        Returns:
            action_id: Final action (0-6)
        """
        self.last_reward = reward
        
        # Extract features
        features = self.extract_features(frame)
        
        # Get RL action
        if self.rl_policy is not None:
            with torch.no_grad():
                rl_action = self.rl_policy(features).argmax(dim=1).item()
        else:
            # Random fallback
            rl_action = np.random.randint(0, 7)
        
        # Apply Reflexion meta-control
        final_action = self.reflexion.choose_action(rl_action, self.last_reward)
        
        return final_action


# Test
if __name__ == '__main__':
    print("Testing RLReflexionSimPolicyWrapper...")
    
    policy = RLReflexionSimPolicyWrapper()
    policy.start_episode(task_type='pick_and_place_simple')
    
    # Test prediction
    frame = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    action = policy.predict(frame)
    
    print(f"✓ Predicted action: {action}")
    print(f"✓ Test complete")
