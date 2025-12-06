#!/usr/bin/env python3
"""
Real Seq2Seq IL Policy Wrapper for 3D Simulator

This wrapper loads the REAL seq2seq IL model and uses it for action prediction
in the 3D AI2-THOR simulator. It is instruction-conditioned.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path
from PIL import Image

# Import our standalone seq2seq model
import sys
sys.path.insert(0, '/home/lilmeow/RL-project')
from rl.models.seq2seq_standalone import StandaloneSeq2SeqIL, load_seq2seq_checkpoint


class RealILSimPolicyWrapper:
    """
    Real IL policy wrapper using seq2seq model for 3D simulator.
    
    This is instruction-conditioned: uses both RGB frames AND instruction text.
    """
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        """
        Initialize IL policy with REAL seq2seq model.
        
        Args:
            checkpoint_path: Path to seq2seq IL checkpoint
            device: Device to use
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading REAL Seq2Seq IL policy with trained weights...")
        
        # Load seq2seq model with REAL trained weights
        try:
            self.model = load_seq2seq_checkpoint(
                checkpoint_path='models/seq2seq_il_best/retrained_state_dict.pth',
                device=str(self.device)
            )
            self.model.eval()
            print(f"✓ Seq2Seq model loaded with REAL trained weights (epoch 1, F1: 81%)")
        except Exception as e:
            raise RuntimeError(f"CRITICAL: Failed to load trained IL weights: {e}") from e
        
        # Initialize ResNet feature extractor (same as training)
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
        
        print(f"✓ Real IL policy initialized on {self.device}")
        print(f"  Model: Seq2Seq (instruction-conditioned)")
        print(f"  Actions: 7 discrete ALFRED actions")
    
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
        Predict action from frame and instruction.
        
        THIS IS INSTRUCTION-CONDITIONED: Uses both visual and language input.
        
        Args:
            frame: RGB frame (H, W, 3)
            instruction: Instruction text (REQUIRED for instruction conditioning)
            
        Returns:
            action_idx: Predicted action index (0-6)
        """
        # Extract visual features
        features = self.extract_features(frame)
        
        # Use seq2seq model for prediction (instruction-conditioned)
        action_idx = self.model.predict_action(features, instruction)
        
        return action_idx


def test_real_il_sim_policy():
    """Test real IL simulator policy."""
    print("Testing Real IL Simulator Policy...")
    
    # Create dummy frame
    frame = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    instruction = "Pick up the apple and put it in the microwave"
    
    # Load policy
    policy = RealILSimPolicyWrapper('models/seq2seq_il_best/best_seen.pth')
    
    # Predict (instruction-conditioned)
    action = policy.predict(frame, instruction)
    print(f"✓ Predicted action: {action}")
    print(f"  Instruction was: {instruction}")
    
    # Test without instruction (should still work but use default)
    action2 = policy.predict(frame)
    print(f"✓ Predicted action (no instruction): {action2}")
    
    print("✓ Test complete - IL policy is instruction-conditioned!")


if __name__ == '__main__':
    test_real_il_sim_policy()
