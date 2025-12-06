#!/usr/bin/env python3
"""
Record IL3D Behavior Cloning Dataset

Replays ALFRED expert trajectories in the REAL AI2-THOR 3D simulator and records:
- ResNet visual features (25,088-dim)
- Instruction text
- Discrete action indices (0-6)

This creates a clean BC dataset for training IL policies in 3D.
"""

import sys
import os
import json
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Add project to path
sys.path.insert(0, '/home/lilmeow/RL-project')

from env.wrappers.alfred_sim_env_3d import AlfredSimEnv3D


class IL3DDatasetRecorder:
    """Records ALFRED expert demonstrations in 3D simulator."""
    
    # Action mapping: ALFRED low-level -> 7 discrete actions
    ACTION_MAP = {
        'LookDown_15': 0,
        'LookUp_15': 1,
        'MoveAhead_25': 2,
        'PickupObject': 3,
        'RotateLeft_90': 4,
        'RotateRight_90': 5,
        'ToggleObjectOn': 6,
        'ToggleObjectOff': 6,  # Same as On for simplicity
        'PutObject': 3,  # Treat as pickup
        'OpenObject': 3,
        'CloseObject': 3,
    }
    
    def __init__(self, output_dir='data/il3d_bc', headless=False):
        """
        Initialize recorder.
        
        Args:
            output_dir: Directory to save dataset
            headless: Run simulator headless
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.headless = headless
        
        # Initialize feature extractor (same as RL/IL policies)
        print("Initializing ResNet feature extractor...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.resnet = models.resnet18(pretrained=True)
        self.resnet = torch.nn.Sequential(*list(self.resnet.children())[:-2])
        self.resnet = self.resnet.to(self.device)
        self.resnet.eval()
        
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✓ Feature extractor ready on {self.device}")
    
    def extract_features(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract ResNet features from RGB frame.
        
        Args:
            frame: RGB frame (H, W, 3)
            
        Returns:
            features: (25088,) feature vector
        """
        img = Image.fromarray(frame.astype('uint8'), 'RGB')
        img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.resnet(img_tensor)  # (1, 512, 7, 7)
            features = features.view(1, -1)  # (1, 25088)
        
        return features.cpu().numpy()[0]
    
    def map_action(self, alfred_action: dict) -> int:
        """
        Map ALFRED low-level action to discrete action index.
        
        Args:
            alfred_action: ALFRED action dict with 'action' key
            
        Returns:
            action_idx: 0-6, or -1 if cannot map
        """
        action_name = alfred_action['action']
        
        # Remove trailing numbers (e.g., MoveAhead_25 -> MoveAhead)
        base_action = action_name.split('_')[0] if '_' in action_name else action_name
        
        # Try exact match first
        if action_name in self.ACTION_MAP:
            return self.ACTION_MAP[action_name]
        
        # Try base action
        for key in self.ACTION_MAP:
            if key.startswith(base_action):
                return self.ACTION_MAP[key]
        
        return -1  # Cannot map
    
    def record_episode(self, env: AlfredSimEnv3D, split: str, episode_idx: int, 
                      max_steps: int = 80) -> dict:
        """
        Record one episode by replaying expert actions.
        
        Args:
            env: 3D simulator environment
            split: Dataset split
            episode_idx: Episode index
            max_steps: Maximum steps to record
            
        Returns:
            episode_data: Dict with features, actions, instruction, or None if failed
        """
        try:
            # Reset environment to this episode
            obs = env.reset(split=split, episode_idx=episode_idx)
            
            instruction = obs.get('instruction', '')
            episode_id = obs.get('episode_id', f'{split}_{episode_idx}')
            
            # Load expert trajectory
            traj_data = env.get_current_traj_data()
            if traj_data is None:
                return None
            
            expert_actions = traj_data['plan']['low_actions']
            
            # Record features and actions
            features_list = []
            actions_list = []
            
            step = 0
            for expert_action in expert_actions:
                if step >= max_steps:
                    break
                
                # Map action
                action_idx = self.map_action(expert_action['api_action'])
                
                if action_idx == -1:
                    # Cannot map this action, skip
                    continue
                
                # Extract features from current frame
                features = self.extract_features(obs['frame'])
                
                # Record
                features_list.append(features)
                actions_list.append(action_idx)
                
                # Step environment with this action
                obs, reward, done, info = env.step(action_idx)
                step += 1
                
                if done:
                    break
            
            if len(features_list) == 0:
                return None
            
            return {
                'episode_id': episode_id,
                'instruction': instruction,
                'features': np.array(features_list, dtype=np.float32),
                'actions': np.array(actions_list, dtype=np.int64)
            }
            
        except Exception as e:
            print(f"  ✗ Failed to record episode {episode_idx}: {e}")
            return None
    
    def record_split(self, split: str, max_episodes: int, env: AlfredSimEnv3D) -> list:
        """
        Record multiple episodes for a split.
        
        Args:
            split: Dataset split (train, valid_seen, valid_unseen)
            max_episodes: Maximum episodes to record
            env: Environment instance
            
        Returns:
            episodes: List of recorded episode dicts
        """
        print(f"\n{'='*70}")
        print(f"Recording {split} split (max {max_episodes} episodes)")
        print(f"{'='*70}")
        
        episodes = []
        successful = 0
        failed = 0
        
        for ep_idx in tqdm(range(max_episodes), desc=f"{split}"):
            episode_data = self.record_episode(env, split, ep_idx)
            
            if episode_data is not None:
                episodes.append(episode_data)
                successful += 1
                
                # Log progress
                tqdm.write(f"  ✓ Episode {ep_idx}: {len(episode_data['features'])} steps")
            else:
                failed += 1
        
        print(f"\n{split} Summary:")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Total transitions: {sum(len(e['features']) for e in episodes)}")
        
        return episodes
    
    def build_dataset(self, max_train: int = 150, max_valid: int = 30):
        """
        Build complete IL3D BC dataset.
        
        Args:
            max_train: Max training episodes
            max_valid: Max validation episodes
        """
        print("="*70)
        print("IL3D Behavior Cloning Dataset Builder")
        print("="*70)
        print(f"\nOutput directory: {self.output_dir}")
        print(f"Max episodes: train={max_train}, valid={max_valid}")
        
        # Initialize 3D environment
        if self.headless:
            from xvfbwrapper import Xvfb
            vdisplay = Xvfb(width=1024, height=768)
            vdisplay.start()
        else:
            vdisplay = None
        
        env = AlfredSimEnv3D(max_steps=100)
        
        # Record train split
        train_episodes = self.record_split('train', max_train, env)
        
        # Record valid split (use valid_seen)
        valid_episodes = self.record_split('valid_seen', max_valid, env)
        
        # Cleanup
        env.close()
        if vdisplay is not None:
            vdisplay.stop()
        
        # Save datasets
        print(f"\n{'='*70}")
        print("Saving datasets...")
        print(f"{'='*70}")
        
        train_path = self.output_dir / 'train_episodes.pkl'
        valid_path = self.output_dir / 'valid_episodes.pkl'
        
        with open(train_path, 'wb') as f:
            pickle.dump(train_episodes, f)
        print(f"✓ Saved train: {train_path} ({len(train_episodes)} episodes)")
        
        with open(valid_path, 'wb') as f:
            pickle.dump(valid_episodes, f)
        print(f"✓ Saved valid: {valid_path} ({len(valid_episodes)} episodes)")
        
        # Save metadata
        total_train_transitions = sum(len(e['features']) for e in train_episodes)
        total_valid_transitions = sum(len(e['features']) for e in valid_episodes)
        
        metadata = {
            'num_train_episodes': len(train_episodes),
            'num_valid_episodes': len(valid_episodes),
            'total_transitions_train': total_train_transitions,
            'total_transitions_valid': total_valid_transitions,
            'feature_dim': 25088,
            'num_actions': 7
        }
        
        metadata_path = self.output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Saved metadata: {metadata_path}")
        
        # Print summary
        print(f"\n{'='*70}")
        print("Dataset Build Complete!")
        print(f"{'='*70}")
        print(f"\nTrain:")
        print(f"  Episodes: {metadata['num_train_episodes']}")
        print(f"  Transitions: {metadata['total_transitions_train']}")
        
        print(f"\nValid:")
        print(f"  Episodes: {metadata['num_valid_episodes']}")
        print(f"  Transitions: {metadata['total_transitions_valid']}")
        
        print(f"\nFeature dim: {metadata['feature_dim']}")
        print(f"Actions: {metadata['num_actions']}\n")
        
        # Show example episode
        if train_episodes:
            print("Example episode shapes:")
            ex = train_episodes[0]
            print(f"  Instruction: '{ex['instruction'][:80]}...'")
            print(f"  Features: {ex['features'].shape}")
            print(f"  Actions: {ex['actions'].shape}")
        
        return metadata


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Record IL3D BC dataset from ALFRED trajectories')
    parser.add_argument('--max-train', type=int, default=150,
                       help='Max training episodes (default: 150)')
    parser.add_argument('--max-valid', type=int, default=30,
                       help='Max validation episodes (default: 30)')
    parser.add_argument('--output', type=str, default='data/il3d_bc',
                       help='Output directory (default: data/il3d_bc)')
    parser.add_argument('--headless', action='store_true',
                       help='Run headless with Xvfb')
    
    args = parser.parse_args()
    
    # Build dataset
    recorder = IL3DDatasetRecorder(output_dir=args.output, headless=args.headless)
    recorder.build_dataset(max_train=args.max_train, max_valid=args.max_valid)


if __name__ == '__main__':
    main()
