#!/usr/bin/env python3
"""
ALFRED Evaluation Environment - Gym-compatible wrapper for simulator evaluation.

This wrapper interfaces with ALFRED's ThorEnv and AI2-THOR for policy evaluation.
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add ALFRED paths properly
sys.path.insert(0, 'env/alfred')
sys.path.insert(0, 'env/alfred/models')
sys.path.insert(0, 'env/alfred/gen')

# Import ALFRED modules
try:
    from env.thor_env import ThorEnv
    import gen.constants as constants
except ImportError as e:
    print(f"Warning: ALFRED import failed: {e}")
    print("Some functionality may be limited")
    ThorEnv = None
    constants = None


class ALFREDEvalEnv:
    """
    Evaluation environment for ALFRED tasks.
    
    Provides a simple interface for running episodes in AI2-THOR simulator.
    """
    
    def __init__(self, 
                 data_path: str,
                 splits_path: str,
                 split: str = 'valid_seen',
                 task_types: List[str] = None,
                 max_steps: int = 100):
        """
        Initialize ALFRED evaluation environment.
        
        Args:
            data_path: Path to ALFRED data
            splits_path: Path to splits JSON
            split: Which split to evaluate on
            task_types: Filter to specific task types
            max_steps: Maximum steps per episode
        """
        self.data_path = Path(data_path)
        self.split = split
        self.max_steps = max_steps
        
        # Load splits
        with open(splits_path) as f:
            splits_data = json.load(f)
        
        self.episodes = splits_data[split]
        
        # Filter by task type if specified
        if task_types:
            self.episodes = [
                ep for ep in self.episodes
                if any(task_type in ep['task'] for task_type in task_types)
            ]
        
        print(f"Loaded {len(self.episodes)} episodes from {split}")
        
        # Initialize ALFRED environment (will be created per episode)
        self.env = None
        self.current_episode_idx = None
        self.current_traj_data = None
        self.step_count = 0
        
    def get_num_episodes(self) -> int:
        """Get total number of episodes."""
        return len(self.episodes)
    
    def reset(self, episode_idx: int = None) -> Dict:
        """
        Reset environment to start a new episode.
        
        Args:
            episode_idx: Index of episode to load (random if None)
            
        Returns:
            Initial observation dict
        """
        if episode_idx is None:
            episode_idx = np.random.randint(0, len(self.episodes))
        
        self.current_episode_idx = episode_idx
        episode_data = self.episodes[episode_idx]
        
        # Load trajectory data
        task_path = episode_data['task']
        traj_json = self.data_path / self.split / task_path / 'traj_data.json'
        
        with open(traj_json) as f:
            self.current_traj_data = json.load(f)
        
        # Initialize Thor environment
        if self.env is None:
            self.env = ThorEnv()
        
        # Reset to scene
        scene_num = self.current_traj_data['scene']['scene_num']
        self.env.reset(scene_num)
        
        # Restore scene state
        self.env.restore_scene(
            object_poses=self.current_traj_data['scene']['object_poses'],
            object_toggles=self.current_traj_data['scene']['object_toggles'],
            dirty_and_empty=self.current_traj_data['scene']['dirty_and_empty']
        )
        
        # Setup task
        self.env.set_task(self.current_traj_data, args=None, reward_type='sparse')
        
        self.step_count = 0
        
        # Get initial observation
        obs = self._get_observation()
        
        return obs
    
    def _get_observation(self) -> Dict:
        """
        Extract observation from current environment state.
        
        Returns:
            Dict with:
                - image: RGB image from simulator
                - instruction: Task instruction text
                - goal_satisfied: Whether task is complete
        """
        # Get current frame from environment
        event = self.env.last_event
        image = event.frame  # RGB image
        
        # Get instruction
        instruction = self.current_traj_data['turk_annotations']['anns'][0]['task_desc']
        
        # Check goal
        goal_satisfied = self.env.get_goal_satisfied()
        
        obs = {
            'image': image,
            'instruction': instruction,
            'goal_satisfied': goal_satisfied,
            'step_count': self.step_count
        }
        
        return obs
    
    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute action in simulator.
        
        Args:
            action: Discrete action index (0-6 for our action space)
            
        Returns:
            observation, reward, done, info
        """
        # Map action index to ALFRED action
        action_map = {
            0: 'LookDown_15',
            1: 'LookUp_15',
            2: 'MoveAhead_25',
            3: 'PickupObject',
            4: 'RotateLeft_90',
            5: 'RotateRight_90',
            6: 'ToggleObjectOn'
        }
        
        action_name = action_map.get(action, 'MoveAhead_25')
        
        # Execute action in ALFRED
        # For simplicity, execute without object mask for now
        action_dict = {'action': action_name}
        try:
            self.env.step(action_dict)
            success = True
        except:
            success = False
        
        self.step_count += 1
        
        # Get new observation
        obs = self._get_observation()
        
        # Compute reward
        goal_satisfied = obs['goal_satisfied']
        if goal_satisfied:
            reward = 10.0  # Task complete
        elif success:
            reward = -0.01  # Small step penalty
        else:
            reward = -0.5  # Invalid action penalty
        
        # Check if done
        done = goal_satisfied or self.step_count >= self.max_steps
        
        # Info
        info = {
            'success': goal_satisfied,
            'episode_idx': self.current_episode_idx,
            'task': self.episodes[self.current_episode_idx]['task'],
            'action': action_name,
            'action_success': success
        }
        
        return obs, reward, done, info
    
    def close(self):
        """Close environment and cleanup."""
        if self.env is not None:
            self.env.stop()
            self.env = None
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()


def test_env():
    """Simple test of the evaluation environment."""
    print("Testing ALFRED Eval Environment...")
    
    env = ALFREDEvalEnv(
        data_path='data/json_feat_subset',
        splits_path='data/splits/subset_oct21.json',
        split='valid_seen',
        max_steps=20
    )
    
    print(f"\nTotal episodes: {env.get_num_episodes()}")
    
    # Test one episode
    print("\nTesting episode reset and steps...")
    obs = env.reset(episode_idx=0)
    print(f"Initial observation keys: {obs.keys()}")
    print(f"Instruction: {obs['instruction'][:100]}...")
    print(f"Image shape: {obs['image'].shape}")
    
    # Take a few random steps
    for i in range(5):
        action = np.random.randint(0, 7)
        obs, reward, done, info = env.step(action)
        print(f"Step {i+1}: action={info['action']}, reward={reward:.2f}, done={done}")
        if done:
            break
    
    env.close()
    print("\n✓ Test complete!")


if __name__ == '__main__':
    test_env()
