#!/usr/bin/env python3
"""
3D Episode Logging for Reflexion

Logs RL episodes in the real AI2-THOR 3D simulator with frames and metadata.
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, List, Optional
from datetime import datetime


class EpisodeLogger3D:
    """
    Logs a single 3D episode with frames and metadata for Reflexion analysis.
    """
    
    def __init__(self, episode_id: str, task_type: str, instruction: str, 
                 split: str, log_dir: str = 'data/logs/episodes_3d'):
        """
        Initialize episode logger.
        
        Args:
            episode_id: Unique episode identifier
            task_type: ALFRED task type
            instruction: Natural language instruction
            split: Dataset split (e.g., 'valid_seen')
            log_dir: Base directory for episode logs
        """
        self.episode_id = episode_id
        self.task_type = task_type
        self.instruction = instruction
        self.split = split
        
        # Create episode directory
        self.episode_dir = Path(log_dir) / episode_id
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        
        # Episode state
        self.steps = []
        self.frames_saved = []
        self.total_steps = 0
        self.total_reward = 0.0
        self.actions = []
        
    def log_step(self, frame: np.ndarray, action_id: int, reward: float, 
                 done: bool, info: Dict):
        """
        Log a single step.
        
        Args:
            frame: RGB frame (H, W, 3)
            action_id: Action taken (0-6)
            reward: Reward received
            done: Episode done flag
            info: Additional info dict
        """
        step_num = self.total_steps
        
        # Save important frames: first, every 10th, near failure, and last
        save_frame = (
            step_num == 0 or  # First frame
            step_num % 10 == 0 or  # Every 10 steps
            done  # Last frame
        )
        
        if save_frame:
            frame_path = self.episode_dir / f'frame_{step_num:03d}.png'
            img = Image.fromarray(frame.astype('uint8'), 'RGB')
            img.save(frame_path)
            self.frames_saved.append(str(frame_path))
        
        # Record action and reward
        self.actions.append(action_id)
        self.total_reward += reward
        self.total_steps += 1
        
    def finish(self, success: bool, failure_reason: str = "unknown"):
        """
        Finish logging and save metadata.
        
        Args:
            success: Whether episode succeeded
            failure_reason: Optional failure explanation
        """
        # Create metadata
        metadata = {
            'episode_id': self.episode_id,
            'task_type': self.task_type,
            'instruction': self.instruction,
            'split': self.split,
            'success': success,
            'total_steps': self.total_steps,
            'total_reward': float(self.total_reward),
            'failure_reason': failure_reason if not success else None,
            'actions': self.actions,
            'actions_compact': self._compact_actions(self.actions),
            'num_frames_saved': len(self.frames_saved),
            'frames': [str(Path(f).relative_to(self.episode_dir)) for f in self.frames_saved],
            'logged_at': datetime.now().isoformat()
        }
        
        # Save metadata
        meta_path = self.episode_dir / 'meta.json'
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata
    
    def _compact_actions(self, actions: List[int]) -> str:
        """
        Create a compact string representation of actions.
        
        Args:
            actions: List of action IDs
            
        Returns:
            compact_str: Run-length encoded string
        """
        if not actions:
            return ""
        
        action_names = {
            0: "LookDown", 1: "LookUp", 2: "MoveAhead",
            3: "Pickup", 4: "RotateLeft", 5: "RotateRight", 6: "Toggle"
        }
        
        # Run-length encode
        result = []
        current_action = actions[0]
        count = 1
        
        for action in actions[1:]:
            if action == current_action:
                count += 1
            else:
                name = action_names.get(current_action, f"Action{current_action}")
                result.append(f"{name}×{count}" if count > 1 else name)
                current_action = action
                count = 1
        
        # Add last group
        name = action_names.get(current_action, f"Action{current_action}")
        result.append(f"{name}×{count}" if count > 1 else name)
        
        return ", ".join(result)


def start_episode_log(episode_id: str, task_type: str, instruction: str, 
                     split: str, log_dir: str = 'data/logs/episodes_3d') -> EpisodeLogger3D:
    """
    Start logging a new episode.
    
    Args:
        episode_id: Unique episode identifier
        task_type: ALFRED task type
        instruction: Natural language instruction
        split: Dataset split
        log_dir: Base directory for logs
        
    Returns:
        logger: Episode logger instance
    """
    return EpisodeLogger3D(episode_id, task_type, instruction, split, log_dir)


# Test
if __name__ == '__main__':
    print("Testing EpisodeLogger3D...")
    
    # Create test logger
    logger = start_episode_log(
        episode_id="test_001",
        task_type="pick_and_place_simple",
        instruction="Pick up the apple and place it on the table",
        split="valid_seen"
    )
    
    # Log some steps
    for i in range(15):
        frame = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        action = np.random.randint(0, 7)
        reward = 0.0 if i < 14 else 1.0
        done = (i == 14)
        
        logger.log_step(frame, action, reward, done, {})
    
    # Finish
    metadata = logger.finish(success=False, failure_reason="Could not find apple")
    
    print(f"\n✓ Test complete")
    print(f"  Episode: {metadata['episode_id']}")
    print(f"  Steps: {metadata['total_steps']}")
    print(f"  Frames saved: {metadata['num_frames_saved']}")
    print(f"  Actions: {metadata['actions_compact']}")
