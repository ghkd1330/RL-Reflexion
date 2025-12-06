#!/usr/bin/env python3
"""
Simplified Offline Policy Evaluator.

Evaluates IL and RL policies on offline validation data and logs failures.
"""

import sys
import json
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from reflection.episode_logging import EpisodeLogger

try:
    import d3rlpy
except ImportError:
    print("Warning: d3rlpy not installed")
    d3rlpy = None


class SimplifiedEvaluator:
    """
    Evaluates policies on offline data and logs failures.
    """
    
    def __init__(self, data_path: str = 'data/offline_rl'):
        self.data_path = Path(data_path)
        self.logger = EpisodeLogger()
        
    def evaluate_policy(self, 
                       policy,
                       split: str = 'valid_seen',
                       policy_name: str = 'policy',
                       log_failures: bool = True) -> dict:
        """
        Evaluate policy on offline data.
        
        Args:
            policy: Policy object with predict() method
            split: Data split to evaluate
            policy_name: Name for logging
            log_failures: Whether to log failed episodes
            
        Returns:
            Metrics dict
        """
        print(f"\n{'='*60}")
        print(f"Evaluating {policy_name} on {split}")
        print(f"{'='*60}")
        
        # Load episodes
        episodes_file = self.data_path / f'{split}_episodes.pkl'
        with open(episodes_file, 'rb') as f:
            episodes = pickle.load(f)
        
        print(f"Loaded {len(episodes)} episodes")
        
        # Evaluate
        correct = 0
        total = 0
        episode_successes = []
        logged_failures = 0
        
        for ep_idx, episode in enumerate(tqdm(episodes, desc=f"Eval {policy_name}")):
            ep_correct = 0
            ep_total = 0
            
            obs = episode.observations
            actions = episode.actions
            
            # Predict on all steps
            for t in range(len(obs)):
                try:
                    # Predict
                    pred_action = policy.predict(obs[t:t+1])
                    if isinstance(pred_action, np.ndarray):
                        pred_action = pred_action[0][0] if pred_action.ndim > 1 else pred_action[0]
                    
                    # Compare
                    expert_action = actions[t][0]
                    if pred_action == expert_action:
                        correct += 1
                        ep_correct += 1
                    total += 1
                    ep_total += 1
                except:
                    total += 1
                    ep_total += 1
            
            # Episode success: >80% action agreement
            ep_success = (ep_correct / ep_total) > 0.8 if ep_total > 0 else False
            episode_successes.append(ep_success)
            
            # Log failures
            if log_failures and not ep_success and logged_failures < 30:
                self._log_failure(episode, ep_idx, split, policy_name)
                logged_failures += 1
        
        # Compute metrics
        action_accuracy = correct / total if total > 0 else 0
        success_rate = sum(episode_successes) / len(episode_successes) if episode_successes else 0
        
        metrics = {
            'policy': policy_name,
            'split': split,
            'action_accuracy': float(action_accuracy),
            'success_rate': float(success_rate),
            'num_episodes': len(episodes),
            'num_steps': int(total),
            'failures_logged': logged_failures
        }
        
        print(f"\n{policy_name} Results:")
        print(f"  Action Accuracy: {action_accuracy*100:.2f}%")
        print(f"  Success Rate (>80% agreement): {success_rate*100:.2f}%")
        print(f"  Failures Logged: {logged_failures}")
        
        return metrics
    
    def _log_failure(self, episode, ep_idx: int, split: str, policy_name: str):
        """Log a failed episode."""
        try:
            # Extract info
            obs = episode.observations
            actions = episode.actions
            rewards = episode.rewards
            
            # Convert observations to frames (placeholder - would need proper conversion)
            # For now, create simple visualizations from features
            frames = self._features_to_frames(obs)
            
            # Action names
            action_map = {
                0: 'LookDown_15',
                1: 'LookUp_15',
                2: 'MoveAhead_25',
                3: 'PickupObject',
                4: 'RotateLeft_90',
                5: 'RotateRight_90',
                6: 'ToggleObjectOn'
            }
            action_names = [action_map.get(int(a[0]), f"Action_{int(a[0])}") for a in actions]
            
            # Log
            episode_id = f"{split}_{policy_name}_ep{ep_idx:04d}"
            self.logger.log_episode(
                episode_id=episode_id,
                task_type='unknown',  # Would extract from metadata
                split=split,
                goal='Task goal (simplified evaluation)',
                instruction='Complete the task',
                frames=frames,
                actions=action_names,
                success=False,
                final_reward=float(rewards.sum()),
                additional_info={'policy': policy_name, 'episode_index': ep_idx}
            )
        except Exception as e:
            print(f"Warning: Could not log episode {ep_idx}: {e}")
    
    def _features_to_frames(self, features: np.ndarray, num_frames: int = 10) -> list:
        """Convert feature vectors to placeholder frames."""
        # Select evenly spaced frames
        indices = np.linspace(0, len(features)-1, min(num_frames, len(features)), dtype=int)
        
        frames = []
        for idx in indices:
            # Create a simple visualization (random for now, would use proper reconstruction)
            frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            frames.append(frame)
        
        return frames


def main():
    """Run simplified evaluation."""
    print("="*60)
    print("Simplified Offline Evaluation with Failure Logging")
    print("="*60)
    
    evaluator = SimplifiedEvaluator()
    
    # Evaluate RL policy
    if d3rlpy:
        print("\nLoading RL policy...")
        try:
            from rl.policies.rl_policy_wrapper import RLPolicyWrapper
            rl_policy = RLPolicyWrapper('models/offline_rl_cql/cql_policy.d3')
            
            # Evaluate
            rl_metrics = evaluator.evaluate_policy(
                policy=rl_policy,
                split='valid_seen',
                policy_name='RL_CQL',
                log_failures=True
            )
            
            # Save metrics
            metrics_file = Path('data/logs/simplified_rl_metrics.json')
            metrics_file.parent.mkdir(parents=True, exist_ok=True)
            with open(metrics_file, 'w') as f:
                json.dump(rl_metrics, f, indent=2)
            
            print(f"\n✓ Metrics saved to {metrics_file}")
            
        except Exception as e:
            print(f"Error evaluating RL policy: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    stats = evaluator.logger.get_statistics()
    print(f"\n{'='*60}")
    print("Episode Logging Summary")
    print(f"{'='*60}")
    print(f"Total logged: {stats['total_logged']}")
    print(f"Failed: {stats['failed']}")
    print(f"Successful: {stats['successful']}")


if __name__ == '__main__':
    main()
