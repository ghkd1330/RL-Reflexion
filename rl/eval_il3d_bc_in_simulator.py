#!/usr/bin/env python3
"""
Evaluate IL3D-BC Policy in Real 3D Simulator

Evaluates the trained behavior cloning policy in the real AI2-THOR 3D environment.
"""

import sys
import os
import json
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Add project to path
sys.path.insert(0, '/home/lilmeow/RL-project')

from env.wrappers.alfred_sim_env_3d import AlfredSimEnv3D
from rl.policies.il3d_bc_sim_policy_wrapper import IL3DBC3DSimPolicy


def evaluate_il3d_bc(num_episodes=20, max_steps=80, split='valid_seen', headless=False):
    """
    Evaluate IL3D-BC policy in real 3D simulator.
    
    Args:
        num_episodes: Number of episodes to evaluate
        max_steps: Maximum steps per episode
        split: Dataset split
        headless: Run headless with Xvfb
        
    Returns:
        metrics: Evaluation metrics dict
    """
    print("="*70)
    print("Evaluating IL3D-BC in Real AI2-THOR 3D Simulator")
    print("="*70)
    
    # Initialize environment
    print(f"\nInitializing 3D environment...")
    if headless:
        from xvfbwrapper import Xvfb
        vdisplay = Xvfb(width=1024, height=768)
        vdisplay.start()
    else:
        vdisplay = None
        print("Running with visual display (Unity window will appear)")
    
    env = AlfredSimEnv3D(max_steps=max_steps)
    print("✓ Environment initialized")
    
    # Initialize policy
    print(f"\nLoading IL3D-BC policy...")
    policy = IL3DBC3DSimPolicy()
    print("✓ Policy loaded")
    
    # Evaluation loop
    print(f"\nRunning {num_episodes} episodes on {split}...")
    print("This may take several minutes...\n")
    
    episode_data = []
    success_count = 0
    
    for ep_idx in tqdm(range(num_episodes), desc="Episodes"):
        try:
            # Reset environment
            obs = env.reset(split=split, episode_idx=ep_idx)
            
            done = False
            steps = 0
            total_reward = 0.0
            
            # Episode loop
            while not done and steps < max_steps:
                # Get instruction
                instruction = obs.get('instruction', '')
                
                # Predict action (vision-only, instruction ignored)
                action_id = policy.predict(obs['frame'], instruction)
                
                # Step environment
                obs, reward, done, info = env.step(action_id)
                total_reward += reward
                steps += 1
            
            # Check success
            success = info.get('goal_satisfied', False)
            if success:
                success_count += 1
            
            # Log episode
            episode_data.append({
                'episode_idx': ep_idx,
                'split': split,
                'success': success,
                'steps': steps,
                'total_reward': total_reward
            })
            
            # Print progress
            status = "✓" if success else "✗"
            tqdm.write(f"  Episode {ep_idx}: {status} {steps} steps, reward={total_reward:.1f}")
            
        except Exception as e:
            print(f"\n✗ Error in episode {ep_idx}: {e}")
            continue
    
    # Cleanup
    env.close()
    if vdisplay is not None:
        vdisplay.stop()
    
    # Compute metrics
    success_rate = success_count / len(episode_data) if episode_data else 0.0
    avg_steps = np.mean([ep['steps'] for ep in episode_data]) if episode_data else 0.0
    avg_reward = np.mean([ep['total_reward'] for ep in episode_data]) if episode_data else 0.0
    
    metrics = {
        'agent_type': 'il3d_bc',
        'num_episodes': len(episode_data),
        'success_rate': float(success_rate),
        'success_count': success_count,
        'avg_steps': float(avg_steps),
        'avg_reward': float(avg_reward),
        'split': split,
        'episodes': episode_data
    }
    
    # Print summary
    print(f"\n{'='*70}")
    print("IL3D-BC Evaluation Results (Real 3D Simulator)")
    print(f"{'='*70}")
    print(f"Episodes: {len(episode_data)}")
    print(f"Success Rate: {success_rate*100:.1f}% ({success_count}/{len(episode_data)})")
    print(f"Avg Steps: {avg_steps:.1f}")
    print(f"Avg Reward: {avg_reward:.2f}")
    print(f"{'='*70}")
    
    return metrics


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Evaluate IL3D-BC in real 3D simulator')
    parser.add_argument('--num-episodes', type=int, default=20,
                       help='Number of episodes (default: 20)')
    parser.add_argument('--max-steps', type=int, default=80,
                       help='Max steps per episode (default: 80)')
    parser.add_argument('--split', type=str, default='valid_seen',
                       choices=['valid_seen', 'valid_unseen'],
                       help='Dataset split (default: valid_seen)')
    parser.add_argument('--visual', action='store_true',
                       help='Keep Unity window visible (default: True)')
    parser.add_argument('--headless', action='store_true',
                       help='Run headless with Xvfb')
    parser.add_argument('--output', type=str, default='data/logs/il3d_bc_3d_metrics.json',
                       help='Output path for metrics')
    
    args = parser.parse_args()
    
    try:
        # Run evaluation
        metrics = evaluate_il3d_bc(
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            split=args.split,
            headless=args.headless
        )
        
        # Save metrics
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n✓ Metrics saved to: {output_path}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
