#!/usr/bin/env python3
"""
Evaluate IL and RL Policies in Real AI2-THOR 3D Simulator

This script:
1. Loads the real ALFRED 3D simulator environment
2. Loads IL or RL policy
3. Runs episodes in the REAL simulator
4. Measures TRUE task success rates (not offline metrics)

CRITICAL: This uses REAL AI2-THOR, not the simplified evaluator.
"""

import argparse
import json
import sys
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Add project root to path
sys.path.insert(0, '/home/lilmeow/RL-project')

from env.wrappers.alfred_sim_env_3d import AlfredSimEnv3D
from rl.policies.il_real_sim_policy_wrapper import RealILSimPolicyWrapper
from rl.policies.rl_sim_policy_wrapper import RLSimPolicyWrapper
from reflection.episode_logging_3d import start_episode_log


def evaluate_agent_in_3d(agent_type: str, num_episodes: int = 10, 
                         split: str = 'valid_seen', headless: bool = False,
                         log_episodes: bool = False, log_dir: str = 'data/logs/episodes_3d'):
    """
    Evaluate agent in real 3D simulator.
    
    Args:
        agent_type: 'il' or 'rl'
        num_episodes: Number of episodes to run
        split: Dataset split
        headless: Whether to run headless (with Xvfb)
        
    Returns:
        metrics: Dictionary with success rate, avg steps, etc.
    """
    print("="*70)
    print(f"Evaluating {agent_type.upper()} Agent in Real AI2-THOR 3D Simulator")
    print("="*70)
    
    # Initialize environment
    print(f"\nInitializing ALFRED 3D environment...")
    if headless:
        print("Running in headless mode with Xvfb")
        from xvfbwrapper import Xvfb
        vdisplay = Xvfb(width=1024, height=768)
        vdisplay.start()
    else:
        vdisplay = None
        print("Running with visual display (Unity window will appear)")
    
    env = AlfredSimEnv3D(max_steps=100)
    print("✓ Environment initialized")
    
    # Load policy
    print(f"\nLoading {agent_type.upper()} policy...")
    if agent_type == 'il':
        policy = RealILSimPolicyWrapper('models/seq2seq_il_best/best_seen.pth')
    else:  # rl
        policy = RLSimPolicyWrapper('models/offline_rl_cql/cql_policy.d3')
    
    print(f"✓ Policy loaded")
    
    # Run episodes
    print(f"\nRunning {num_episodes} episodes on {split}...")
    print("This may take several minutes...\n")
    
    episode_data = []
    
    for ep_idx in tqdm(range(num_episodes), desc="Episodes"):
        try:
            # Reset environment
            obs = env.reset(split=split, episode_idx=ep_idx)
            
            # Start episode logging if enabled
            episode_logger = None
            if log_episodes:
                episode_id = f"{split}_{ep_idx:04d}"
                task_type = obs.get('task_type', 'unknown')
                instruction = obs.get('instruction', '')
                episode_logger = start_episode_log(episode_id, task_type, instruction, split, log_dir)
            
            total_reward = 0.0
            done = False
            step = 0
            
            while not done and step < 100:  # Max 100 steps per episode
                # Get current instruction
                instruction = obs.get('instruction', '')
                
                # Predict action (instruction-conditioned for IL)
                if agent_type == 'il':
                    action = policy.predict(obs['frame'], instruction)
                else:
                    action = policy.predict(obs['frame'])
                
                # Step environment
                obs, reward, done, info = env.step(action)
                total_reward += reward
                
                # Log step if enabled
                if episode_logger:
                    episode_logger.log_step(obs['frame'], action, reward, done, info)
                
                step += 1
            
            # Log episode result
            success = info.get('goal_satisfied', False)
            
            # Finish episode logging if enabled
            if episode_logger:
                episode_logger.finish(success, failure_reason=info.get('failure_reason', 'unknown'))
            
            episode_data.append({
                'episode_idx': ep_idx,
                'split': split,
                'success': success,
                'steps': step,
                'total_reward': total_reward
            })
            
            # Print progress
            status = "✓" if success else "✗"
            tqdm.write(f"  Episode {ep_idx}: {status} {step} steps, reward={total_reward:.1f}")
            
        except Exception as e:
            print(f"\nWarning: Could not load episode {ep_idx}: {e}")
            continue
    
    # Cleanup
    env.close()
    if vdisplay is not None:
        vdisplay.stop()
    
    # Compute metrics
    success_count = sum(1 for r in episode_data if r['success'])
    success_rate = success_count / len(episode_data) if episode_data else 0.0
    avg_steps = np.mean([r['steps'] for r in episode_data]) if episode_data else 0.0
    avg_reward = np.mean([r['total_reward'] for r in episode_data]) if episode_data else 0.0
    
    metrics = {
        'agent_type': agent_type,
        'split': split,
        'num_episodes': len(episode_data),
        'success_rate': float(success_rate),
        'success_count': success_count,
        'avg_steps': float(avg_steps),
        'avg_reward': float(avg_reward),
        'episodes': episode_data
    }
    
    # Print summary
    print("\n" + "="*70)
    print(f"{agent_type.upper()} Agent Results (Real 3D Simulator)")
    print("="*70)
    print(f"Episodes: {len(episode_data)}")
    print(f"Success Rate: {success_rate*100:.1f}% ({success_count}/{len(episode_data)})")
    print(f"Avg Steps: {avg_steps:.1f}")
    print(f"Avg Reward: {avg_reward:.2f}")
    print("="*70)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate IL or RL agent in real AI2-THOR 3D simulator'
    )
    parser.add_argument('--agent', type=str, required=True, choices=['il', 'rl'],
                        help='Agent type: il or rl')
    parser.add_argument('--num-episodes', type=int, default=10,
                        help='Number of episodes to evaluate (default: 10)')
    parser.add_argument('--split', type=str, default='valid_seen',
                        choices=['valid_seen', 'valid_unseen'],
                        help='Dataset split (default: valid_seen)')
    parser.add_argument('--headless', action='store_true',
                        help='Run headless with Xvfb (no visual window)')
    parser.add_argument('--log-episodes', action='store_true',
                        help='Enable episode logging for Reflexion')
    parser.add_argument('--log-dir', type=str, default='data/logs/episodes_3d',
                        help='Directory for episode logs (default: data/logs/episodes_3d)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for metrics (default: data/logs/{agent}_3d_metrics.json)')
    
    args = parser.parse_args()
    
    # Run evaluation
    try:
        metrics = evaluate_agent_in_3d(
            agent_type=args.agent,
            num_episodes=args.num_episodes,
            split=args.split,
            headless=args.headless,
            log_episodes=args.log_episodes,
            log_dir=args.log_dir
        )
        
        # Save metrics
        if args.output is None:
            output_file = Path(f'data/logs/{args.agent}_3d_metrics.json')
        else:
            output_file = Path(args.output)
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n✓ Metrics saved to: {output_file}")
        
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
