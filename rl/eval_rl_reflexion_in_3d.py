#!/usr/bin/env python3
"""
Evaluate RL vs RL+Reflexion in Real 3D Simulator

Compares baseline RL (CQL) with RL+Reflexion across same episodes.
"""

import sys
import os
import json
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

#  Add project to path
sys.path.insert(0, '/home/lilmeow/RL-project')

from env.wrappers.alfred_sim_env_3d import AlfredSimEnv3D
from rl.policies.rl_sim_policy_wrapper import RLSimPolicyWrapper
from rl.policies.rl_reflexion_sim_policy_wrapper import RLReflexionSimPolicyWrapper


def evaluate_policy(policy, policy_name, num_episodes=30, max_steps=100, split='valid_seen'):
    """
    Evaluate a policy in the real 3D simulator.
    
    Args:
        policy: Policy wrapper
        policy_name: Name for logging
        num_episodes: Number of episodes
        max_steps: Max steps per episode
        split: Dataset split
        
    Returns:
        metrics: Evaluation metrics dict
    """
    print(f"\n{'='*70}")
    print(f"Evaluating {policy_name} in Real AI2-THOR 3D")
    print(f"{'='*70}")
    
    # Initialize environment
    env = AlfredSimEnv3D(max_steps=max_steps)
    
    episode_data = []
    success_count = 0
    
    for ep_idx in tqdm(range(num_episodes), desc=f"{policy_name}"):
        try:
            # Reset environment
            obs = env.reset(split=split, episode_idx=ep_idx)
            
            # Start episode for policy
            episode_id = f"{split}_{ep_idx:04d}"
            if hasattr(policy, 'start_episode'):
                policy.start_episode(episode_id=episode_id)
            
            total_reward = 0.0
            done = False
            step = 0
            actions_taken = []
            
            while not done and step < max_steps:
                # Get action from policy
                if hasattr(policy, 'predict'):
                    # RL+Reflexion passes reward
                    if 'Reflexion' in policy_name:
                        action = policy.predict(obs['frame'], reward=total_reward)
                    else:
                        action = policy.predict(obs['frame'])
                else:
                    action = policy(obs['frame'])
                
                # Step environment
                obs, reward, done, info = env.step(action)
                total_reward += reward
                actions_taken.append(action)
                step += 1
            
            # Check success
            success = info.get('goal_satisfied', False)
            if success:
                success_count += 1
            
            episode_data.append({
                'episode_idx': ep_idx,
                'episode_id': episode_id,
                'success': success,
                'steps': step,
                'total_reward': total_reward,
                'actions': actions_taken
            })
            
            # Print progress
            status = "✓" if success else "✗"
            tqdm.write(f"  Episode {ep_idx}: {status} {step} steps, reward={total_reward:.1f}")
            
        except Exception as e:
            print(f"\n⚠ Error in episode {ep_idx}: {e}")
            continue
    
    # Cleanup
    env.close()
    
    # Compute metrics
    success_rate = success_count / len(episode_data) if episode_data else 0.0
    avg_steps = np.mean([ep['steps'] for ep in episode_data]) if episode_data else 0.0
    avg_reward = np.mean([ep['total_reward'] for ep in episode_data]) if episode_data else 0.0
    
    metrics = {
        'policy': policy_name,
        'num_episodes': len(episode_data),
        'success_rate': float(success_rate),
        'success_count': success_count,
        'avg_steps': float(avg_steps),
        'avg_reward': float(avg_reward),
        'episodes': episode_data
    }
    
    # Print summary
    print(f"\n{policy_name} Results:")
    print(f"  Success Rate: {success_rate*100:.1f}% ({success_count}/{len(episode_data)})")
    print(f"  Avg Steps: {avg_steps:.1f}")
    print(f"  Avg Reward: {avg_reward:.2f}")
    
    return metrics


def compare_policies(rl_metrics, reflexion_metrics, output_path='data/logs/rl_vs_reflexion_3d_summary.json'):
    """
    Compare RL baseline vs RL+Reflexion.
    
    Args:
        rl_metrics: RL baseline metrics
        reflexion_metrics: RL+Reflexion metrics
        output_path: Output file
        
    Returns:
        comparison: Comparison dict
    """
    print(f"\n{'='*70}")
    print("RL vs RL+Reflexion Comparison")
    print(f"{'='*70}")
    
    # Overall comparison
    improvement = reflexion_metrics['success_rate'] - rl_metrics['success_rate']
    improvement_pct = (improvement / rl_metrics['success_rate'] * 100) if rl_metrics['success_rate'] > 0 else 0
    
    print(f"\nOverall Performance:")
    print(f"  RL:            {rl_metrics['success_rate']*100:.1f}% ({rl_metrics['success_count']}/{rl_metrics['num_episodes']})")
    print(f"  RL+Reflexion: {reflexion_metrics['success_rate']*100:.1f}% ({reflexion_metrics['success_count']}/{reflexion_metrics['num_episodes']})")
    print(f"  Improvement:   {improvement*100:+.1f}% ({improvement_pct:+.1f}% relative)")
    
    # Find episodes where Reflexion helped
    helped_episodes = []
    hurt_episodes = []
    
    for rl_ep, ref_ep in zip(rl_metrics['episodes'], reflexion_metrics['episodes']):
        if rl_ep['episode_idx'] == ref_ep['episode_idx']:
            if not rl_ep['success'] and ref_ep['success']:
                helped_episodes.append({
                    'episode_idx': rl_ep['episode_idx'],
                    'episode_id': rl_ep['episode_id'],
                    'rl_steps': rl_ep['steps'],
                    'reflexion_steps': ref_ep['steps'],
                    'reflexion_reward': ref_ep['total_reward']
                })
            elif rl_ep['success'] and not ref_ep['success']:
                hurt_episodes.append({
                    'episode_idx': rl_ep['episode_idx'],
                    'episode_id': rl_ep['episode_id']
                })
    
    print(f"\nEpisode-Level Comparison:")
    print(f"  Reflexion helped: {len(helped_episodes)} episodes")
    print(f"  Reflexion hurt:   {len(hurt_episodes)} episodes")
    
    if helped_episodes:
        print(f"\n  Examples where Reflexion helped:")
        for ep in helped_episodes[:3]:
            print(f"    - Episode {ep['episode_idx']}: RL failed, Reflexion succeeded in {ep['reflexion_steps']} steps")
    
    # Save comparison
    comparison = {
        'rl_baseline': {
            'success_rate': rl_metrics['success_rate'],
            'success_count': rl_metrics['success_count'],
            'num_episodes': rl_metrics['num_episodes']
        },
        'rl_reflexion': {
            'success_rate': reflexion_metrics['success_rate'],
            'success_count': reflexion_metrics['success_count'],
            'num_episodes': reflexion_metrics['num_episodes']
        },
        'improvement': {
            'absolute': float(improvement),
            'relative_pct': float(improvement_pct)
        },
        'helped_episodes': helped_episodes,
        'hurt_episodes': hurt_episodes
    }
    
    # Save
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n✓ Comparison saved to: {output_path}")
    
    return comparison


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Evaluate RL vs RL+Reflexion in 3D')
    parser.add_argument('--mode', type=str, default='both', choices=['rl', 'reflexion', 'both'],
                       help='Evaluation mode (default: both)')
    parser.add_argument('--num-episodes', type=int, default=30,
                       help='Number of episodes (default: 30)')
    parser.add_argument('--split', type=str, default='valid_seen',
                       choices=['valid_seen', 'valid_unseen'],
                       help='Dataset split (default: valid_seen)')
    
    args = parser.parse_args()
    
    try:
        rl_metrics = None
        reflexion_metrics = None
        
        # Evaluate RL baseline
        if args.mode in ['rl', 'both']:
            print("\n" + "="*70)
            print("PHASE 1: RL Baseline Evaluation")
            print("="*70)
            
            rl_policy = RLSimPolicyWrapper(checkpoint_path='models/offline_rl_cql/cql_policy.d3')
            rl_metrics = evaluate_policy(
                rl_policy,
                'RL (CQL)',
                num_episodes=args.num_episodes,
                split=args.split
            )
            
            # Save RL metrics
            with open('data/logs/rl_3d_baseline_metrics.json', 'w') as f:
                json.dump(rl_metrics, f, indent=2)
        
        # Evaluate RL+Reflexion
        if args.mode in ['reflexion', 'both']:
            print("\n" + "="*70)
            print("PHASE 2: RL+Reflexion Evaluation")
            print("="*70)
            
            reflexion_policy = RLReflexionSimPolicyWrapper()
            reflexion_metrics = evaluate_policy(
                reflexion_policy,
                'RL+Reflexion',
                num_episodes=args.num_episodes,
                split=args.split
            )
            
            # Save Reflexion metrics
            with open('data/logs/rl_reflexion_3d_metrics.json', 'w') as f:
                json.dump(reflexion_metrics, f, indent=2)
        
        # Compare if both evaluated
        if args.mode == 'both' and rl_metrics and reflexion_metrics:
            comparison = compare_policies(rl_metrics, reflexion_metrics)
        
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
