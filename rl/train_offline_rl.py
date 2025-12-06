#!/usr/bin/env python3
"""
Train Conservative Q-Learning (CQL) on ALFRED offline dataset.
"""

import os
import pickle
import json
import numpy as np
from pathlib import Path
import d3rlpy
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import CQLConfig
from d3rlpy.metrics import TDErrorEvaluator, DiscountedSumOfAdvantageEvaluator

def load_offline_dataset(dataset_path):
    """Load offline dataset (episodes pickle)."""
    print(f"Loading episodes from {dataset_path}...")
    
    with open(dataset_path, 'rb') as f:
        episodes = pickle.load(f)
    
    print(f"✓ Episodes loaded:")
    print(f"  Num episodes: {len(episodes)}")
    print(f"  Total transitions: {sum(len(ep) for ep in episodes)}")
    print(f"  Observation shape: {episodes[0].observations.shape}")
    
    return episodes

def create_mdp_dataset(episodes):
    """Create MDPDataset from episodes."""
    print("\nCreating MDPDataset...")
    
    # d3rlpy can work with replay buffer created from episodes
    from d3rlpy.dataset import create_fifo_replay_buffer
    
    # Create replay buffer
    buffer = create_fifo_replay_buffer(
        limit=sum(len(ep) for ep in episodes) + 1000,
        episodes=episodes
    )
    
    print(f"✓ Replay buffer created:")
    print(f"  Transitions: {sum(len(ep) for ep in episodes)}")
    
    return buffer, episodes

def train_cql(buffer, episodes, action_dim, output_dir, n_steps=100000):
    """Train CQL algorithm."""
    print(f"\n{'='*60}")
    print(f"Training Conservative Q-Learning (CQL)")
    print(f"{'='*60}\n")
    
    # CQL configuration
    # DiscreteCQLConfig uses different parameters than continuous CQL
    from d3rlpy.algos import DiscreteCQLConfig
    
    cql = DiscreteCQLConfig(
        learning_rate=3e-4,
        batch_size=256,
        alpha=1.0,  # Conservative weight
    ).create(device='cuda:0')
    
    print("CQL Configuration:")
    print(f"  Learning rate: 3e-4")
    print(f"  Batch size: 256")
    print(f"  Conservative weight (alpha): 1.0")
    print(f"  Training steps: {n_steps:,}")
    print(f"  Device: cuda:0")
    print(f"  Action dim: {action_dim}")
    
    # Setup logging
    os.makedirs(output_dir, exist_ok=True)
    
    # Train
    print(f"\nTraining for {n_steps:,} steps...")
    print("(This may take 2-4 hours)\n")
    
    cql.fit(
        buffer,
        n_steps=n_steps,
        n_steps_per_epoch=10000,
        save_interval=10,
        experiment_name='cql_alfred_offline',
        with_timestamp=False,
        show_progress=True
    )
    
    # Save final model
    model_path = Path(output_dir) / 'cql_final.d3'
    cql.save(model_path)
    print(f"\n✓ Model saved to: {model_path}")
    
    return cql

def save_training_summary(output_dir, n_steps, episodes):
    """Save training summary."""
    total_transitions = sum(len(ep) for ep in episodes)
    total_reward = sum(ep.rewards.sum() for ep in episodes)
    
    summary = {
        'algorithm': 'Conservative Q-Learning (CQL) - Discrete',
        'training_steps': n_steps,
        'batch_size': 256,
        'actor_lr': 1e-4,
        'critic_lr': 3e-4,
        'conservative_weight': 1.0,
        'dataset': {
            'num_transitions': total_transitions,
            'num_episodes': len(episodes),
            'observation_dim': episodes[0].observations.shape[1],
            'action_dim': 7,
            'avg_reward': float(total_reward / len(episodes))
        },
        'output': {
            'checkpoint': 'models/offline_rl_cql/cql_policy.d3',
            'logs': 'exp/offline_rl_cql/'
        }
    }
    
    summary_file = Path(output_dir) / 'training_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Training summary saved to: {summary_file}")

def main():
    # Paths
    dataset_path = 'data/offline_rl/train_episodes.pkl'
    output_dir = 'exp/offline_rl_cql'
    model_dir = 'models/offline_rl_cql'
    
    print("="*60)
    print("ALFRED Offline RL Training (CQL)")
    print("="*60)
    
    # Load dataset
    episodes = load_offline_dataset(dataset_path)
    
    # Create replay buffer
    buffer, _ = create_mdp_dataset(episodes)
    
    # Train CQL
    cql = train_cql(buffer, episodes, action_dim=7, output_dir=output_dir, n_steps=100000)
    
    # Save to models directory
    os.makedirs(model_dir, exist_ok=True)
    final_model = Path(model_dir) / 'cql_policy.d3'
    cql.save(final_model)
    print(f"\n✓ Final model copied to: {final_model}")
    
    # Save summary
    save_training_summary(model_dir, n_steps=100000, episodes=episodes)
    
    print("\n" + "="*60)
    print("✓ Training Complete!")
    print("="*60)
    print(f"Model checkpoint: {final_model}")
    print(f"Training logs: {output_dir}/")

if __name__ == '__main__':
    main()
