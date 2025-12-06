#!/usr/bin/env python3
"""
Build offline RL dataset using d3rlpy's native Episode format.

This avoids pickle save issues by using d3rlpy's built-in dataset handling.
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import d3rlpy
from d3rlpy.dataset import Episode, MDPDataset


def build_action_vocab(data_path, splits_path):
    """Build action vocabulary from the dataset."""
    with open(splits_path) as f:
        splits = json.load(f)
    
    action_set = set()
    
    # Sample episodes to get action vocabulary
    for episode in splits['train'][:100]:
        try:
            task_path = episode['task']
            traj_json = Path(data_path) / 'train' / task_path / 'traj_data.json'
            
            with open(traj_json) as f:
                traj_data = json.load(f)
            
            for action in traj_data['plan']['low_actions']:
                action_name = action['discrete_action']['action']
                action_set.add(action_name)
        except:
            continue
    
    action_list = sorted(list(action_set))
    action_to_idx = {a: i for i, a in enumerate(action_list)}
    
    print(f"Found {len(action_to_idx)} unique actions:")
    for a in action_list:
        print(f"  - {a}")
    
    return action_to_idx


def compute_reward(step_idx, total_steps, high_idx, prev_high_idx):
    """Compute shaped reward for a step."""
    reward = -0.01  # Step penalty
    
    # Subgoal completion bonus
    if high_idx > prev_high_idx:
        reward += 1.0
    
    # Task completion bonus (final step)
    if step_idx == total_steps - 1:
        reward += 5.0
    
    return reward


def process_episode(task_path, split, data_path, action_to_idx):
    """Process a single episode into d3rlpy Episode format."""
    try:
        # Load trajectory data
        traj_json = Path(data_path) / split / task_path / 'traj_data.json'
        with open(traj_json) as f:
            traj_data = json.load(f)
        
        # Load visual features
        feat_path = Path(data_path) / split / task_path / 'feat_conv.pt'
        if not feat_path.exists():
            return None
        
        features = torch.load(feat_path, weights_only=False)
        
        # Extract action sequence
        low_actions = traj_data['plan']['low_actions']
        
        # Check alignment
        num_actions = len(low_actions)
        num_frames = features.shape[0]
        
        if num_actions + 1 != num_frames:
            return None
        
        # Build episode data
        observations = []
        actions = []
        rewards = []
        
        prev_high_idx = -1
        for t, action_data in enumerate(low_actions):
            # Observation (flattened ResNet features)
            obs = features[t].flatten().cpu().numpy()
            observations.append(obs)
            
            # Action
            action_name = action_data['discrete_action']['action']
            if action_name not in action_to_idx:
                return None  # Skip episodes with unknown actions
            action_idx = action_to_idx[action_name]
            actions.append(action_idx)
            
            # Reward
            high_idx = action_data['high_idx']
            reward = compute_reward(t, num_actions, high_idx, prev_high_idx)
            rewards.append(reward)
            prev_high_idx = high_idx
        
        # Convert to numpy arrays
        observations = np.array(observations, dtype=np.float32)
        # d3rlpy expects all arrays as 2D: actions (N, 1), rewards (N, 1)
        actions = np.array(actions, dtype=np.int64).reshape(-1, 1)
        rewards = np.array(rewards, dtype=np.float32).reshape(-1, 1)
        
        # Create d3rlpy Episode
        episode = Episode(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminated=True  # All IL demonstrations complete successfully
        )
        
        return episode
        
    except Exception as e:
        return None


def build_dataset(data_path, splits_path, split, action_to_idx):
    """Build MDPDataset from a split."""
    print(f"\n{'='*60}")
    print(f"Building MDPDataset for {split}")
    print(f"{'='*60}\n")
    
    with open(splits_path) as f:
        splits_data = json.load(f)
    
    episode_list = splits_data[split]
    print(f"Processing {len(episode_list)} episodes...")
    
    episodes = []
    success_count = 0
    error_count = 0
    
    for ep_data in tqdm(episode_list, desc=f"Processing {split}"):
        task_path = ep_data['task']
        episode = process_episode(task_path, split, data_path, action_to_idx)
        
        if episode is not None:
            episodes.append(episode)
            success_count += 1
        else:
            error_count += 1
    
    print(f"\n{'='*60}")
    print(f"Dataset Construction Complete")
    print(f"{'='*60}")
    print(f"Successful episodes: {success_count}")
    print(f"Failed episodes: {error_count}")
    
    if not episodes:
        raise ValueError(f"No valid episodes found for {split}")
    
    # Create MDPDataset from list of Episode objects
    # d3rlpy 2.5.0 uses MDPDataset(observations, actions, rewards, terminals)
    # but we can use MDPDataset.from_episodes for a list of Episode objects
    from d3rlpy.dataset import create_fifo_replay_buffer
    dataset = create_fifo_replay_buffer(limit=len(episodes) * 1000, episodes=episodes)
    
    # Get dataset statistics
    total_steps = sum(len(ep) for ep in episodes)
    print(f"Total transitions: {total_steps}")
    print(f"Observation shape: {episodes[0].observations.shape}")
    print(f"Action range: [0, {max(action_to_idx.values())}]")
    
    return dataset, episodes


def save_dataset(episodes, output_path, split, action_to_idx):
    """Save episodes list directly - d3rlpy can load from pickle."""
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save episodes as pickle (simpler for d3rlpy)
    import pickle
    dataset_file = output_dir / f'{split}_episodes.pkl'
    with open(dataset_file, 'wb') as f:
        pickle.dump(episodes, f)
    
    print(f"\n✓ Episodes saved to: {dataset_file}")
    print(f"  Size: {dataset_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Test loading
    print(f"\nTesting episodes load...")
    with open(dataset_file, 'rb') as f:
        test_episodes = pickle.load(f)
    print(f"✓ Episodes load successfully!")
    print(f"  Verified {len(test_episodes)} episodes")
    
    # Save metadata
    total_transitions = sum(len(ep) for ep in episodes)
    metadata = {
        'split': split,
        'num_episodes': len(episodes),
        'num_transitions': total_transitions,
        'observation_dim': int(episodes[0].observations.shape[1]),
        'action_dim': len(action_to_idx),
        'action_vocab': action_to_idx,
        'reward_structure': {
            'task_completion': 5.0,
            'subgoal_completion': 1.0,
            'step_penalty': -0.01
        }
    }
    
    metadata_file = output_dir / f'{split}_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Metadata saved to: {metadata_file}")
    
    return dataset_file


def main():
    # Paths
    data_path = 'data/json_feat_subset'
    splits_path = 'data/splits/subset_oct21.json'
    output_path = 'data/offline_rl'
    
    print("="*60)
    print("ALFRED Offline RL Dataset Builder (d3rlpy native)")
    print("="*60)
    
    # Build action vocabulary
    print("\nBuilding action vocabulary...")
    action_to_idx = build_action_vocab(data_path, splits_path)
    
    # Save action vocab
    vocab_file = Path(output_path) / 'action_vocab.json'
    vocab_file.parent.mkdir(parents=True, exist_ok=True)
    with open(vocab_file, 'w') as f:
        json.dump(action_to_idx, f, indent=2)
    print(f"✓ Action vocab saved to: {vocab_file}")
    
    # Build training dataset
    print("\n" + "="*60)
    print("Building TRAINING dataset")
    print("="*60)
    train_buffer, train_episodes = build_dataset(data_path, splits_path, 'train', action_to_idx)
    train_file = save_dataset(train_episodes, output_path, 'train', action_to_idx)
    
    # Build validation datasets
    print("\n" + "="*60)
    print("Building VALIDATION datasets")
    print("="*60)
    
    for split in ['valid_seen', 'valid_unseen']:
        buffer, episodes = build_dataset(data_path, splits_path, split, action_to_idx)
        save_dataset(episodes, output_path, split, action_to_idx)
    
    print("\n" + "="*60)
    print("✓ All datasets built successfully!")
    print("="*60)
    print(f"\nDatasets saved as episode pickles:")
    print(f"  - {output_path}/train_episodes.pkl")
    print(f"  - {output_path}/valid_seen_episodes.pkl")
    print(f"  - {output_path}/valid_unseen_episodes.pkl")
    print(f"\nThese can be loaded with pickle.load() and used with d3rlpy")


if __name__ == '__main__':
    main()
