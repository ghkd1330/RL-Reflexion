#!/usr/bin/env python3
"""
Generate Reflexion Rules from Real 3D RL Failures using Qwen2-VL

Analyzes failed RL episodes in the real AI2-THOR 3D simulator and generates
actionable rules using the real Qwen2-VL-7B model.
"""

import sys
import os
import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# Add project to path
sys.path.insert(0, '/home/lilmeow/RL-project')

from reflection.vlm_client_real import Qwen2VLClientReal


def load_failed_episodes(episodes_dir: str = 'data/logs/episodes_3d') -> List[Dict]:
    """
    Load failed episodes from logged 3D runs.
    
    Args:
        episodes_dir: Base directory with episode logs
        
    Returns:
        failed_episodes: List of failed episode metadata dicts
    """
    episodes_path = Path(episodes_dir)
    
    if not episodes_path.exists():
        print(f"⚠ Warning: Episodes directory not found: {episodes_dir}")
        return []
    
    failed_episodes = []
    
    for episode_folder in sorted(episodes_path.iterdir()):
        if not episode_folder.is_dir():
            continue
        
        meta_file = episode_folder / 'meta.json'
        if not meta_file.exists():
            continue
        
        with open(meta_file) as f:
            meta = json.load(f)
        
        # Only process failures
        if not meta.get('success', True):
            # Add full paths to frames
            meta['frames_full'] = [
                str(episode_folder / frame) for frame in meta.get('frames', [])
            ]
            failed_episodes.append(meta)
    
    return failed_episodes


def generate_rules_for_episode(vlm: Qwen2VLClientReal, episode: Dict) -> Dict:
    """
    Generate Reflexion rules for a failed episode using real VLM.
    
    Args:
        vlm: Real Qwen2-VL client
        episode: Episode metadata dict
        
    Returns:
        rules_data: Dict with failure_reason and rules list
    """
    # Select key frames (first, middle, last)
    frames = episode['frames_full']
    if len(frames) == 0:
        return {'failure_reason': 'No frames available', 'rules': []}
    
    selected_frames = []
    if len(frames) == 1:
        selected_frames = frames
    elif len(frames) == 2:
        selected_frames = frames
    else:
        # First, middle, last
        selected_frames = [frames[0], frames[len(frames)//2], frames[-1]]
    
    # Build prompt
    prompt = f"""Task type: {episode['task_type']}
Split: {episode['split']}
Instruction: {episode['instruction']}
Success: False

The robot failed to complete this task in the real 3D AI2-THOR environment.
Below are {len(selected_frames)} frames from the episode.

Actions taken (compact): {episode['actions_compact']}
Total steps: {episode['total_steps']}

Please:
1. Explain briefly (1-2 sentences) why the attempt likely failed.
2. Propose 3-5 concise, actionable rules that would help the robot succeed next time.
   Each rule should be one sentence, imperative style (e.g., "Always do X before Y").

Format your response as:
FAILURE REASON: [your explanation]

RULES:
1. [rule 1]
2. [rule 2]
3. [rule 3]
..."""
    
    print(f"\n  Generating rules for episode {episode['episode_id']}...")
    print(f"    Frames: {len(selected_frames)}")
    print(f"    Actions: {episode['actions_compact']}")
    
    try:
        # Call real VLM
        output = vlm.generate(selected_frames, prompt, max_new_tokens=512)
        
        # Parse output
        rules_data = parse_vlm_output(output)
        
        print(f"    ✓ Generated {len(rules_data['rules'])} rules")
        
        return rules_data
        
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return {'failure_reason': f'VLM error: {e}', 'rules': []}


def parse_vlm_output(output: str) -> Dict:
    """
    Parse VLM output to extract failure reason and rules.
    
    Args:
        output: Raw VLM text output
        
    Returns:
        parsed: Dict with 'failure_reason' and 'rules' list
    """
    # Extract failure reason
    failure_match = re.search(r'FAILURE REASON:\s*(.+?)(?:\n\n|RULES:)', output, re.DOTALL | re.IGNORECASE)
    failure_reason = failure_match.group(1).strip() if failure_match else "Unknown"
    
    # Extract rules (numbered or bulleted)
    rules = []
    rules_section = re.search(r'RULES:(.+)$', output, re.DOTALL | re.IGNORECASE)
    
    if rules_section:
        rules_text = rules_section.group(1)
        
        # Match numbered rules (1. , 2. , etc) or bulleted (- , * )
        for line in rules_text.split('\n'):
            line = line.strip()
            
            # Match "1. Rule text" or "- Rule text"
            match = re.match(r'^(?:\d+\.|-|\*)\s*(.+)$', line)
            if match:
                rule = match.group(1).strip()
                if len(rule) > 10:  # Filter out too-short lines
                    rules.append(rule)
    
    # Fallback: if no structured rules found, try to extract sentences
    if not rules:
        sentences = [s.strip() for s in output.split('.') if len(s.strip()) > 20]
        rules = sentences[:5]  # Take up to 5
    
    return {
        'failure_reason': failure_reason,
        'rules': rules[:5]  # Limit to 5 rules
    }


def generate_rules_for_all_failures(episodes_dir: str = 'data/logs/episodes_3d',
                                    output_path: str = 'data/rules/rule_database_real_3d.json'):
    """
    Generate rules for all failed episodes and save to database.
    
    Args:
        episodes_dir: Directory with episode logs
        output_path: Output path for rule database
    """
    print("="*70)
    print("Generating Reflexion Rules from Real 3D Failures using Qwen2-VL")
    print("="*70)
    
    # Load VLM
    print("\nLoading real Qwen2-VL client...")
    vlm = Qwen2VLClientReal()
    
    # Load failed episodes
    print(f"\nLoading failed episodes from {episodes_dir}...")
    failed_episodes = load_failed_episodes(episodes_dir)
    
    if not failed_episodes:
        print("✗ No failed episodes found. Run RL evaluation with --log-episodes first.")
        return
    
    print(f"✓ Found {len(failed_episodes)} failed episodes")
    
    # Generate rules for each failure
    print(f"\nGenerating rules for failures...")
    
    rule_database = {
        'metadata': {
            'source': 'ai2thor_3d_rl_failures',
            'model': 'Qwen2-VL-7B-Instruct-4bit',
            'generated_at': datetime.now().isoformat(),
            'num_episodes_analyzed': len(failed_episodes)
        },
        'episodes': {}
    }
    
    for episode in failed_episodes:
        episode_id = episode['episode_id']
        
        try:
            rules_data = generate_rules_for_episode(vlm, episode)
            
            rule_database['episodes'][episode_id] = {
                'task_type': episode['task_type'],
                'instruction': episode['instruction'],
                'split': episode['split'],
                'total_steps': episode['total_steps'],
                'actions_compact': episode['actions_compact'],
                'failure_reason': rules_data['failure_reason'],
                'rules': rules_data['rules']
            }
            
        except Exception as e:
            print(f"  ✗ Failed to process episode {episode_id}: {e}")
            continue
    
    # Save rule database
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(rule_database, f, indent=2)
    
    print(f"\n{'='*70}")
    print("Rule Generation Complete!")
    print(f"{'='*70}")
    print(f"\nProcessed: {len(rule_database['episodes'])} episodes")
    print(f"Saved to: {output_path}")
    
    # Print sample
    if rule_database['episodes']:
        sample_id = list(rule_database['episodes'].keys())[0]
        sample = rule_database['episodes'][sample_id]
        
        print(f"\nSample (Episode {sample_id}):")
        print(f"  Task: {sample['task_type']}")
        print(f"  Failure: {sample['failure_reason']}")
        print(f"  Rules:")
        for i, rule in enumerate(sample['rules'][:3], 1):
            print(f"    {i}. {rule}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Reflexion rules from 3D RL failures')
    parser.add_argument('--episodes-dir', type=str, default='data/logs/episodes_3d',
                       help='Directory with logged episodes')
    parser.add_argument('--output', type=str, default='data/rules/rule_database_real_3d.json',
                       help='Output path for rule database')
    
    args = parser.parse_args()
    
    try:
        generate_rules_for_all_failures(args.episodes_dir, args.output)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
