#!/usr/bin/env python3
"""
Reflexion Controller for 3D RL

Meta-controller that uses VLM-generated rules to modify RL agent behavior.
Implements keyword-based heuristics to prevent common failure patterns.
"""

import re
from typing import List, Dict, Optional
from collections import deque

import sys
sys.path.insert(0, '/home/lilmeow/RL-project')

from reflection.rule_db_real_3d import RuleDatabase3D


class ReflexionController3D:
    """
    Reflexion meta-controller for 3D RL agent.
    
    Parses VLM rules and applies action override heuristics.
    """
    
    # Action indices (0-6)
    ACTIONS = {
        'LookDown_15': 0,
        'LookUp_15': 1,
        'MoveAhead_25': 2,
        'PickupObject': 3,
        'RotateLeft_90': 4,
        'RotateRight_90': 5,
        'ToggleObjectOn': 6
    }
    
    def __init__(self, rule_db_path: str = 'data/rules/rule_database_real_3d.json'):
        """
        Initialize Reflexion controller.
        
        Args:
            rule_db_path: Path to rule database
        """
        self.rule_db = RuleDatabase3D(rule_db_path)
        
        # Episode state
        self.rules = []
        self.episode_step = 0
        self.action_history = deque(maxlen=10)
        self.reward_history = deque(maxlen=10)
        
        # Rule-based config (extracted from rules)
        self.enable_scan_rotation = False
        self.encourage_lookdown = False
        self.discourage_repeat_pickup = False
        self.encourage_toggle_light = False
        self.scan_complete = False
        
    def start_episode(self, task_type: Optional[str] = None, episode_id: Optional[str] = None):
        """
        Start a new episode and load relevant rules.
        
        Args:
            task_type: ALFRED task type
            episode_id: Specific episode ID
        """
        # Reset state
        self.episode_step = 0
        self.action_history.clear()
        self.reward_history.clear()
        self.scan_complete = False
        
        # Load rules
        if episode_id:
            self.rules = self.rule_db.get_episode_rules(episode_id)
        elif task_type:
            self.rules = self.rule_db.get_rules_for_task(task_type, top_k=5)
        else:
            self.rules = self.rule_db.get_all_rules(top_k=5)
        
        # Parse rules to extract heuristics
        self._parse_rules()
        
        print(f"  Reflexion: Loaded {len(self.rules)} rules")
        if self.enable_scan_rotation:
            print(f"    → Will perform 360° scan at start")
        if self.discourage_repeat_pickup:
            print(f"    → Will prevent repeated failed pickups")
        if self.encourage_lookdown:
            print(f"    → Will encourage LookDown when stuck")
        if self.encourage_toggle_light:
            print(f"    → Will try Toggle if struggling late")
    
    def _parse_rules(self):
        """
        Parse rules to extract action heuristics.
        
        Uses keyword matching to set behavioral flags.
        """
        rules_text = " ".join(self.rules).lower()
        
        # Heuristic 1: 360° scan / rotation
        # Keywords: "360", "rotation", "rotate", "scan", "check all", "surroundings"
        if any(kw in rules_text for kw in ['360', 'rotation', 'rotate all', 'scan', 'check all', 'surroundings']):
            self.enable_scan_rotation = True
        
        # Heuristic 2: LookDown encouragement
        # Keywords: "lookdown", "look down", "floor", "check below"
        if any(kw in rules_text for kw in ['lookdown', 'look down', 'floor', 'check below', 'search floor']):
            self.encourage_lookdown = True
        
        # Heuristic 3: Discourage repeated pickup
        # Keywords: "verify", "view", "before pickup", "don't repeat", "failed action"
        if any(kw in rules_text for kw in ['verify', 'view before', 'before pickup', "don't repeat", 'failed action', 'repeated']):
            self.discourage_repeat_pickup = True
        
        # Heuristic 4: Toggle light
        # Keywords: "light", "lamp", "toggle", "turn on"
        if any(kw in rules_text for kw in ['light', 'lamp', 'toggle', 'turn on']):
            self.encourage_toggle_light = True
    
    def choose_action(self, rl_action: int, current_reward: float = 0.0) -> int:
        """
        Choose final action based on RL policy + Reflexion rules.
        
        Args:
            rl_action: Action proposed by RL policy (0-6)
            current_reward: Reward from previous step
            
        Returns:
            final_action: Action to execute (0-6)
        """
        # Update history
        if len(self.action_history) > 0:
            self.reward_history.append(current_reward)
        self.action_history.append(rl_action)
        
        final_action = rl_action
        override_reason = None
        
        # Rule 1: 360° scan at episode start
        if self.enable_scan_rotation and not self.scan_complete and self.episode_step < 4:
            final_action = self.ACTIONS['RotateRight_90']
            override_reason = "Initial 360° scan"
            if self.episode_step == 3:
                self.scan_complete = True
        
        # Rule 2: Anti-repeat pickup (if last 3+ actions were pickup with no reward)
        elif self.discourage_repeat_pickup and len(self.action_history) >= 3:
            recent_actions = list(self.action_history)[-3:]
            recent_rewards = list(self.reward_history)[-3:] if len(self.reward_history) >= 3 else []
            
            # All pickup with zero reward
            if (all(a == self.ACTIONS['PickupObject'] for a in recent_actions) and
                all(r == 0.0 for r in recent_rewards) and
                rl_action == self.ACTIONS['PickupObject']):
                
                # Override with rotation to break loop
                final_action = self.ACTIONS['RotateRight_90']
                override_reason = "Breaking repeated failed pickup"
        
        # Rule 3: Encourage LookDown when stuck (same action 5+ times, no reward)
        elif self.encourage_lookdown and len(self.action_history) >= 5:
            recent_actions = list(self.action_history)[-5:]
            recent_rewards = list(self.reward_history)[-5:] if len(self.reward_history) >= 5 else []
            
            # Same action repeated, no reward, not already looking down
            if (len(set(recent_actions)) == 1 and
                all(r == 0.0 for r in recent_rewards) and
                rl_action != self.ACTIONS['LookDown_15']):
                
                final_action = self.ACTIONS['LookDown_15']
                override_reason = "Stuck - trying LookDown"
        
        # Rule 4: Try Toggle if struggling late in episode
        elif self.encourage_toggle_light and self.episode_step > 60:
            recent_rewards = list(self.reward_history)[-10:] if len(self.reward_history) >= 10 else list(self.reward_history)
            
            # No positive reward recently
            if all(r == 0.0 for r in recent_rewards) and rl_action != self.ACTIONS['ToggleObjectOn']:
                # Occasionally try toggle (every 10 steps)
                if self.episode_step % 10 == 0:
                    final_action = self.ACTIONS['ToggleObjectOn']
                    override_reason = "Late episode - trying Toggle"
        
        self.episode_step += 1
        
        # Log overrides
        if override_reason and final_action != rl_action:
            action_names = {v: k for k, v in self.ACTIONS.items()}
            print(f"    Step {self.episode_step}: Reflexion override: {action_names.get(rl_action, '?')} → {action_names.get(final_action, '?')} ({override_reason})")
        
        return final_action


# Test
if __name__ == '__main__':
    print("Testing ReflexionController3D...")
    
    controller = ReflexionController3D()
    controller.start_episode(task_type='pick_and_place_simple')
    
    print("\nSimulating episode:")
    for step in range(10):
        rl_action = 3  # Pickup
        reward = 0.0
        
        final_action = controller.choose_action(rl_action, reward)
        
        action_names = {v: k for k, v in controller.ACTIONS.items()}
        if final_action != rl_action:
            print(f"  Step {step}: Override {action_names.get(rl_action)} → {action_names.get(final_action)}")
    
    print("\n✓ Test complete")
