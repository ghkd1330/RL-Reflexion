#!/usr/bin/env python3
"""
Real ALFRED 3D Simulator Environment Wrapper - UPGRADED

This provides full ALFRED task support including:
- Object detection and ID resolution
- PickupObject, ToggleObjectOn, PutObject with automatic objectId
- Real goal condition checking
- pick_and_place_simple and look_at_obj_in_light tasks
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np

from ai2thor.controller import Controller

# Add ALFRED paths
alfred_path = os.path.abspath('env/alfred')
sys.path.insert(0, alfred_path)
sys.path.insert(0, os.path.join(alfred_path, 'gen'))

try:
    import gen.constants as constants
except:
    constants = None


class AlfredSimEnv3D:
    """
    Real ALFRED 3D Simulator Environment with full object interaction support.
    """
    
    # ALFRED action space (7 discrete actions)
    ACTIONS = [
        'LookDown_15',
        'LookUp_15', 
        'MoveAhead_25',
        'PickupObject',
        'RotateLeft_90',
        'RotateRight_90',
        'ToggleObjectOn'
    ]
    
    def __init__(self, 
                 data_path: str = 'data/json_feat_subset',
                 splits_path: str = 'data/splits/subset_oct21.json',
                 max_steps: int = 100,
                 frame_width: int = 300,
                 frame_height: int = 300):
        """Initialize ALFRED 3D environment."""
        self.data_path = Path(data_path)
        self.max_steps = max_steps
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Load splits
        with open(splits_path) as f:
            self.splits = json.load(f)
        
        self.controller = None
        self.current_episode = None
        self.current_traj_data = None
        self.step_count = 0
        self.picked_up_object = None  # Track what agent is holding
        
    def _load_episode_data(self, split: str, episode_idx: int):
        """Load ALFRED episode metadata and trajectory."""
        episodes = self.splits[split]
        episode = episodes[episode_idx]
        
        # Load trajectory
        task_path = episode['task']
        traj_file = self.data_path / split / task_path / 'traj_data.json'
        
        with open(traj_file) as f:
            traj_data = json.load(f)
        
        return episode, traj_data
    
    def _find_object_by_type(self, object_type: str, prefer_visible: bool = True) -> Optional[str]:
        """
        Find an object ID by type name.
        
        Args:
            object_type: Object type (e.g., "Apple", "Mug", "Lamp")
            prefer_visible: If True, prefer objects that are visible
            
        Returns:
            objectId string or None
        """
        event = self.controller.last_event
        objects = event.metadata['objects']
        
        # Filter by type
        matching_objects = [obj for obj in objects if object_type.lower() in obj['objectType'].lower()]
        
        if not matching_objects:
            return None
        
        # Prefer visible objects
        if prefer_visible:
            visible = [obj for obj in matching_objects if obj['visible']]
            if visible:
                matching_objects = visible
        
        # Return closest object (by distance to agent)
        agent_pos = event.metadata['agent']['position']
        
        def distance(obj):
            pos = obj['position']
            return ((pos['x'] - agent_pos['x'])**2 + 
                   (pos['z'] - agent_pos['z'])**2)**0.5
        
        closest = min(matching_objects, key=distance)
        return closest['objectId']
    
    def _find_receptacle(self, receptacle_type: str) -> Optional[str]:
        """Find a receptacle objectId by type (e.g., Cabinet, Microwave)."""
        return self._find_object_by_type(receptacle_type, prefer_visible=False)
    
    def reset(self, split: str = 'valid_seen', episode_idx: int = 0) -> Dict:
        """Reset environment to a new episode."""
        # Load episode
        self.current_episode, self.current_traj_data = self._load_episode_data(split, episode_idx)
        
        # Get scene info
        scene_num = self.current_traj_data['scene']['scene_num']
        scene_name = f"FloorPlan{scene_num}"
        
        # Initialize controller if needed
        if self.controller is None:
            self.controller = Controller()
        
        # Reset to scene (AI2-THOR 4.3 API)
        self.controller.reset(scene_name)
        
        # Initialize scene settings
        self.controller.step(
            action="Initialize",
            gridSize=0.25,
            renderDepthImage=False,
            renderInstanceSegmentation=False
        )
        
        # Restore initial state
        init_action = self.current_traj_data['scene']['init_action']
        
        position = {
            'x': init_action['x'],
            'y': init_action['y'],
            'z': init_action['z']
        }
        rotation = {'y': init_action['rotation']}
        horizon = init_action['horizon']
        
        # Teleport agent
        self.controller.step(
            action="TeleportFull",
            x=position['x'],
            y=position['y'],
            z=position['z'],
            rotation={'x': 0, 'y': rotation['y'], 'z': 0},
            horizon=horizon,
            standing=True
        )
        
        # Reset tracking
        self.step_count = 0
        self.picked_up_object = None
        
        # Get initial observation
        event = self.controller.last_event
        frame = event.frame
        instruction = self.current_traj_data['turk_annotations']['anns'][0]['task_desc']
        
        obs = {
            'frame': frame,
            'instruction': instruction
        }
        
        return obs
    
    def step(self, action_idx: int) -> Tuple[Dict, float, bool, Dict]:
        """Execute action in simulator with object ID resolution."""
        self.step_count += 1
        
        # Validate action index
        if action_idx < 0 or action_idx >= len(self.ACTIONS):
            event = self.controller.last_event
            obs = {
                'frame': event.frame,
                'instruction': self.current_traj_data['turk_annotations']['anns'][0]['task_desc']
            }
            return obs, 0.0, False, {'goal_satisfied': False, 'action_success': False}
        
        action_name = self.ACTIONS[action_idx]
        
        # Get goal params
        goal_params = self.current_traj_data['pddl_params']
        
        # Resolve action with objectIds
        if action_name == 'PickupObject':
            # Find target object
            object_type = goal_params.get('object_target', '')
            if object_type:
                object_id = self._find_object_by_type(object_type)
                if object_id:
                    event = self.controller.step(action='PickupObject', objectId=object_id)
                    if event.metadata['lastActionSuccess']:
                        self.picked_up_object = object_type
                else:
                    # Object not found - action fails
                    event = self.controller.last_event
            else:
                event = self.controller.last_event
                
        elif action_name == 'ToggleObjectOn':
            # Find toggle target (usually a lamp)
            toggle_type = goal_params.get('toggle_target', '')
            if toggle_type:
                object_id = self._find_object_by_type(toggle_type)
                if object_id:
                    event = self.controller.step(action='ToggleObjectOn', objectId=object_id)
                else:
                    event = self.controller.last_event
            else:
                event = self.controller.last_event
                
        elif action_name == 'PutObject':
            # Put object in receptacle
            if self.picked_up_object:
                receptacle_type = goal_params.get('parent_target', '')
                if receptacle_type:
                    receptacle_id = self._find_receptacle(receptacle_type)
                    if receptacle_id:
                        event = self.controller.step(action='PutObject', 
                                                    objectId=receptacle_id,
                                                    forceAction=True)
                        if event.metadata['lastActionSuccess']:
                            self.picked_up_object = None
                    else:
                        event = self.controller.last_event
                else:
                    event = self.controller.last_event
            else:
                event = self.controller.last_event
                
        else:
            # Navigation actions
            action_map = {
                'LookDown_15': 'LookDown',
                'LookUp_15': 'LookUp',
                'MoveAhead_25': 'MoveAhead',
                'RotateLeft_90': 'RotateLeft',
                'RotateRight_90': 'RotateRight'
            }
            thor_action = action_map.get(action_name, action_name)
            event = self.controller.step(action=thor_action)
        
        # Get new observation
        frame = event.frame
        instruction = self.current_traj_data['turk_annotations']['anns'][0]['task_desc']
        
        obs = {
            'frame': frame,
            'instruction': instruction
        }
        
        # Check goal
        goal_satisfied = self._check_goal_conditions()
        
        # Reward
        reward = 1.0 if goal_satisfied else 0.0
        
        # Done
        done = goal_satisfied or (self.step_count >= self.max_steps)
        
        info = {
            'goal_satisfied': goal_satisfied,
            'action_success': event.metadata['lastActionSuccess'],
            'step': self.step_count
        }
        
        return obs, reward, done, info
    
    def _check_goal_conditions(self) -> bool:
        """
        Check if goal conditions are satisfied based on task type.
        """
        task_type = self.current_traj_data.get('task_type', '')
        goal_params = self.current_traj_data['pddl_params']
        
        if 'pick_and_place' in task_type:
            return self._check_pick_and_place_goal(goal_params)
        elif 'look_at_obj_in_light' in task_type:
            return self._check_look_at_obj_goal(goal_params)
        else:
            # Unknown task type
            return False
    
    def _check_pick_and_place_goal(self, goal_params: Dict) -> bool:
        """
        Check pick_and_place_simple goal:
        - object_target must be in parent_target receptacle
        """
        object_type = goal_params.get('object_target', '')
        receptacle_type = goal_params.get('parent_target', '')
        
        if not object_type or not receptacle_type:
            return False
        
        # Get all objects
        objects = self.controller.last_event.metadata['objects']
        
        # Find the target object
        target_objs = [obj for obj in objects if object_type.lower() in obj['objectType'].lower()]
        if not target_objs:
            return False
        
        # Find the receptacle
        receptacles = [obj for obj in objects if receptacle_type.lower() in obj['objectType'].lower()]
        if not receptacles:
            return False
        
        receptacle_id = receptacles[0]['objectId']
        
        # Check if any target object is in the receptacle
        for obj in target_objs:
            if obj.get('parentReceptacles') and receptacle_id in obj['parentReceptacles']:
                return True
        
        return False
    
    def _check_look_at_obj_goal(self, goal_params: Dict) -> bool:
        """
        Check look_at_obj_in_light goal:
        - toggle_target (lamp) must be on
        - object_target must be visible
        """
        object_type = goal_params.get('object_target', '')
        toggle_type = goal_params.get('toggle_target', '')
        
        if not object_type or not toggle_type:
            return False
        
        objects = self.controller.last_event.metadata['objects']
        
        # Check if toggle object is on
        toggle_objs = [obj for obj in objects if toggle_type.lower() in obj['objectType'].lower()]
        if not toggle_objs:
            return False
        
        lamp_on = any(obj.get('isToggled', False) for obj in toggle_objs)
        if not lamp_on:
            return False
        
        # Check if target object is visible
        target_objs = [obj for obj in objects if object_type.lower() in obj['objectType'].lower()]
        if not target_objs:
            return False
        
        object_visible = any(obj.get('visible', False) for obj in target_objs)
        
        return object_visible
    
    def get_current_traj_data(self):
        """Get current episode's trajectory data for BC dataset recording."""
        return self.current_traj_data
    
    def close(self):
        """Close the simulator."""
        if self.controller is not None:
            try:
                self.controller.stop()
            except:
                pass  # Ignore errors during cleanup
            self.controller = None
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except:
            pass  # Ignore errors during cleanup


# Test
if __name__ == '__main__':
    print("Testing upgraded AlfredSimEnv3D...")
    
    env = AlfredSimEnv3D()
    
    # Reset
    obs = env.reset(split='valid_seen', episode_idx=0)
    print(f"✓ Reset complete")
    print(f"  Frame shape: {obs['frame'].shape}")
    print(f"  Instruction: {obs['instruction'][:60]}...")
    
    # Take a few actions
    for i in range(10):
        action = np.random.randint(0, 7)
        obs, reward, done, info = env.step(action)
        print(f"  Step {i+1}: action={action}, success={info['action_success']}, done={done}")
        
        if done:
            print(f"  Goal satisfied: {info['goal_satisfied']}")
            break
    
    env.close()
    print("✓ Test complete")
