#!/usr/bin/env python3
"""
Rule Database Helper for Real 3D Reflexion

Loads and queries Reflexion rules generated from real 3D RL failures.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from collections import Counter


class RuleDatabase3D:
    """
    Helper for managing and querying Reflexion rules from 3D failures.
    """
    
    def __init__(self, db_path: str = 'data/rules/rule_database_real_3d.json'):
        """
        Initialize rule database.
        
        Args:
            db_path: Path to rule database JSON
        """
        self.db_path = Path(db_path)
        self.database = None
        
        if self.db_path.exists():
            self.load()
        else:
            print(f"⚠ Warning: Rule database not found: {db_path}")
            self.database = {'metadata': {}, 'episodes': {}}
    
    def load(self):
        """Load rule database from disk."""
        with open(self.db_path) as f:
            self.database = json.load(f)
        
        print(f"✓ Loaded rule database: {self.db_path}")
        print(f"  Episodes: {len(self.database.get('episodes', {}))}")
    
    def get_rules_for_task(self, task_type: str, top_k: int = 3) -> List[str]:
        """
        Get top-k most frequent rules for a specific task type.
        
        Args:
            task_type: ALFRED task type
            top_k: Number of rules to return
            
        Returns:
            rules: List of rule strings
        """
        if not self.database:
            return []
        
        # Collect all rules for this task type
        task_rules = []
        
        for episode_id, episode_data in self.database.get('episodes', {}).items():
            if episode_data.get('task_type') == task_type:
                task_rules.extend(episode_data.get('rules', []))
        
        if not task_rules:
            return []
        
        # Count frequency
        rule_counts = Counter(task_rules)
        
        # Return top-k
        top_rules = [rule for rule, count in rule_counts.most_common(top_k)]
        
        return top_rules
    
    def get_all_rules(self, top_k: int = 5) -> List[str]:
        """
        Get top-k most frequently occurring rules across all tasks.
        
        Args:
            top_k: Number of rules to return
            
        Returns:
            rules: List of rule strings
        """
        if not self.database:
            return []
        
        # Collect all rules from all episodes
        all_rules = []
        
        for episode_data in self.database.get('episodes', {}).values():
            all_rules.extend(episode_data.get('rules', []))
        
        # Count and return top-k
        rule_counts = Counter(all_rules)
        top_rules = [rule for rule, count in rule_counts.most_common(top_k)]
        
        return top_rules
    
    def get_episode_rules(self, episode_id: str) -> List[str]:
        """
        Get rules for a specific episode.
        
        Args:
            episode_id: Episode identifier
            
        Returns:
            rules: List of rule strings
        """
        if not self.database:
            return []
        
        episode = self.database.get('episodes', {}).get(episode_id)
        if not episode:
            return []
        
        return episode.get('rules', [])


# Test
if __name__ == '__main__':
    print("Testing RuleDatabase3D...")
    
    db = RuleDatabase3D()
    
    if db.database and db.database.get('episodes'):
        print(f"\n✓ Database loaded")
        print(f"  Episodes: {len(db.database['episodes'])}")
        
        # Test getting all rules
        all_rules = db.get_all_rules(top_k=3)
        print(f"\nTop 3 rules (all tasks):")
        for i, rule in enumerate(all_rules, 1):
            print(f"  {i}. {rule}")
    else:
        print("\n⚠ No episodes in database")
        print("  Run generate_rules_real_3d.py first")
