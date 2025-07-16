"""
Simple reward function that implements basic speed-based rewards with collision penalties.
"""

import numpy as np
from typing import Dict, Any, Tuple
import carla

from .base_reward import BaseReward


class SimpleReward(BaseReward):
    """
    Simple reward function that rewards forward movement and penalizes collisions.
    
    This is the current reward function extracted from the CarlaEnvironment class.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the simple reward function.
        
        Args:
            config: Configuration dictionary with the following keys:
                - speed_reward_factor: Multiplier for speed reward (default: 0.1)
                - collision_penalty: Penalty for collisions (default: -100)
                - stuck_penalty: Penalty for being stuck (default: -10)
                - max_episode_steps: Maximum episode length (default: 1000)
                - stuck_threshold: Speed threshold for being considered stuck (default: 0.1)
                - stuck_min_steps: Minimum steps before checking if stuck (default: 100)
        """
        default_config = {
            'speed_reward_factor': 0.1,
            'collision_penalty': -100,
            'stuck_penalty': -10,
            'max_episode_steps': 1000,
            'stuck_threshold': 0.1,
            'stuck_min_steps': 100
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(default_config)
        self.reset()
    
    def reset(self):
        """Reset episode-specific variables."""
        self.total_distance = 0.0
        self.last_location = None
    
    def calculate_reward(self, 
                        vehicle: carla.Vehicle,
                        episode_step: int,
                        collision_detected: bool,
                        **kwargs) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Calculate reward based on speed, collisions, and episode progress.
        
        Args:
            vehicle: The ego vehicle
            episode_step: Current step in the episode
            collision_detected: Whether a collision was detected
            **kwargs: Additional state information
            
        Returns:
            Tuple of (reward, done, info)
        """
        # Get current vehicle state
        vehicle_location = vehicle.get_location()
        vehicle_velocity = vehicle.get_velocity()
        
        # Calculate distance traveled
        if self.last_location:
            distance = vehicle_location.distance(self.last_location)
            self.total_distance += distance
        self.last_location = vehicle_location
        
        # Calculate speed
        speed = np.sqrt(vehicle_velocity.x**2 + vehicle_velocity.y**2)
        
        # Initialize reward and done
        reward = speed * self.config['speed_reward_factor']
        done = False
        
        # Check for collisions
        if collision_detected:
            reward += self.config['collision_penalty']
            done = True
        
        # Episode timeout
        if episode_step >= self.config['max_episode_steps']:
            done = True
        
        # Check if vehicle is stuck (not moving)
        if (speed < self.config['stuck_threshold'] and 
            episode_step > self.config['stuck_min_steps']):
            reward += self.config['stuck_penalty']
            done = True
        
        # Create info dictionary
        info = {
            'distance': self.total_distance,
            'speed': speed,
            'episode_step': episode_step,
            'collision_detected': collision_detected
        }
        
        return reward, done, info
    
    def get_info(self) -> Dict[str, Any]:
        """Get additional information about the reward function state."""
        return {
            'total_distance': self.total_distance,
            'config': self.config
        } 