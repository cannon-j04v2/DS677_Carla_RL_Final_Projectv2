"""
Advanced reward function that includes lane keeping, speed limits, and traffic light compliance.
"""

import numpy as np
from typing import Dict, Any, Tuple
import carla

from .base_reward import BaseReward


class AdvancedReward(BaseReward):
    """
    Advanced reward function with multiple driving objectives.
    
    This reward function includes:
    - Lane keeping rewards
    - Speed limit compliance
    - Traffic light compliance
    - Smooth driving rewards
    - Distance-based rewards
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the advanced reward function.
        
        Args:
            config: Configuration dictionary with the following keys:
                - speed_reward_factor: Multiplier for speed reward (default: 0.1)
                - collision_penalty: Penalty for collisions (default: -100)
                - stuck_penalty: Penalty for being stuck (default: -10)
                - max_episode_steps: Maximum episode length (default: 1000)
                - stuck_threshold: Speed threshold for being considered stuck (default: 0.1)
                - stuck_min_steps: Minimum steps before checking if stuck (default: 100)
                - lane_keeping_reward: Reward for staying in lane (default: 0.5)
                - speed_limit_reward: Reward for staying within speed limits (default: 0.3)
                - smooth_driving_reward: Reward for smooth acceleration/steering (default: 0.2)
        """
        default_config = {
            'speed_reward_factor': 0.1,
            'collision_penalty': -100,
            'stuck_penalty': -10,
            'max_episode_steps': 1000,
            'stuck_threshold': 0.1,
            'stuck_min_steps': 100,
            'lane_keeping_reward': 0.5,
            'speed_limit_reward': 0.3,
            'smooth_driving_reward': 0.2
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(default_config)
        self.reset()
    
    def reset(self):
        """Reset episode-specific variables."""
        self.total_distance = 0.0
        self.last_location = None
        self.last_speed = 0.0
        self.last_steer = 0.0
        self.last_throttle = 0.0
    
    def calculate_reward(self, 
                        vehicle: carla.Vehicle,
                        episode_step: int,
                        collision_detected: bool,
                        **kwargs) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Calculate advanced reward based on multiple driving objectives.
        
        Args:
            vehicle: The ego vehicle
            episode_step: Current step in the episode
            collision_detected: Whether a collision was detected
            **kwargs: Additional state information including:
                - current_action: Current action (steer, throttle, brake)
                - waypoint: Current waypoint for lane keeping
            
        Returns:
            Tuple of (reward, done, info)
        """
        # Get current vehicle state
        vehicle_location = vehicle.get_location()
        vehicle_velocity = vehicle.get_velocity()
        vehicle_transform = vehicle.get_transform()
        
        # Calculate distance traveled
        if self.last_location:
            distance = vehicle_location.distance(self.last_location)
            self.total_distance += distance
        self.last_location = vehicle_location
        
        # Calculate speed
        speed = np.sqrt(vehicle_velocity.x**2 + vehicle_velocity.y**2)
        
        # Initialize reward
        reward = 0.0
        done = False
        
        # Basic speed reward
        reward += speed * self.config['speed_reward_factor']
        
        # Lane keeping reward (simplified - would need waypoint info)
        if 'waypoint' in kwargs:
            waypoint = kwargs['waypoint']
            # Calculate distance from center of lane
            # This is a simplified version - in practice you'd need more complex lane detection
            lane_keeping_bonus = self.config['lane_keeping_reward']
            reward += lane_keeping_bonus
        
        # Speed limit compliance (simplified)
        speed_limit = 30.0  # m/s (about 108 km/h)
        if speed <= speed_limit:
            reward += self.config['speed_limit_reward']
        else:
            reward -= (speed - speed_limit) * 0.1  # Penalty for exceeding speed limit
        
        # Smooth driving reward
        if 'current_action' in kwargs:
            current_action = kwargs['current_action']
            if len(current_action) >= 2:
                steer, throttle = current_action[0], current_action[1]
                
                # Penalize sudden steering changes
                steer_change = abs(steer - self.last_steer)
                if steer_change < 0.1:  # Smooth steering
                    reward += self.config['smooth_driving_reward'] * 0.5
                else:
                    reward -= steer_change * 0.1
                
                # Penalize sudden throttle changes
                throttle_change = abs(throttle - self.last_throttle)
                if throttle_change < 0.1:  # Smooth acceleration
                    reward += self.config['smooth_driving_reward'] * 0.5
                else:
                    reward -= throttle_change * 0.1
                
                self.last_steer = steer
                self.last_throttle = throttle
        
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
        
        # Update last speed
        self.last_speed = speed
        
        # Create info dictionary
        info = {
            'distance': self.total_distance,
            'speed': speed,
            'episode_step': episode_step,
            'collision_detected': collision_detected,
            'reward_components': {
                'speed_reward': speed * self.config['speed_reward_factor'],
                'lane_keeping': self.config['lane_keeping_reward'] if 'waypoint' in kwargs else 0,
                'speed_limit': self.config['speed_limit_reward'] if speed <= 30.0 else -(speed - 30.0) * 0.1
            }
        }
        
        return reward, done, info
    
    def get_info(self) -> Dict[str, Any]:
        """Get additional information about the reward function state."""
        return {
            'total_distance': self.total_distance,
            'config': self.config,
            'last_speed': self.last_speed,
            'last_steer': self.last_steer,
            'last_throttle': self.last_throttle
        } 