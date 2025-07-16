"""
Movement-focused reward function that strongly incentivizes forward movement.
"""

import numpy as np
from typing import Dict, Any, Tuple
import carla

from .base_reward import BaseReward


class MovementReward(BaseReward):
    """
    Movement-focused reward function that strongly incentivizes forward movement.
    
    This reward function is designed to solve the "agent not moving" problem by:
    1. Providing strong positive rewards for forward movement
    2. Heavy penalties for being stationary
    3. Progressive rewards for maintaining speed
    4. Clear termination conditions
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the movement reward function.
        
        Args:
            config: Configuration dictionary with the following keys:
                - forward_reward_factor: Multiplier for forward movement (default: 2.0)
                - speed_reward_factor: Multiplier for speed reward (default: 1.0)
                - stationary_penalty: Penalty for being stationary (default: -2.0)
                - collision_penalty: Penalty for collisions (default: -100)
                - max_episode_steps: Maximum episode length (default: 1000)
                - stuck_threshold: Speed threshold for being stuck (default: 0.1)
                - stuck_penalty: Penalty for being stuck (default: -5.0)
                - stuck_min_steps: Minimum steps before checking if stuck (default: 200)
                - target_speed: Target speed for optimal reward (default: 10.0)
                - road_following_reward: Reward for staying on road (default: 1.0)
                - lane_center_reward: Reward for staying in lane center (default: 0.5)
                - off_road_penalty: Penalty for going off-road (default: -10.0)
        """
        default_config = {
            'forward_reward_factor': 2.0,
            'speed_reward_factor': 1.0,
            'stationary_penalty': -2.0,  # Reduced penalty
            'collision_penalty': -100,
            'max_episode_steps': 1000,
            'stuck_threshold': 0.1,  # Lower threshold
            'stuck_penalty': -5.0,  # Reduced penalty
            'stuck_min_steps': 200,  # Much longer before considering stuck
            'target_speed': 10.0,
            'road_following_reward': 1.0,
            'lane_center_reward': 0.5,
            'off_road_penalty': -10.0
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(default_config)
        self.reset()
    
    def reset(self):
        """Reset episode-specific variables."""
        self.total_distance = 0.0
        self.last_location = None
        self.stuck_counter = 0
        self.last_speed = 0.0
    
    def _calculate_forward_movement_reward(self, vehicle: carla.Vehicle) -> float:
        """
        Calculate reward based on forward movement.
        
        Args:
            vehicle: The ego vehicle
            
        Returns:
            Forward movement reward
        """
        # Get vehicle transform and velocity
        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()
        
        # Calculate forward direction (vehicle's forward vector)
        forward_vector = transform.get_forward_vector()
        
        # Calculate velocity in forward direction
        forward_velocity = (
            velocity.x * forward_vector.x + 
            velocity.y * forward_vector.y
        )
        
        # Strong positive reward for forward movement
        if forward_velocity > 0:
            reward = forward_velocity * self.config['forward_reward_factor']
        else:
            # Penalty for moving backward or sideways
            reward = forward_velocity * self.config['forward_reward_factor'] * 0.5
        
        return reward
    
    def _calculate_speed_reward(self, vehicle: carla.Vehicle) -> float:
        """
        Calculate reward based on speed.
        
        Args:
            vehicle: The ego vehicle
            
        Returns:
            Speed-based reward
        """
        # Get vehicle velocity
        velocity = vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2)
        
        # Progressive reward for speed
        if speed < 0.1:
            # Heavy penalty for being stationary
            reward = self.config['stationary_penalty']
        elif speed < self.config['target_speed']:
            # Linear reward up to target speed
            reward = speed * self.config['speed_reward_factor']
        else:
            # Diminishing reward above target speed
            reward = self.config['target_speed'] * self.config['speed_reward_factor']
            reward += (speed - self.config['target_speed']) * self.config['speed_reward_factor'] * 0.1
        
        return reward
    
    def _check_stuck_condition(self, vehicle: carla.Vehicle) -> bool:
        """
        Check if vehicle is stuck.
        
        Args:
            vehicle: The ego vehicle
            
        Returns:
            True if vehicle is stuck
        """
        velocity = vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2)
        
        if speed < self.config['stuck_threshold']:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        
        return self.stuck_counter > self.config['stuck_min_steps']
    
    def _calculate_road_following_reward(self, vehicle: carla.Vehicle, world: carla.World) -> float:
        """
        Calculate reward for staying on the road and in lane center.
        
        Args:
            vehicle: The ego vehicle
            world: The CARLA world
            
        Returns:
            Road following reward
        """
        try:
            # Get vehicle location and transform
            vehicle_location = vehicle.get_location()
            vehicle_transform = vehicle.get_transform()
            
            # Get the waypoint closest to the vehicle
            waypoint = world.get_map().get_waypoint(vehicle_location)
            
            if waypoint is None:
                # Vehicle is off-road
                return self.config['off_road_penalty']
            
            # Calculate distance from lane center
            lane_center = waypoint.transform.location
            distance_from_center = vehicle_location.distance(lane_center)
            
            # Get lane width (approximate)
            lane_width = 4.0  # Typical lane width in meters
            
            # Reward for staying in lane center
            if distance_from_center < lane_width / 4:  # Very close to center
                lane_reward = self.config['lane_center_reward']
            elif distance_from_center < lane_width / 2:  # In lane
                lane_reward = self.config['lane_center_reward'] * 0.5
            elif distance_from_center < lane_width:  # Near edge of lane
                lane_reward = 0.0
            else:  # Off-road
                lane_reward = self.config['off_road_penalty']
            
            # Base reward for being on road
            road_reward = self.config['road_following_reward']
            
            return road_reward + lane_reward
            
        except Exception as e:
            # If waypoint calculation fails, assume off-road
            return self.config['off_road_penalty']
    
    def calculate_reward(self, 
                        vehicle: carla.Vehicle,
                        episode_step: int,
                        collision_detected: bool,
                        **kwargs) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Calculate movement-focused reward.
        
        Args:
            vehicle: The ego vehicle
            episode_step: Current step in the episode
            collision_detected: Whether a collision was detected
            **kwargs: Additional state information
            
        Returns:
            Tuple of (reward, done, info)
        """
        # Calculate distance traveled
        vehicle_location = vehicle.get_location()
        if self.last_location:
            distance = vehicle_location.distance(self.last_location)
            self.total_distance += distance
        self.last_location = vehicle_location
        
        # Calculate reward components
        forward_reward = self._calculate_forward_movement_reward(vehicle)
        speed_reward = self._calculate_speed_reward(vehicle)
        road_reward = self._calculate_road_following_reward(vehicle, kwargs.get('world'))
        
        # Total reward
        total_reward = forward_reward + speed_reward + road_reward
        
        # Debug print for first few steps (only in first episode)
        if episode_step < 3:
            print(f"Reward Step {episode_step}: forward={forward_reward:.3f}, speed={speed_reward:.3f}, road={road_reward:.3f}, total={total_reward:.3f}")
        
        # Check for collisions
        done = False
        if collision_detected:
            total_reward += self.config['collision_penalty']
            done = True
            print(f"Episode ending due to collision at step {episode_step}")
        
        # Check if stuck
        if self._check_stuck_condition(vehicle):
            total_reward += self.config['stuck_penalty']
            done = True
            print(f"Episode ending due to being stuck at step {episode_step} (stuck_counter: {self.stuck_counter})")
        
        # Episode timeout
        if episode_step >= self.config['max_episode_steps']:
            done = True
            print(f"Episode ending due to timeout at step {episode_step}")
        
        # Get additional info
        velocity = vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2)
        
        info = {
            'distance': self.total_distance,
            'speed': speed,
            'episode_step': episode_step,
            'collision_detected': collision_detected,
            'stuck_counter': self.stuck_counter,
            'reward_components': {
                'forward_reward': forward_reward,
                'speed_reward': speed_reward,
                'road_reward': road_reward,
                'total_reward': total_reward
            }
        }
        
        return total_reward, done, info
    
    def get_info(self) -> Dict[str, Any]:
        """Get additional information about the reward function state."""
        return {
            'total_distance': self.total_distance,
            'stuck_counter': self.stuck_counter,
            'config': self.config
        } 