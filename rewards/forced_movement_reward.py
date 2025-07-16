"""
Forced movement reward function that strongly incentivizes the agent to move forward.
This is designed to solve the "agent getting stuck" problem.
"""

import numpy as np
from typing import Dict, Any, Tuple
import carla

from .base_reward import BaseReward


class ForcedMovementReward(BaseReward):
    """
    Forced movement reward function that strongly incentivizes forward movement.
    
    This reward function is designed to solve the "agent not moving" problem by:
    1. Heavy penalties for being stationary
    2. Strong rewards for any forward movement
    3. Minimal rewards for road positioning (to avoid local optima)
    4. Clear termination conditions
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the forced movement reward function.
        
        Args:
            config: Configuration dictionary with the following keys:
                - movement_bonus: Bonus for forward movement (default: 5.0)
                - stationary_penalty: Penalty for being stationary (default: -10.0)
                - collision_penalty: Penalty for collisions (default: -100)
                - max_episode_steps: Maximum episode length (default: 1000)
                - stuck_threshold: Speed threshold for being stuck (default: 0.5)
                - stuck_penalty: Penalty for being stuck (default: -20.0)
                - stuck_min_steps: Minimum steps before checking if stuck (default: 50)
                - off_road_penalty: Penalty for going off-road (default: -15.0)
                - road_bonus: Small bonus for staying on road (default: 0.1)
        """
        default_config = {
            'movement_bonus': 10.0,  # Much higher bonus for movement
            'stationary_penalty': 0.0,  # No penalty for being stationary initially
            'collision_penalty': -100,
            'max_episode_steps': 1000,
            'stuck_threshold': 0.1,  # Lower threshold
            'stuck_penalty': -5.0,  # Gentler penalty
            'stuck_min_steps': 300,  # Much longer before considering stuck
            'off_road_penalty': -5.0,
            'road_bonus': 0.1,  # Lower bonus for staying on road
            'learning_phase_steps': 50,  # Shorter learning phase
            'movement_threshold': 0.5,  # Speed threshold to start penalizing
            'progressive_penalty': True  # Start penalizing stationary behavior after learning phase
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
    
    def _calculate_movement_reward(self, vehicle: carla.Vehicle) -> float:
        """
        Calculate reward based on forward movement.
        
        Args:
            vehicle: The ego vehicle
            
        Returns:
            Movement reward
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
            reward = forward_velocity * self.config['movement_bonus']
        else:
            # Small penalty for moving backward or sideways (not too harsh)
            reward = forward_velocity * self.config['movement_bonus'] * 0.5
        
        return reward
    
    def _calculate_speed_reward(self, vehicle: carla.Vehicle, episode_step: int) -> float:
        """
        Calculate reward based on speed.
        
        Args:
            vehicle: The ego vehicle
            episode_step: Current step in episode
            
        Returns:
            Speed-based reward
        """
        # Get vehicle velocity
        velocity = vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2)
        
        # During learning phase, be very encouraging
        if episode_step < self.config['learning_phase_steps']:
            # No penalty for being stationary during learning
            if speed < 0.1:
                return 0.0  # Neutral reward
            # Positive reward for any movement
            return speed * 2.0
        else:
            # After learning phase, start to penalize stationary behavior
            if speed < self.config['movement_threshold']:
                return -1.0  # Progressive penalty for being stationary
            # Strong reward for movement
            return speed * 5.0
    
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
    
    def _calculate_road_reward(self, vehicle: carla.Vehicle, world: carla.World) -> float:
        """
        Calculate minimal reward for staying on road.
        
        Args:
            vehicle: The ego vehicle
            world: The CARLA world
            
        Returns:
            Road reward
        """
        try:
            # Get the waypoint at the vehicle's location
            waypoint = world.get_map().get_waypoint(vehicle.get_location())
            
            if waypoint is None:
                # Vehicle is off-road
                return self.config['off_road_penalty']
            
            # Small bonus for being on road
            return self.config['road_bonus']
            
        except Exception as e:
            # If waypoint calculation fails, assume off-road
            return self.config['off_road_penalty']
    
    def calculate_reward(self, 
                        vehicle: carla.Vehicle,
                        episode_step: int,
                        collision_detected: bool,
                        **kwargs) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Calculate forced movement reward.
        
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
        movement_reward = self._calculate_movement_reward(vehicle)
        speed_reward = self._calculate_speed_reward(vehicle, episode_step)
        road_reward = self._calculate_road_reward(vehicle, kwargs.get('world'))
        
        # Total reward (heavily weighted toward movement)
        total_reward = movement_reward + speed_reward + road_reward
        
        # Debug print for first few steps
        if episode_step < 3:
            print(f"Forced Movement Step {episode_step}: movement={movement_reward:.3f}, speed={speed_reward:.3f}, road={road_reward:.3f}, total={total_reward:.3f}")
        
        # Check for collisions
        done = False
        if collision_detected:
            total_reward += self.config['collision_penalty']
            done = True
        
        # Check if stuck (only after learning phase)
        if episode_step >= self.config['learning_phase_steps'] and self._check_stuck_condition(vehicle):
            total_reward += self.config['stuck_penalty']
            done = True
        
        # Episode timeout
        if episode_step >= self.config['max_episode_steps']:
            done = True
        
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
                'movement_reward': movement_reward,
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