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
                - collision_penalty: Penalty for collisions (default: -300)
                - stuck_penalty: Penalty for being stuck (default: -10)
                - max_episode_steps: Maximum episode length (default: 1000)
                - stuck_threshold: Speed threshold for being considered stuck (default: 0.1)
                - stuck_min_steps: Minimum steps before checking if stuck (default: 100)
                - lane_keeping_reward: Reward multiplier for staying in lane (default: 2.0)
                - lane_center_bonus: Extra bonus for being very close to lane center (default: 1.0)
                - lane_deviation_penalty: Penalty multiplier for lane deviations (default: -5.0)
                - lane_change_penalty: Penalty for unnecessary lane changes (default: -15.0)
                - lane_stability_reward: Reward for staying in same lane (default: 1.0)
                - optimal_speed_min: Minimum optimal speed in m/s (default: 8.0)
                - optimal_speed_max: Maximum optimal speed in m/s (default: 12.0)
                - max_speed_reward: Maximum reward for speed (default: 2.0)
                - speed_limit_reward: Reward for staying within speed limits (default: 0.3)
                - smooth_driving_reward: Reward for smooth acceleration/steering (default: 0.2)
                - traffic_light_penalty: Penalty for running red lights (default: -50)
                - progress_reward: Reward for forward progress (default: 0.5)
                - offroad_penalty: Penalty for going off-road (default: -2.0)
                - yaw_rate_penalty: Penalty for high yaw rate (default: -0.05)
                - max_yaw_rate: Maximum yaw rate before penalty (default: 15.0)
                - lane_dense_bonus: Bonus for staying very close to lane center (default: 0.1)
                - lane_reward_scale_ramp_steps: Steps over which lane reward scales (default: 10000)
                - lane_reward_scale_max: Maximum scale factor for lane reward (default: 2.0)
                - global_reward_scale: Global reward scale factor (default: 0.1)
        """
        default_config = {
            'speed_reward_factor': 0.1,
            'collision_penalty_initial': -100,
            'collision_penalty_final': -200,
            'collision_penalty_ramp_steps': 10000,
            'max_episode_steps': 200,
            # 'stuck_threshold': 0.1, # Handled by new speed reward
            # 'stuck_min_steps': 50, # Handled by new speed reward
            'lane_keeping_reward': 4.0,
            'lane_center_bonus': 3.0,
            'lane_deviation_penalty': -1.5,
            'slight_over_penalty_scale': 0.5,
            'lane_change_penalty': -12.0,
            'lane_stability_reward': 6.0,
            'lane_stability_interval': 20,
            'lane_recovery_steps': 200,
            'lane_recovery_bonus': 20.0,
            'offroad_frames_threshold': 5,
            'allowed_lane_types': [carla.LaneType.Driving],
            'optimal_speed_min': 8.0,
            'optimal_speed_max': 12.0,
            'max_speed_reward': 4.0,
            'speed_limit_reward': 1.0,
            'smooth_driving_reward': 3.0,
            'traffic_light_penalty': -15,
            'progress_reward': 15.0,
            'offroad_penalty': -50.0,
            'yaw_rate_penalty': -0.05,
            'max_yaw_rate': 15.0,
            'survival_reward': 0.2,
            'base_forward_reward': 2.0,
            'wrong_way_penalty': -25.0,
            'opposite_lane_penalty': -20.0,
            'double_yellow_crossing_penalty': -40,
            'wrong_direction_penalty': -30,
            'success_distance_threshold': 100.0,
            'success_reward_bonus': 50.0,
            'lane_dense_bonus': 0.3,
            'straight_driving_bonus': 0.2,
            'lane_reward_scale_ramp_steps': 10000,
            'lane_reward_scale_max': 2.0,
            'global_reward_scale': 0.1,
            'steer_penalty_factor': 4.0,
            'low_progress_penalty': -0.2,
            'throttle_bonus': 0.1,
            'cumulative_steer_window': 20,
            'cumulative_steer_limit': 4.0,
            'cumulative_steer_penalty': -2.0,
            'yaw_threshold_deg': 5.0,
            'lateral_yaw_penalty': -0.1,
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(default_config)
        self.reset()
        self.global_step = 0
    
    def reset(self):
        """Reset episode-specific variables."""
        self.total_distance = 0.0
        self.last_location = None
        self.last_speed = 0.0
        self.last_steer = 0.0
        self.last_throttle = 0.0
        self.deviation_from_center = 0.0
        self.current_lane_id = None
        self.lane_changes = 0
        self.off_lane_frames = 0
        self.steer_window = []
        self.steps_in_current_lane = 0
        self.last_progress = 0.0
    
    def calculate_reward(self, 
                        vehicle: carla.Vehicle,
                        episode_step: int,
                        collision_detected: bool,
                        lane_invasion_detected: bool,
                        **kwargs) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Calculate advanced reward based on multiple driving objectives.
        
        Args:
            vehicle: The ego vehicle
            episode_step: Current step in the episode
            collision_detected: Whether a collision was detected
            lane_invasion_detected: Whether a lane invasion was detected
            **kwargs: Additional state information including:
                - current_action: Current action (steer, throttle, brake)
                - waypoint: Current waypoint for lane keeping
            
        Returns:
            Tuple of (reward, done, info)
        """
        self.global_step += 1
        # Get current vehicle state
        vehicle_location = vehicle.get_location()
        vehicle_velocity = vehicle.get_velocity()
        vehicle_transform = vehicle.get_transform()
        
        # Calculate distance traveled
        if self.last_location:
            distance = vehicle_location.distance(self.last_location)
            self.total_distance += distance
            self.last_progress = distance  # Track progress since last step
        self.last_location = vehicle_location
        
        # Calculate speed
        speed = np.sqrt(vehicle_velocity.x**2 + vehicle_velocity.y**2)
        
        # Initialize reward with base survival reward
        reward = self.config['survival_reward']  # Small positive reward for each step
        done = False
        
        # Base forward movement reward
        if self.last_progress > 0.1:  # Moving forward at all
            reward += self.config['base_forward_reward']
        
        # Corrected speed reward logic to penalize being stuck
        optimal_min = self.config['optimal_speed_min']
        optimal_max = self.config['optimal_speed_max']
        max_reward = self.config['max_speed_reward']
        
        speed_reward = 0.0
        if speed < optimal_min:
            # Penalize low speeds and reward moving up to optimal_min
            # Linear scaling from a penalty at speed 0 to max_reward at speed optimal_min
            speed_reward = -1.0 + (speed / optimal_min) * (max_reward + 1.0)
        elif speed <= optimal_max:
            # Max reward in the optimal range
            speed_reward = max_reward
        else: # speed > optimal_max
            # Penalty for exceeding speed limit, scales down to 0
            excess_speed = speed - optimal_max
            speed_reward = max_reward * max(0.0, 1.0 - excess_speed / optimal_max)
        
        reward += speed_reward
        
        # Enhanced lane keeping reward
        lane_reward = 0.0
        if 'world' in kwargs:
            world = kwargs['world']
            map_obj = world.get_map()
            
            # Get the closest waypoint to the vehicle
            waypoint = map_obj.get_waypoint(vehicle_location)
            
            if waypoint:
                # Calculate distance from lane center
                lane_center = waypoint.transform.location
                lateral_distance = self._calculate_lateral_distance(vehicle_location, vehicle_transform, lane_center)
                self.deviation_from_center = lateral_distance
                
                # Lane width is typically around 3.5-4.0 meters in CARLA
                lane_width = waypoint.lane_width
                max_deviation = lane_width / 2.0  # Half lane width
                
                # Calculate lane keeping reward based on distance from center
                lane_scale = 1.0 + min(self.global_step / self.config['lane_reward_scale_ramp_steps'], self.config['lane_reward_scale_max'] - 1.0)
                if lateral_distance <= 0.5:  # Very close to center (within 0.5m)
                    lane_reward = (self.config['lane_keeping_reward'] + self.config['lane_center_bonus']) * lane_scale
                elif lateral_distance <= 1.0:  # Good lane keeping (within 1m)
                    lane_reward = self.config['lane_keeping_reward'] * (1.0 - lateral_distance / 1.0) * lane_scale
                elif lateral_distance <= max_deviation:  # Still in lane but not ideal
                    normalized_deviation = lateral_distance / max_deviation
                    lane_reward = self.config['lane_keeping_reward'] * 0.2 * (1.0 - normalized_deviation) * lane_scale
                else:  # Outside lane or very close to edge
                    lane_reward = self.config['lane_deviation_penalty'] * (lateral_distance / max_deviation) * lane_scale * self.config['slight_over_penalty_scale']
                
                reward += lane_reward
                
                # Dense lane staying bonus
                if lateral_distance <= 0.3:
                    reward += self.config['lane_dense_bonus'] * lane_scale
                
                # Lane change detection and penalties
                current_lane_id = waypoint.lane_id
                
                if self.current_lane_id is None:
                    # First step, just record the lane
                    self.current_lane_id = current_lane_id
                    self.steps_in_current_lane = 1
                elif self.current_lane_id == current_lane_id:
                    # Staying in same lane - reward stability
                    self.steps_in_current_lane += 1
                    if self.steps_in_current_lane % self.config['lane_stability_interval'] == 0:
                        reward += self.config['lane_stability_reward'] * lane_scale
                    if self.steps_in_current_lane == self.config['lane_recovery_steps']:
                        reward += self.config['lane_recovery_bonus']
                else:
                    # Lane change detected
                    self.lane_changes += 1
                    self.current_lane_id = current_lane_id
                    self.steps_in_current_lane = 1
                    
                    # Penalty for lane changes (discourage unnecessary lane changes)
                    # More penalty for frequent lane changes
                    lane_change_penalty = self.config['lane_change_penalty'] * (1 + 1.0 * self.lane_changes)
                    if self.lane_changes >= 3:  # Multiple lane changes in episode
                        lane_change_penalty *= 1.5  # Increased penalty
                    
                    reward += lane_change_penalty
        
        # Yaw misalignment penalty
        if 'world' in kwargs and waypoint:
            lane_yaw = waypoint.transform.rotation.yaw
            vehicle_yaw = vehicle_transform.rotation.yaw
            yaw_err = abs((lane_yaw - vehicle_yaw + 180) % 360 - 180)
            if yaw_err > self.config['yaw_threshold_deg']:
                reward += self.config['lateral_yaw_penalty'] * (yaw_err - self.config['yaw_threshold_deg'])

        # Wrong-way driving detection - Fixed to prevent excessive penalties
        if waypoint:
            current_lane_id = waypoint.lane_id
            
            # Get the direction the vehicle should be traveling in this lane
            lane_direction = waypoint.transform.get_forward_vector()
            vehicle_direction = vehicle_transform.get_forward_vector()
            
            # Calculate the dot product to determine if driving in the correct direction
            direction_dot = (lane_direction.x * vehicle_direction.x + 
                           lane_direction.y * vehicle_direction.y)
            
            # Wrong direction penalty - only small continuous penalty
            if direction_dot < -0.2:  # More restrictive threshold
                reward += self.config['wrong_direction_penalty']  # Fixed amount, not escalating
            
            # Oncoming lane penalty - only when in wrong lanes, not escalating
            if current_lane_id < 0:
                reward += self.config['opposite_lane_penalty']  # Fixed penalty per step
                
            # Double yellow line crossing detection - ONE-TIME penalty only
            if (self.current_lane_id is not None and 
                self.current_lane_id > 0 and current_lane_id < 0 and
                not getattr(self, 'already_penalized_crossing', False)):
                reward += self.config['double_yellow_crossing_penalty']
                self.already_penalized_crossing = True  # Prevent repeated penalties
                
            # Reset crossing penalty when back in correct lanes
            if current_lane_id > 0:
                self.already_penalized_crossing = False
                
            # Simplified lane change penalty - only on actual changes
            if (self.current_lane_id is not None and 
                current_lane_id != self.current_lane_id):
                # Only penalize if it's not an adjacent legal lane change
                if not (self.current_lane_id > 0 and current_lane_id > 0 and 
                       abs(current_lane_id - self.current_lane_id) == 1):
                    reward += self.config['lane_change_penalty']
        
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
                
                # Small bonus for using throttle (encourage forward motion)
                # Cumulative steering window update for persistent steering penalty
                self.steer_window.append(abs(steer))
                if len(self.steer_window) > self.config['cumulative_steer_window']:
                    self.steer_window.pop(0)
                if sum(self.steer_window) > self.config['cumulative_steer_limit']:
                    reward += self.config['cumulative_steer_penalty']
                if throttle > 0.3:
                    reward += self.config['throttle_bonus']
                # Penalty for high steering magnitude (driving in tight circles)
                reward += -abs(steer) * self.config['steer_penalty_factor']
                
                self.last_steer = steer
                self.last_throttle = throttle
        
        # Yaw rate penalty for instability
        angular_velocity = vehicle.get_angular_velocity()
        yaw_rate = abs(angular_velocity.z)
        yaw_rate_penalty = 0.0
        if yaw_rate > self.config['max_yaw_rate']:
            yaw_rate_penalty = self.config['yaw_rate_penalty'] * (yaw_rate - self.config['max_yaw_rate'])
            reward += yaw_rate_penalty
        
        # Bonus for keeping steering/yaw small (driving straight)
        if abs(yaw_rate) < 0.05:
            reward += self.config['straight_driving_bonus']

        # Forward progress reward - more generous
        if self.last_progress > 1.0:  # Good forward progress
            reward += self.config['progress_reward'] * 1.5
        elif self.last_progress > 0.5:  # Decent progress
            reward += self.config['progress_reward']
        elif self.last_progress > 0.1:  # Any progress
            reward += self.config['progress_reward'] * 0.5
        else:  # Barely moving - smaller penalty
            reward += self.config['low_progress_penalty']
        
        # Off-road detection - more specific and punitive
        if waypoint and waypoint.lane_type not in self.config['allowed_lane_types']:
            self.off_lane_frames += 1
            if self.off_lane_frames >= self.config['offroad_frames_threshold']:
                reward += self.config['offroad_penalty']
                done = True  # terminate episode when persistently off-road
        else:
            self.off_lane_frames = 0
        if lane_invasion_detected:
            # Apply a smaller penalty for just crossing a line
            reward += self.config['lane_deviation_penalty']
        
        # Traffic light compliance (simplified)
        if 'traffic_lights' in kwargs:
            traffic_lights = kwargs['traffic_lights']
            for light in traffic_lights:
                if light.is_affecting(vehicle):
                    if light.state == carla.TrafficLightState.Red and speed > 0.1:
                        reward += self.config['traffic_light_penalty']
                        done = True  # Optional: end episode on red light violation
        
        # Check for collisions
        if collision_detected:
            frac = min(1.0, self.global_step / self.config['collision_penalty_ramp_steps'])
            coll_penalty = (1 - frac) * self.config['collision_penalty_initial'] + frac * self.config['collision_penalty_final']
            reward += coll_penalty
            done = True
        
        # Episode timeout or success
        if episode_step >= self.config['max_episode_steps']:
            done = True
            
        # Early success termination - if agent has driven well for a distance, end episode positively
        if (self.total_distance >= self.config['success_distance_threshold'] and 
            episode_step >= 50):  # Minimum steps to ensure some learning
            reward += self.config['success_reward_bonus']
            done = True
        
        # The 'stuck' check is now handled by the new speed reward logic
        
        # Update last speed
        self.last_speed = speed
        
        # Apply global reward scale
        reward *= self.config['global_reward_scale']

        # Create info dictionary
        info = {
            'distance': self.total_distance,
            'speed': speed,
            'episode_step': episode_step,
            'collision_detected': collision_detected,
            'deviation_from_center': self.deviation_from_center,
            'lane_changes': self.lane_changes,
            'steps_in_current_lane': self.steps_in_current_lane,
            'yaw_rate': yaw_rate,
            'reward_components': {
                'speed_reward': speed_reward,
                'lane_keeping': lane_reward if 'world' in kwargs else 0,
                'speed_limit': self.config['speed_limit_reward'] if speed <= 30.0 else -(speed - 30.0) * 0.1,
                'smooth_driving': 0,  # This gets updated in the smooth driving section above
                'yaw_rate_penalty': yaw_rate_penalty
            }
        }
        
        return reward, done, info
    
    def _calculate_lateral_distance(self, vehicle_location, vehicle_transform, lane_center) -> float:
        """
        Calculate the lateral (perpendicular) distance from vehicle to lane center.
        
        Args:
            vehicle_location: Current vehicle location
            vehicle_transform: Current vehicle transform
            lane_center: Center point of the lane
            
        Returns:
            Lateral distance in meters
        """
        # Vector from lane center to vehicle
        dx = vehicle_location.x - lane_center.x
        dy = vehicle_location.y - lane_center.y
        
        # Calculate Euclidean distance (simplified lateral distance)
        lateral_distance = np.sqrt(dx * dx + dy * dy)
        
        return lateral_distance
    
    def get_info(self) -> Dict[str, Any]:
        """Get additional information about the reward function state."""
        return {
            'total_distance': self.total_distance,
            'config': self.config,
            'last_speed': self.last_speed,
            'last_steer': self.last_steer,
            'last_throttle': self.last_throttle,
            'deviation_from_center': self.deviation_from_center,
            'lane_changes': self.lane_changes,
            'steps_in_current_lane': self.steps_in_current_lane,
            'current_lane_id': self.current_lane_id
        } 