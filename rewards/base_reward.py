"""
Base reward class that defines the interface for all reward functions.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import carla


class BaseReward(ABC):
    """
    Abstract base class for reward functions.
    
    All reward functions must inherit from this class and implement
    the calculate_reward method.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the reward function.
        
        Args:
            config: Configuration dictionary for the reward function
        """
        self.config = config or {}
        self.reset()
    
    def reset(self):
        """
        Reset the reward function state for a new episode.
        Override this method to reset episode-specific variables.
        """
        pass
    
    @abstractmethod
    def calculate_reward(self, 
                        vehicle: carla.Vehicle,
                        episode_step: int,
                        collision_detected: bool,
                        **kwargs) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Calculate the reward for the current state.
        
        Args:
            vehicle: The ego vehicle
            episode_step: Current step in the episode
            collision_detected: Whether a collision was detected
            **kwargs: Additional state information
            
        Returns:
            Tuple of (reward, done, info)
            - reward: The calculated reward value
            - done: Whether the episode should end
            - info: Additional information dictionary
        """
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get additional information about the reward function state.
        
        Returns:
            Dictionary containing reward function state information
        """
        return {} 