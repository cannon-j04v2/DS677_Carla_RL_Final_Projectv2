"""
Reward factory for creating and managing different reward functions.
"""

from typing import Dict, Any, Type
from .base_reward import BaseReward
from .simple_reward import SimpleReward
from .advanced_reward import AdvancedReward
from .custom_reward import CustomReward
from .movement_reward import MovementReward
from .forced_movement_reward import ForcedMovementReward


class RewardFactory:
    """
    Factory class for creating reward functions.
    
    This allows easy selection and configuration of different reward functions
    without modifying the main environment code.
    """
    
    # Registry of available reward functions
    _reward_registry: Dict[str, Type[BaseReward]] = {
        'simple': SimpleReward,
        'advanced': AdvancedReward,
        'custom': CustomReward,
        'movement': MovementReward,
        'forced_movement': ForcedMovementReward,
    }
    
    @classmethod
    def register_reward(cls, name: str, reward_class: Type[BaseReward]):
        """
        Register a new reward function.
        
        Args:
            name: Name to register the reward function under
            reward_class: The reward function class
        """
        cls._reward_registry[name] = reward_class
    
    @classmethod
    def create_reward(cls, reward_type: str, config: Dict[str, Any] = None) -> BaseReward:
        """
        Create a reward function instance.
        
        Args:
            reward_type: Type of reward function to create
            config: Configuration dictionary for the reward function
            
        Returns:
            Instance of the specified reward function
            
        Raises:
            ValueError: If the reward type is not registered
        """
        if reward_type not in cls._reward_registry:
            available = list(cls._reward_registry.keys())
            raise ValueError(f"Unknown reward type '{reward_type}'. Available types: {available}")
        
        reward_class = cls._reward_registry[reward_type]
        return reward_class(config)
    
    @classmethod
    def get_available_rewards(cls) -> list:
        """
        Get list of available reward function names.
        
        Returns:
            List of registered reward function names
        """
        return list(cls._reward_registry.keys())
    
    @classmethod
    def get_reward_info(cls, reward_type: str) -> Dict[str, Any]:
        """
        Get information about a specific reward function.
        
        Args:
            reward_type: Type of reward function
            
        Returns:
            Dictionary with reward function information
            
        Raises:
            ValueError: If the reward type is not registered
        """
        if reward_type not in cls._reward_registry:
            available = list(cls._reward_registry.keys())
            raise ValueError(f"Unknown reward type '{reward_type}'. Available types: {available}")
        
        reward_class = cls._reward_registry[reward_type]
        
        # Create a temporary instance to get default config
        temp_instance = reward_class()
        
        return {
            'name': reward_type,
            'class': reward_class.__name__,
            'description': reward_class.__doc__,
            'default_config': temp_instance.config
        } 