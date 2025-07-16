# Rewards System

This directory contains a modular reward system for CARLA autonomous driving reinforcement learning. The system allows you to easily swap and extend reward functions without modifying the main environment code.

## Overview

The rewards system consists of:

- **BaseReward**: Abstract base class that defines the interface for all reward functions
- **SimpleReward**: Basic reward function that rewards forward movement and penalizes collisions
- **AdvancedReward**: More complex reward function with lane keeping, speed limits, and smooth driving
- **RewardFactory**: Factory class for creating and managing reward functions

## Usage

### Command Line Usage

You can specify which reward function to use when running the training script:

```bash
# Use simple reward (default)
python continuous_driver.py --algo ppo --reward-type simple

# Use advanced reward
python continuous_driver.py --algo dqn --reward-type advanced

# Use custom reward with CARLA sensor data
python continuous_driver.py --algo ppo --reward-type custom

# List available reward functions
python continuous_driver.py --list-rewards
```

### Programmatic Usage

```python
from rewards import RewardFactory

# Create a simple reward function
simple_reward = RewardFactory.create_reward('simple')

# Create an advanced reward function with custom config
config = {
    'speed_reward_factor': 0.2,
    'collision_penalty': -200,
    'lane_keeping_reward': 1.0
}
advanced_reward = RewardFactory.create_reward('advanced', config)

# Create a custom reward function with CARLA sensor data
custom_config = {
    'angle_weight': 2.0,
    'distance_weight': 1.5,
    'target_speed': 20.0
}
custom_reward = RewardFactory.create_reward('custom', custom_config)

# List available reward functions
available = RewardFactory.get_available_rewards()
print(f"Available rewards: {available}")

# Get information about a reward function
info = RewardFactory.get_reward_info('advanced')
print(f"Advanced reward description: {info['description']}")
```

## Creating Custom Reward Functions

To create a new reward function:

1. Create a new file in the `rewards/` directory (e.g., `my_reward.py`)
2. Inherit from `BaseReward` and implement the required methods
3. Register your reward function in the factory

### Example Custom Reward Function

```python
from rewards.base_reward import BaseReward
from typing import Dict, Any, Tuple
import carla

class MyCustomReward(BaseReward):
    """My custom reward function."""
    
    def __init__(self, config: Dict[str, Any] = None):
        default_config = {
            'my_parameter': 1.0
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)
        self.reset()
    
    def reset(self):
        """Reset episode-specific variables."""
        self.my_variable = 0.0
    
    def calculate_reward(self, 
                        vehicle: carla.Vehicle,
                        episode_step: int,
                        collision_detected: bool,
                        **kwargs) -> Tuple[float, bool, Dict[str, Any]]:
        """Calculate reward based on my custom logic."""
        # Your reward calculation logic here
        reward = 0.0
        done = False
        
        # Example: reward based on vehicle speed
        velocity = vehicle.get_velocity()
        speed = (velocity.x**2 + velocity.y**2)**0.5
        reward = speed * self.config['my_parameter']
        
        # Check for collisions
        if collision_detected:
            done = True
        
        info = {
            'speed': speed,
            'my_variable': self.my_variable
        }
        
        return reward, done, info
```

### Registering Your Custom Reward Function

```python
from rewards.reward_factory import RewardFactory
from rewards.my_reward import MyCustomReward

# Register your reward function
RewardFactory.register_reward('my_custom', MyCustomReward)

# Now you can use it
reward = RewardFactory.create_reward('my_custom')
```

## Available Reward Functions

### SimpleReward

The default reward function that implements basic speed-based rewards with collision penalties.

**Features:**
- Rewards forward movement (speed-based)
- Penalizes collisions
- Handles episode timeouts
- Detects when vehicle is stuck

**Configuration:**
- `speed_reward_factor`: Multiplier for speed reward (default: 0.1)
- `collision_penalty`: Penalty for collisions (default: -100)
- `stuck_penalty`: Penalty for being stuck (default: -10)
- `max_episode_steps`: Maximum episode length (default: 1000)
- `stuck_threshold`: Speed threshold for being considered stuck (default: 0.1)
- `stuck_min_steps`: Minimum steps before checking if stuck (default: 100)

### AdvancedReward

A more complex reward function with multiple driving objectives.

**Features:**
- All features from SimpleReward
- Lane keeping rewards
- Speed limit compliance
- Smooth driving rewards (penalizes sudden steering/throttle changes)

**Configuration:**
- All SimpleReward configuration options
- `lane_keeping_reward`: Reward for staying in lane (default: 0.5)
- `speed_limit_reward`: Reward for staying within speed limits (default: 0.3)
- `smooth_driving_reward`: Reward for smooth acceleration/steering (default: 0.2)

### CustomReward

A sophisticated reward function that uses CARLA sensor data and waypoints to calculate rewards based on multiple driving aspects.

**Features:**
- **Angle alignment**: Rewards proper alignment with road orientation
- **Lane centering**: Rewards staying close to the center of the road
- **Speed optimization**: Rewards maintaining target speed
- **Collision penalties**: Heavy penalties for collisions
- **Detailed feedback**: Provides breakdown of reward components

**Configuration:**
- `angle_weight`: Weight for angle alignment reward (default: 1.0)
- `distance_weight`: Weight for distance from center reward (default: 1.0)
- `speed_weight`: Weight for speed reward (default: 0.1)
- `collision_penalty`: Penalty for collisions (default: -100)
- `max_episode_steps`: Maximum episode length (default: 1000)
- `max_angle_penalty`: Maximum penalty for angle deviation (default: -5.0)
- `max_distance_penalty`: Maximum penalty for distance from center (default: -5.0)
- `target_speed`: Target speed for optimal reward (default: 15.0 m/s)
- `speed_tolerance`: Tolerance around target speed (default: 5.0 m/s)

**Reward Components:**
1. **Angle Reward**: Uses cosine function to reward proper alignment with road
2. **Distance Reward**: Uses exponential decay to reward staying near road center
3. **Speed Reward**: Rewards maintaining target speed within tolerance
4. **Collision Penalty**: Heavy penalty for any collision

## Testing

Run the test script to verify the rewards system works correctly:

```bash
python test_rewards.py
```

## Integration with Environment

The rewards system is integrated into the `CarlaEnvironment` class. The environment now:

1. Accepts a `reward_type` parameter in its constructor
2. Creates the specified reward function using the factory
3. Calls the reward function's `calculate_reward` method instead of the old `calculate_reward_and_done` method
4. Resets the reward function at the start of each episode

This maintains backward compatibility while providing the flexibility to easily swap reward functions.

## Project Structure

The project has been modularized for better organization:

```
├── continuous_driver.py    # Main training/testing script
├── get_args.py            # Argument parser and validation
├── parameters.py          # Hyperparameters and constants
├── rewards/               # Modular reward system
│   ├── __init__.py
│   ├── base_reward.py
│   ├── simple_reward.py
│   ├── advanced_reward.py
│   ├── custom_reward.py
│   ├── reward_factory.py
│   └── README.md
├── networks/              # RL algorithms
├── test_rewards.py        # Test rewards system
├── test_custom_reward.py  # Test custom reward
└── test_args.py          # Test argument parser
```

### Argument Parser

The argument parser has been moved to `get_args.py` for better modularity:

- **`get_args()`**: Main function to get and validate arguments
- **`parse_args()`**: Parse command line arguments
- **`validate_args()`**: Validate argument values
- **`boolean_string()`**: Helper function for boolean arguments

All existing command line arguments are preserved with the same functionality. 