import numpy as np

# Discrete action space: 9 steer × 3 throttle = 27 actions
STEER_VALUES = np.linspace(-1.0, 1.0, 9)  # 9 steering values from -1 to 1
THROTTLE_VALUES = np.linspace(0.0, 1.0, 3)  # 3 throttle values from 0 to 1
BRAKE_VALUE = 0.0  # Fixed brake value for now

def index_to_action(index):
    """
    Map discrete index to (steer, throttle, brake) vector.
    
    Args:
        index (int): Discrete action index [0, 26]
        
    Returns:
        tuple: (steer, throttle, brake) action vector
    """
    if not 0 <= index < 27:
        raise ValueError(f"Action index {index} out of range [0, 26]")
    
    # Calculate steer and throttle indices
    steer_idx = index % 9
    throttle_idx = index // 9
    
    # Get actual values
    steer = STEER_VALUES[steer_idx]
    throttle = THROTTLE_VALUES[throttle_idx]
    brake = BRAKE_VALUE
    
    return (steer, throttle, brake)

def get_action_space_size():
    """Return the size of the discrete action space."""
    return 27

def get_action_space():
    """Return all possible actions as a list."""
    return [index_to_action(i) for i in range(27)] 