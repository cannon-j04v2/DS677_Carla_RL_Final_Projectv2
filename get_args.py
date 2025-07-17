"""
Argument parser for CARLA autonomous driving reinforcement learning.

This module contains the argument parser and related functions for configuring
the training and testing of RL agents in CARLA.
"""

import argparse
from distutils.util import strtobool
from parameters import *


def boolean_string(s):
    """
    Convert string to boolean for argument parsing.
    
    Args:
        s: String to convert ('True' or 'False')
        
    Returns:
        Boolean value
        
    Raises:
        ValueError: If string is not 'True' or 'False'
    """
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def parse_args():
    """
    Parse command line arguments for the CARLA RL training/testing.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='CARLA Autonomous Driving Reinforcement Learning',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Algorithm and training parameters
    parser.add_argument('--algo', '--exp-name', type=str, default='ppo', 
                       choices=['ppo', 'dqn'], 
                       help='RL algorithm to use (ppo or dqn)')
    parser.add_argument('--env-name', type=str, default='carla', 
                       help='name of the simulation environment')
    parser.add_argument('--learning-rate', type=float, default=PPO_LEARNING_RATE, 
                       help='learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=SEED, 
                       help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=TOTAL_TIMESTEPS, 
                       help='total timesteps of the experiment')
    parser.add_argument('--action-std-init', type=float, default=ACTION_STD_INIT, 
                       help='initial exploration noise')
    parser.add_argument('--test-timesteps', type=int, default=TEST_TIMESTEPS, 
                       help='timesteps to test our model')
    parser.add_argument('--episode-length', type=int, default=EPISODE_LENGTH, 
                       help='max timesteps in an episode')
    parser.add_argument('--train', default=True, type=boolean_string, 
                       help='is it training?')
    parser.add_argument('--load-checkpoint', type=bool, default=MODEL_LOAD, 
                       help='resume training?')
    
    # Environment parameters
    parser.add_argument('--town', type=str, default="Town10HD_Opt", 
                       help='which town do you like?')
    parser.add_argument('--carla-host', type=str, default='localhost', 
                       help='CARLA server host (default: localhost)')
    parser.add_argument('--carla-port', type=int, default=2000, 
                       help='CARLA server port (default: 2000)')
    parser.add_argument('--driver-view', action='store_true', 
                       help='Enable driver camera view during training (requires pygame)')
    
    # Reward function parameters
    parser.add_argument('--reward-type', type=str, default='simple', 
                       help='Type of reward function to use (default: simple)')
    parser.add_argument('--list-rewards', action='store_true', 
                       help='List available reward functions and exit')
    
    # Testing and debugging parameters
    parser.add_argument('--test-env', action='store_true', 
                       help='Test environment setup only')
    
    # PyTorch parameters
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), 
                       default=True, nargs='?', const=True, 
                       help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), 
                       default=True, nargs='?', const=True, 
                       help='if toggled, cuda will not be enabled by default')
    
    args = parser.parse_args()
    
    return args


def validate_args(args):
    """
    Validate parsed arguments and provide helpful error messages.
    
    Args:
        args: Parsed arguments from parse_args()
        
    Raises:
        ValueError: If arguments are invalid
    """
    # Validate algorithm choice
    if args.algo not in ['ppo', 'dqn']:
        raise ValueError(f"Algorithm '{args.algo}' not supported. Choose from: ppo, dqn")
    
    # Validate timesteps
    if args.total_timesteps <= 0:
        raise ValueError("Total timesteps must be positive")
    
    if args.test_timesteps <= 0:
        raise ValueError("Test timesteps must be positive")
    
    if args.episode_length <= 0:
        raise ValueError("Episode length must be positive")
    
    # Validate learning rate
    if args.learning_rate <= 0:
        raise ValueError("Learning rate must be positive")
    
    # Validate action std init
    if args.action_std_init <= 0:
        raise ValueError("Action std init must be positive")
    
    # Validate CARLA port
    if not (1024 <= args.carla_port <= 65535):
        raise ValueError("CARLA port must be between 1024 and 65535")
    
    # Validate seed
    if args.seed < 0:
        raise ValueError("Seed must be non-negative")


def get_args():
    """
    Get and validate command line arguments.
    
    Returns:
        argparse.Namespace: Validated arguments
    """
    args = parse_args()
    validate_args(args)
    return args


if __name__ == "__main__":
    # Test the argument parser
    args = get_args()
    print("Parsed arguments:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}") 