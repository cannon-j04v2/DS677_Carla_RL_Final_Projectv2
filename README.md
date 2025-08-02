# CARLA Autonomous Driving with Reinforcement Learning

This project combines the CARLA autonomous driving simulator with advanced reinforcement learning baselines for autonomous vehicle training and testing.

## 🚗 About CARLA

CARLA is an open-source simulator for autonomous driving research. It provides a realistic urban environment with traffic, pedestrians, and various weather conditions for testing autonomous driving algorithms.

**Website**: [carla.org](http://carla.org/)  
**Documentation**: [carla.readthedocs.io](http://carla.readthedocs.io)

## 🎯 Features

- 🚗 CARLA 0.9.15 simulator integration
- 🧠 Modular reward system with multiple reward functions
- 📈 PPO (Proximal Policy Optimization) with continuous action space
- 🎮 Enhanced DQN with Double DQN, soft target updates, and early stopping
- 🌍 Real CARLA environment integration with sensors
- 🔄 Experience replay and target networks
- 🎲 ε-greedy exploration with configurable parameters
- 💾 Checkpoint saving/loading system
- 📊 TensorBoard logging and monitoring
- 🗺️ Multi-town support
- ⚙️ Comprehensive argument parser with validation
- 🧪 Extensive testing suite

## 🏗️ Project Structure

```
├── continuous_driver.py    # Main training/testing script
├── get_args.py            # Argument parser and validation
├── parameters.py          # Hyperparameters and constants
├── rewards/               # Modular reward system
│   ├── __init__.py
│   ├── base_reward.py     # Abstract base class
│   ├── simple_reward.py   # Basic speed-based rewards
│   ├── advanced_reward.py # Multi-objective rewards
│   ├── custom_reward.py   # CARLA sensor-based rewards
│   ├── reward_factory.py  # Reward function factory
│   └── README.md
├── networks/              # RL algorithms
│   ├── common/
│   │   └── base_agent.py
│   ├── on_policy/
│   │   └── PPO/
│   └── off_policy/
│       ├── DQN/
│       └── DQN_FINALTEST/      # Enhanced DQN with advanced features
├── test_rewards.py        # Test rewards system
├── test_custom_reward.py  # Test custom reward
└── test_args.py          # Test argument parser
```

## 🚀 Quick Start

### Prerequisites
- CARLA 0.9.15
- Python 3.7+
- PyTorch
- TensorBoard
- Virtual environment (recommended)

### Setup

1. **Start CARLA Server**:
   ```bash
   # Windows
   CarlaUE4.exe
   
   # Linux
   ./CarlaUE4.sh
   ```
   Wait for CARLA to fully load (30-60 seconds)

2. **Activate Virtual Environment**:
   ```bash
   # Windows
   .venv\Scripts\activate
   
   # Linux/Mac
   source .venv/bin/activate
   ```

3. **Add Traffic** (Optional):
   ```bash
   ./generate_traffic.py -n 80
   ```

4. **Dynamic Weather** (Optional):
   ```bash
   ./dynamic_weather.py
   ```

5. **Camera Position**: The 3rd person camera is positioned at (x=-6, z=3)
   with a -15° pitch angle relative to the vehicle center

## 🎯 Running the RL Baselines

### Basic Training

#### Enhanced DQN Training
```bash
python continuous_driver.py --algo dqn_finaltest --town Town10HD_Opt --total-timesteps 15000 --train True --reward-type advanced --seed 22 --episode-length 200 --target-update-method soft --soft-update-tau 0.01 --driver-view --double-dqn --early-stop --early-stop-patience 40 --early-stop-tolerance 40 --learn-starts 2000 --learn-freq 8
```

#### Enhanced DQN Testing
```bash
python continuous_driver.py --algo dqn_finaltest --town Town10HD_Opt --total-timesteps 2000 --train False --reward-type advanced --seed 22 --episode-length 200 --driver-view --double-dqn
```

#### PPO Training
```bash
python continuous_driver.py --algo ppo --town Town10HD_Opt --total-timesteps 1000000 --train True
```

### Advanced Usage with Custom Rewards

#### Using Custom Reward Function
```bash
python continuous_driver.py --algo ppo --reward-type custom --town Town10HD_Opt
```

#### Using Advanced Reward Function
```bash
python continuous_driver.py --algo dqn_finaltest --reward-type advanced --town Town10HD_Opt --double-dqn
```

#### List Available Reward Functions
```bash
python continuous_driver.py --list-rewards
```

### Testing and Evaluation

#### Testing (Evaluation)
```bash
python continuous_driver.py --algo dqn_finaltest --town Town10HD_Opt --total-timesteps 2000 --train False --reward-type advanced --seed 22 --episode-length 200 --driver-view --double-dqn
```

#### Environment Testing
```bash
python continuous_driver.py --test-env --town Town10HD_Opt
```

#### Manual Control (Original CARLA)
```bash
./manual_control.py
```

## 📋 Command Line Arguments

### Algorithm Parameters
- `--algo`: Algorithm to use (`ppo`, `dqn`, or `dqn_finaltest`)
- `--learning-rate`: Learning rate of the optimizer
- `--total-timesteps`: Total training timesteps
- `--test-timesteps`: Number of timesteps for testing
- `--episode-length`: Maximum episode length
- `--action-std-init`: Initial exploration noise (PPO)
- `--seed`: Random seed for reproducibility

### Enhanced DQN Parameters
- `--double-dqn`: Enable Double DQN for reduced overestimation bias
- `--target-update-method`: Target network update method (`hard` or `soft`)
- `--soft-update-tau`: Soft update coefficient for target network (e.g., 0.01)
- `--early-stop`: Enable early stopping based on performance
- `--early-stop-patience`: Episodes to wait before early stopping
- `--early-stop-tolerance`: Reward improvement threshold for early stopping
- `--learn-starts`: Timesteps before learning begins
- `--learn-freq`: Frequency of learning updates
- `--random-test`: Force random actions during testing (for baseline comparison)

### Environment Parameters
- `--town`: CARLA town to use (default: Town10HD_Opt)
- `--carla-host`: CARLA server host (default: localhost)
- `--carla-port`: CARLA server port (default: 2000)
- `--env-name`: Simulation environment name
- `--driver-view`: Enable driver camera view during training/testing (requires pygame)

### Driver View Feature
The `--driver-view` flag enables a real-time 3rd person camera view of the agent during training or testing. This feature:

- **Real-time Visualization**: Shows a 3rd person perspective of the agent's vehicle
- **Interactive**: Close the pygame window to continue without the view
- **Performance**: Minimal impact on training speed
- **Requirements**: Requires pygame to be installed (`pip install pygame`)

**Usage Examples:**
```bash
# Training with driver view
python continuous_driver.py --algo dqn_finaltest --total-timesteps 15000 --driver-view --double-dqn

# Testing with driver view
python continuous_driver.py --algo dqn_finaltest --train False --total-timesteps 2000 --driver-view --double-dqn

# Test the feature
python test_driver_view.py
```

### Training Parameters
- `--train`: Whether to train or test (True/False)
- `--load-checkpoint`: Resume training from checkpoint
- `--torch-deterministic`: Enable deterministic PyTorch
- `--cuda`: Enable CUDA acceleration

### Reward System Parameters
- `--reward-type`: Type of reward function (`simple`, `advanced`, `custom`)
- `--list-rewards`: List available reward functions and exit

### Testing Parameters
- `--test-env`: Test environment setup only

## 🎮 Reward Functions

### SimpleReward (Default)
Basic speed-based rewards with collision penalties.
- Rewards forward movement
- Penalizes collisions
- Handles episode timeouts
- Detects when vehicle is stuck

### AdvancedReward (Recommended)
Sophisticated multi-objective reward function designed for highway driving scenarios.
- **Dense reward shaping**: Continuous feedback for all driving behaviors
- **Lane keeping system**: Dynamic rewards that scale over training with stability bonuses
- **Speed optimization**: Encourages 8-12 m/s optimal speed range
- **Anti-exploit measures**: Prevents circular driving and sustained steering
- **Progressive penalties**: Collision penalties increase over training (annealed)
- **Traffic compliance**: Off-road detection, wrong-way penalties
- **Smooth driving incentives**: Rewards for gradual throttle/steering changes
- **Early success termination**: Ends episodes early when driving well

### CustomReward
Sophisticated reward function using CARLA sensor data and waypoints.
- **Angle alignment**: Rewards proper alignment with road orientation
- **Lane centering**: Rewards staying close to the center of the road
- **Speed optimization**: Rewards maintaining target speed
- **Collision penalties**: Heavy penalties for collisions

## 🧠 Algorithm Details

### PPO (Proximal Policy Optimization)
- **Type**: On-policy actor-critic algorithm
- **Action Space**: Continuous (steer, throttle)
- **Features**: Action noise decay, episodic learning
- **Network**: Actor-critic with shared encoder
- **Hyperparameters**:
  - Learning Rate: 1e-4
  - Clip Ratio: 0.2
  - Gamma: 0.99
  - Action Std: 0.2 → 0.05 (decay)

### Enhanced DQN (Deep Q-Network)
- **Type**: Off-policy Q-learning algorithm with advanced features
- **Action Space**: Discrete (10 actions)
- **Features**: 
  - Double DQN for reduced overestimation bias
  - Soft target network updates with configurable tau
  - Early stopping with patience and tolerance thresholds
  - Configurable learning start and frequency
  - Experience replay and ε-greedy exploration
- **Network**: Fully connected Q-network
- **Hyperparameters**:
  - Learning Rate: 1e-3
  - Epsilon: 1.0 → 0.01 (decay: 0.995)
  - Buffer Size: 10,000
  - Batch Size: 32
  - Soft Update Tau: 0.01 (configurable)
  - Learn Starts: 2000 timesteps
  - Learn Frequency: Every 8 steps

## 🗺️ Available Towns

The following towns are available in CARLA 0.9.15:
- `Town01` / `Town01_Opt`
- `Town02` / `Town02_Opt`
- `Town03` / `Town03_Opt`
- `Town04` / `Town04_Opt`
- `Town05` / `Town05_Opt`
- `Town10HD` / `Town10HD_Opt` (default)

## 📊 Checkpoints and Logging

### Checkpoints
- **Location**: `checkpoints/{ALGO}/{TOWN}/`
- **DQN**: `dqn_model_{number}.pth`
- **DQN_FINALTEST**: `dqn_model_{number}.pth`
- **PPO**: `ppo_policy_{number}.pth` + `checkpoint_ppo_{number}.pickle`

### TensorBoard Logs
- **Location**: `runs/{ALGO}_{PARAMS}/{TOWN}/`
- **Metrics**: Episode rewards, epsilon values, Q-loss tracking, reward components

## 🧪 Testing

### Test the Rewards System
```bash
python test_rewards.py
```

### Test Custom Reward Function
```bash
python test_custom_reward.py
```

### Test Argument Parser
```bash
python test_args.py
```

### Test Import Structure
```bash
python continuous_driver.py --test-baseline-imports
```

### Test Driver View Feature
```bash
python test_driver_view.py
```

## 🔧 Troubleshooting

### Connection Issues
- **"Connection refused"**: Make sure CARLA server is running
- **"Map not found"**: Use available towns (see list above)
- **"Timeout"**: Increase timeout or wait for CARLA to fully load

### Training Issues
- **No vehicle movement**: Check if environment test passes
- **No checkpoints saved**: Ensure training loop reaches save points
- **Replay buffer warnings**: Fixed in latest version

### Environment Issues
- **Vehicle not visible**: Check spawn point collisions
- **Camera issues**: Verify sensor setup
- **Performance**: Reduce image resolution if needed
- **Driver view not working**: Install pygame (`pip install pygame`)

### Reward Function Issues
- **Waypoint errors**: Ensure CARLA world is properly loaded
- **Invalid reward type**: Use `--list-rewards` to see available options

## 📈 Training Progress

### What to Expect
1. **Episode 1-10**: Random exploration (high epsilon)
2. **Episode 10-50**: Learning phase (epsilon decaying)
3. **Episode 50+**: Policy improvement (lower epsilon)

### Monitoring
- **Console Output**: Episode rewards, epsilon values, reward components
- **CARLA Window**: Real-time vehicle behavior
- **TensorBoard**: Detailed training metrics

### Success Indicators
- Increasing average rewards
- Decreasing epsilon (DQN)
- Vehicle staying on road
- No frequent collisions
- Proper lane keeping (with custom rewards)

## 🔮 Future Enhancements

The modular architecture supports easy addition of:
- **New Algorithms**: SAC, TD3, A2C
- **New Reward Functions**: Traffic light compliance, pedestrian avoidance
- **New Sensors**: LiDAR, radar, GPS
- **New Environments**: Different weather conditions, traffic scenarios

## 📚 Implementation Details

### Environment Features
- **Vehicle**: Tesla Model 3
- **Camera**: RGB sensor (160x80)
- **Collision Detection**: Collision sensor
- **Episode Length**: 1000 steps max

### Argument Parser
- **Location**: `get_args.py`
- **Features**: Validation, error handling, comprehensive help
- **Functions**: `get_args()`, `parse_args()`, `validate_args()`

### Reward System
- **Modular Design**: Easy to swap and extend reward functions
- **Factory Pattern**: Clean creation and management
- **Configuration**: Customizable parameters for each reward function
- **Validation**: Robust error handling and fallbacks

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is based on CARLA, which is licensed under the MIT License. See the original CARLA documentation for more details. 