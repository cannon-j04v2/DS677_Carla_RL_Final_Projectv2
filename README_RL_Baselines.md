# RL Baselines for CARLA Autonomous Driving

This project implements multiple reinforcement learning baselines for autonomous driving in CARLA 0.9.15, including PPO (Proximal Policy Optimization) and DQN (Deep Q-Network).

## ✅ Current Status

**FULLY IMPLEMENTED AND TESTED** ✅
- ✅ BaseAgent interface with abstract contract
- ✅ DQN implementation with discrete action space
- ✅ PPO implementation with continuous action space
- ✅ Real CARLA environment integration
- ✅ Experience replay and target networks
- ✅ ε-greedy exploration
- ✅ Checkpoint saving/loading
- ✅ TensorBoard logging
- ✅ Multi-town support

## Quick Start

### Prerequisites
- CARLA 0.9.15
- Python 3.7+
- PyTorch
- TensorBoard
- Virtual environment (recommended)

### Setup
1. **Start CARLA Server**:
   ```bash
   CarlaUE4.exe
   ```
   Wait for CARLA to fully load (30-60 seconds)

2. **Activate Virtual Environment**:
   ```bash
   .venv\Scripts\activate  # Windows
   ```

### Running the Baselines

#### DQN Training
```bash
python continuous_driver.py --algo dqn --town Town10HD_Opt --total-timesteps 1000000 --train True
```

#### PPO Training
```bash
python continuous_driver.py --algo ppo --town Town10HD_Opt --total-timesteps 1000000 --train True
```

#### Testing (Evaluation)
```bash
python continuous_driver.py --algo dqn --town Town10HD_Opt --train False --test-timesteps 10000
```

#### Environment Testing
```bash
python continuous_driver.py --test-env --town Town10HD_Opt
```

### Command Line Arguments

- `--algo`: Algorithm to use (`ppo` or `dqn`)
- `--town`: CARLA town to use (default: Town10HD_Opt)
- `--total-timesteps`: Total training timesteps
- `--train`: Whether to train or test (True/False)
- `--test-timesteps`: Number of timesteps for testing
- `--episode-length`: Maximum episode length
- `--seed`: Random seed for reproducibility
- `--carla-host`: CARLA server host (default: localhost)
- `--carla-port`: CARLA server port (default: 2000)
- `--test-env`: Test environment setup only

### Algorithm Details

#### PPO (Proximal Policy Optimization)
- **Type**: On-policy actor-critic algorithm
- **Action Space**: Continuous (steer, throttle)
- **Features**: Action noise decay, episodic learning
- **Network**: Actor-critic with shared encoder

#### DQN (Deep Q-Network)
- **Type**: Off-policy Q-learning algorithm
- **Action Space**: Discrete (27 actions: 9 steer × 3 throttle)
- **Features**: Experience replay, target networks, ε-greedy exploration
- **Network**: Fully connected Q-network
- **Replay Buffer**: 10,000 experience capacity
- **Target Update**: Every 1000 steps

### Checkpoints and Logging

- **Checkpoints**: Saved in `checkpoints/{ALGO}/{TOWN}/`
  - DQN: `dqn_model_{number}.pth`
  - PPO: `ppo_policy_{number}.pth` + `checkpoint_ppo_{number}.pickle`
- **Logs**: TensorBoard logs in `runs/{ALGO}_{PARAMS}/{TOWN}/`
- **DQN-specific logs**: Epsilon values, Q-loss tracking

### Available Towns

The following towns are available in CARLA 0.9.15:
- `Town01` / `Town01_Opt`
- `Town02` / `Town02_Opt`
- `Town03` / `Town03_Opt`
- `Town04` / `Town04_Opt`
- `Town05` / `Town05_Opt`
- `Town10HD` / `Town10HD_Opt` (default)

### Testing Import Structure
```bash
python continuous_driver.py --test-baseline-imports
```

## Architecture

```
networks/
├── common/
│   └── base_agent.py          # Abstract base class
├── on_policy/
│   └── PPO/
│       ├── agent.py           # PPO implementation
│       └── ppo.py             # Actor-critic networks
└── off_policy/
    └── DQN/
        ├── agent.py           # DQN implementation
        ├── action_map.py      # Discrete action mapping (27 actions)
        └── replay_buffer.py   # Experience replay buffer
```

## Troubleshooting

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

## Training Progress

### What to Expect
1. **Episode 1-10**: Random exploration (high epsilon)
2. **Episode 10-50**: Learning phase (epsilon decaying)
3. **Episode 50+**: Policy improvement (lower epsilon)

### Monitoring
- **Console Output**: Episode rewards, epsilon values
- **CARLA Window**: Real-time vehicle behavior
- **TensorBoard**: Detailed training metrics

### Success Indicators
- Increasing average rewards
- Decreasing epsilon (DQN)
- Vehicle staying on road
- No frequent collisions

## Future Baselines

The architecture supports easy addition of new algorithms:
- SAC (Soft Actor-Critic)
- TD3 (Twin Delayed DDPG)
- A2C (Advantage Actor-Critic)

Simply implement the `BaseAgent` interface and register in `ALGO_MAP`.

## Implementation Details

### DQN Hyperparameters
- Learning Rate: 1e-3
- Epsilon: 1.0 → 0.01 (decay: 0.995)
- Buffer Size: 10,000
- Batch Size: 32
- Target Update: Every 1000 steps

### PPO Hyperparameters
- Learning Rate: 1e-4
- Clip Ratio: 0.2
- Gamma: 0.99
- Action Std: 0.2 → 0.05 (decay)

### Environment Features
- **Vehicle**: Tesla Model 3
- **Camera**: RGB sensor (160x80)
- **Collision Detection**: Collision sensor
- **Reward**: Speed-based with collision penalties
- **Episode Length**: 1000 steps max 