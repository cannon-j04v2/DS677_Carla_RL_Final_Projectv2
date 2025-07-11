import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from networks.common.base_agent import BaseAgent
from networks.off_policy.DQN.replay_buffer import ReplayBuffer
from networks.off_policy.DQN.action_map import index_to_action, get_action_space_size

class DQNNetwork(nn.Module):
    """Q-Network for DQN."""
    def __init__(self, obs_dim, action_dim):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent(BaseAgent):
    def __init__(self, town, obs_dim=100, action_dim=27, lr=1e-3, gamma=0.99, 
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, 
                 buffer_size=10000, batch_size=32, target_update_freq=1000):
        super().__init__()
        
        self.town = town
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.online_net = DQNNetwork(obs_dim, action_dim).to(self.device)
        self.target_net = DQNNetwork(obs_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Training variables
        self.step_count = 0
        self.checkpoint_file_no = 0

    def get_action(self, state, train=True):
        """
        Select action using ε-greedy policy.
        
        Args:
            state: Current state (list of [image, nav_info])
            train: Whether in training mode
            
        Returns:
            tuple: (steer, throttle, brake) action
        """
        if train and np.random.random() < self.epsilon:
            # Random action
            action_idx = np.random.randint(0, self.action_dim)
        else:
            # Greedy action
            with torch.no_grad():
                # Process state through encoder (simplified for now)
                if isinstance(state, list) and len(state) == 2:
                    # Use navigation info as state for now
                    nav_info = state[1]
                    if isinstance(nav_info, torch.Tensor):
                        state_tensor = nav_info.unsqueeze(0).to(self.device)
                    else:
                        state_tensor = torch.FloatTensor(nav_info).unsqueeze(0).to(self.device)
                else:
                    # Fallback
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
                q_values = self.online_net(state_tensor)
                action_idx = q_values.argmax().item()
        
        return index_to_action(action_idx), action_idx  # Return both action tuple and index

    def learn(self):
        """Perform one learning step."""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.online_net(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values (target network)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path=None):
        """Save model checkpoint."""
        if path is None:
            # Create checkpoint directory if it doesn't exist
            checkpoint_dir = f'checkpoints/DQN/{self.town}'
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            self.checkpoint_file_no = len([f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')])
            path = f'{checkpoint_dir}/dqn_model_{self.checkpoint_file_no}.pth'
        
        checkpoint = {
            'online_net_state_dict': self.online_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'checkpoint_file_no': self.checkpoint_file_no
        }
        torch.save(checkpoint, path)

    def load(self, path=None):
        """Load model checkpoint."""
        if path is None:
            # Load latest checkpoint
            checkpoint_dir = f'checkpoints/DQN/{self.town}'
            if not os.path.exists(checkpoint_dir):
                print(f"No checkpoint directory found: {checkpoint_dir}")
                return
            
            checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
            if not checkpoint_files:
                print(f"No checkpoint files found in: {checkpoint_dir}")
                return
            
            self.checkpoint_file_no = len(checkpoint_files) - 1
            path = f'{checkpoint_dir}/dqn_model_{self.checkpoint_file_no}.pth'
        
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.online_net.load_state_dict(checkpoint['online_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.step_count = checkpoint.get('step_count', 0)
            self.checkpoint_file_no = checkpoint.get('checkpoint_file_no', 0)
            print(f"Loaded checkpoint from: {path}")
        else:
            print(f"Checkpoint file not found: {path}") 