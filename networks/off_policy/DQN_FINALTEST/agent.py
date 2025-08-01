import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from networks.common.base_agent import BaseAgent
from networks.off_policy.DQN_FINALTEST.replay_buffer import ReplayBuffer
from networks.off_policy.DQN_FINALTEST.action_map import index_to_action, get_action_space_size

class DQNNetwork(nn.Module):
    """Q-Network for DQN with flexible architecture."""
    def __init__(self, num_hidden=4, hidden_dim=512, state_dim=115, n_action=10):
        super(DQNNetwork, self).__init__()

        self.input_layer = nn.Linear(state_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden)])
        self.output_layer = nn.Linear(hidden_dim, n_action)
        
    def forward(self, x):
        """
        Define the forward pass.
        The forward pass includes the input layer, hidden layers, and output layer.
        Between each layer, we apply the ReLU activation function.
        No activation function is applied after the output layer.
        """
        x = self.input_layer(x)
        x = torch.relu(x)
        
        for layer in self.hidden_layers:
            x = layer(x)
            x = torch.relu(x)
            
        action_value = self.output_layer(x)
        return action_value

class DQNAgent(BaseAgent):
    def __init__(self, town, obs_dim=115, action_dim=10, lr=1e-4, gamma=0.99, 
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.9995, 
                 buffer_size=50000, batch_size=64, target_update_freq=1000,
                 num_hidden=4, hidden_dim=512, target_update_method='soft', 
                 soft_update_tau=0.01, use_double_dqn=False):
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
        self.num_hidden = num_hidden
        self.hidden_dim = hidden_dim
        self.target_update_method = target_update_method
        self.soft_update_tau = soft_update_tau
        self.use_double_dqn = use_double_dqn
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks - updated to use new architecture
        self.online_net = DQNNetwork(
            num_hidden=num_hidden,
            hidden_dim=hidden_dim,
            state_dim=obs_dim,
            n_action=action_dim
        ).to(self.device)
        
        self.target_net = DQNNetwork(
            num_hidden=num_hidden,
            hidden_dim=hidden_dim,
            state_dim=obs_dim,
            n_action=action_dim
        ).to(self.device)
        
        self.target_net.load_state_dict(self.online_net.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        # LR scheduler: halve LR every 10k gradient steps
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.5)
        
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
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.online_net(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values (target network)
        with torch.no_grad():
            if self.use_double_dqn:
                best_actions = self.online_net(next_states).argmax(1)
                next_q_values = self.target_net(next_states).gather(1, best_actions.unsqueeze(1)).squeeze()
            else:
                next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        td_error = target_q_values - current_q_values.squeeze()
        td_error = torch.clamp(td_error, -10.0, 10.0)
        loss = torch.nn.functional.smooth_l1_loss(current_q_values.squeeze() + td_error - td_error, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 5)   # new
        self.optimizer.step()
        # step LR scheduler
        self.lr_scheduler.step()
        
        # Update target network
        self.step_count += 1
        if self.target_update_method == 'hard':
            # Hard update: periodic full copy
            if self.step_count % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.online_net.state_dict())
        elif self.target_update_method == 'soft':
            # Soft update: gradual update every step
            self._soft_update_target_network()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()

    def _soft_update_target_network(self):
        """
        Soft update of target network parameters.
        θ_target = τ * θ_local + (1 - τ) * θ_target
        """
        # Your Code #
        # update the target network by using the soft update method. The formula is:
        # target_net = target_net * (1 - tau) + eval_net * tau
        # tau is the soft_update_tau
        # You need to use the state_dict() method to get the parameters of the network
        # The parameters of the network are stored in a dictionary. So you need to update the parameters for each key in the dictionary
        # You also need to use the load_state_dict() method to load the parameters to the target network
        
        tau = self.soft_update_tau
        target_state_dict = self.target_net.state_dict()
        online_state_dict = self.online_net.state_dict()
        
        # Update each parameter in the target network
        for key in target_state_dict:
            target_state_dict[key] = target_state_dict[key] * (1 - tau) + online_state_dict[key] * tau
        
        # Load the updated parameters back to the target network
        self.target_net.load_state_dict(target_state_dict)

    def save(self, path=None):
        """Save model checkpoint."""
        if path is None:
            # Create checkpoint directory if it doesn't exist
            checkpoint_dir = f'checkpoints/DQN_FINALTEST/{self.town}'
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            self.checkpoint_file_no = len([f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')])
            path = f'{checkpoint_dir}/dqn_model_{self.checkpoint_file_no}.pth'
        
        checkpoint = {
            'online_net_state_dict': self.online_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'checkpoint_file_no': self.checkpoint_file_no,
            'num_hidden': self.num_hidden,
            'hidden_dim': self.hidden_dim,
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'target_update_method': self.target_update_method,
            'soft_update_tau': self.soft_update_tau,
            'use_double_dqn': self.use_double_dqn
        }
        torch.save(checkpoint, path)

    def load(self, path=None):
        """Load model checkpoint."""
        if path is None:
            # Load latest checkpoint
            checkpoint_dir = f'checkpoints/DQN_FINALTEST/{self.town}'
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
            
            # Load network architecture parameters if available
            if 'num_hidden' in checkpoint and 'hidden_dim' in checkpoint:
                # Recreate networks with saved architecture
                self.num_hidden = checkpoint['num_hidden']
                self.hidden_dim = checkpoint['hidden_dim']
                
                self.online_net = DQNNetwork(
                    num_hidden=self.num_hidden,
                    hidden_dim=self.hidden_dim,
                    state_dim=self.obs_dim,
                    n_action=self.action_dim
                ).to(self.device)
                
                self.target_net = DQNNetwork(
                    num_hidden=self.num_hidden,
                    hidden_dim=self.hidden_dim,
                    state_dim=self.obs_dim,
                    n_action=self.action_dim
                ).to(self.device)
                
                # Recreate optimizer
                self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.lr)
            
            self.online_net.load_state_dict(checkpoint['online_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.step_count = checkpoint.get('step_count', 0)
            self.checkpoint_file_no = checkpoint.get('checkpoint_file_no', 0)
            self.target_update_method = checkpoint.get('target_update_method', self.target_update_method)
            self.soft_update_tau = checkpoint.get('soft_update_tau', self.soft_update_tau)
            self.use_double_dqn = checkpoint.get('use_double_dqn', self.use_double_dqn)
            print(f"Loaded checkpoint from: {path}")
            print(f"Target update method: {self.target_update_method}, Tau: {self.soft_update_tau}")
        else:
            print(f"Checkpoint file not found: {path}") 