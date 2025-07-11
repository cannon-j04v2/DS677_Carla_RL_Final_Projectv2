import numpy as np
import random
import torch

class ReplayBuffer:
    def __init__(self, capacity):
        """
        Initialize replay buffer with given capacity.
        
        Args:
            capacity (int): Maximum number of experiences to store
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def add(self, state, action, reward, next_state, done):
        """
        Add experience to buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        experience = (state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
            
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Sample a batch of experiences.
        
        Args:
            batch_size (int): Number of experiences to sample
            
        Returns:
            tuple: (states, actions, rewards, next_states, dones)
        """
        if len(self.buffer) < batch_size:
            batch = self.buffer
        else:
            batch = random.sample(self.buffer, batch_size)
            
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert tensors to numpy arrays if needed
        def convert_to_numpy(data):
            if isinstance(data, torch.Tensor):
                return data.detach().cpu().numpy()
            elif isinstance(data, list) and len(data) > 0:
                return [convert_to_numpy(item) for item in data]
            else:
                return data
        
        states = [convert_to_numpy(state) for state in states]
        next_states = [convert_to_numpy(state) for state in next_states]
        
        return (states, 
                np.array(actions), 
                np.array(rewards), 
                next_states, 
                np.array(dones))

    def __len__(self):
        """Return current size of buffer."""
        return len(self.buffer) 