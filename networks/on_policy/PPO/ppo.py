import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import MultivariateNormal



class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, action_std_init):
        super(ActorCritic, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create our variable for the matrix.
        # Note that I chose 0.2 for stdev arbitrarily.
        self.cov_var = torch.full((self.action_dim,), action_std_init).to(self.device)

        # Create the covariance matrix
        self.cov_mat = torch.diag(self.cov_var).unsqueeze(dim=0).to(self.device)

        # actor
        self.actor = nn.Sequential(
                        nn.Linear(self.obs_dim, 500),
                        nn.Tanh(),
                        nn.Linear(500, 300),
                        nn.Tanh(),
                        nn.Linear(300, 100),
                        nn.Tanh(),
                        nn.Linear(100, self.action_dim),
                        nn.Tanh()
                    )
        
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(self.obs_dim, 500),
                        nn.Tanh(),
                        nn.Linear(500, 300),
                        nn.Tanh(),
                        nn.Linear(300, 100),
                        nn.Tanh(),
                        nn.Linear(100, 1)
                    )

    def forward(self):
        raise NotImplementedError
    
    def set_action_std(self, new_action_std):
        self.cov_var = torch.full((self.action_dim,), new_action_std).to(self.device)
        # Update the covariance matrix as well
        self.cov_mat = torch.diag(self.cov_var).unsqueeze(dim=0).to(self.device)


    def get_value(self, obs):
        # Ensure obs is a tensor and on the correct device
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        elif not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
            
        # Move to device if not already there
        if obs.device != self.device:
            obs = obs.to(self.device)
            
        # Ensure obs has the right shape (batch dimension)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)  # Add batch dimension
            
        return self.critic(obs)
    
    def get_action_and_log_prob(self, obs):
        # Query the actor network for a mean action.
        # Same thing as calling self.actor.forward(obs)
        
        # Ensure obs is a tensor and on the correct device
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        elif not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
            
        # Move to device if not already there
        if obs.device != self.device:
            obs = obs.to(self.device)
            
        # Ensure obs has the right shape (batch dimension)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)  # Add batch dimension
            
        mean = self.actor(obs)
        # Create our Multivariate Normal Distribution
        dist = MultivariateNormal(mean, self.cov_mat)
        # Sample an action from the distribution and get its log prob
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # Return the sampled action and the log prob of that action
        # Note that I'm calling detach() since the action and log_prob  
        # are tensors with computation graphs, so I want to get rid
        # of the graph and just convert the action to numpy array.
        # log prob as tensor is fine. Our computation graph will
        # start later down the line.
        return action.detach(), log_prob.detach()
    
    def evaluate(self, obs, action):

        # Ensure obs and action are on the correct device
        if obs.device != self.device:
            obs = obs.to(self.device)
        if action.device != self.device:
            action = action.to(self.device)

        mean = self.actor(obs)
        cov_var = self.cov_var.expand_as(mean)
        cov_mat = torch.diag_embed(cov_var)
        dist = MultivariateNormal(mean, cov_mat)
        
        logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        values = self.critic(obs)
        
        return logprobs, values, dist_entropy