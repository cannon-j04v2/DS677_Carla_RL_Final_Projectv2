import sys
import torch
from autoencoder.encoder import VariationalEncoder

class EncodeState():
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            self.conv_encoder = VariationalEncoder(self.latent_dim).to(self.device)
            self.conv_encoder.load()
            self.conv_encoder.eval()

            for params in self.conv_encoder.parameters():
                params.requires_grad = False
        except Exception as e:
            print(f'Encoder could not be initialized: {e}. Using placeholder for testing.')
            # Create a simple placeholder encoder for testing
            self.conv_encoder = None
    
    def process(self, observation):
        if self.conv_encoder is None:
            # Return placeholder observation for testing
            return torch.zeros(self.latent_dim + len(observation[1]) if len(observation) > 1 else self.latent_dim)
        
        image_obs = torch.tensor(observation[0], dtype=torch.float).to(self.device)
        image_obs = image_obs.unsqueeze(0)
        image_obs = image_obs.permute(0,3,2,1)
        image_obs = self.conv_encoder(image_obs)
        navigation_obs = torch.tensor(observation[1], dtype=torch.float).to(self.device)
        observation = torch.cat((image_obs.view(-1), navigation_obs), -1)
        
        return observation