import sys
import torch
from autoencoder.encoder import VariationalEncoder
import numpy as np

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
        if self.conv_encoder is not None:
            image_obs = torch.from_numpy(observation[0].copy()).float().to(self.device)
            image_obs = image_obs.permute(2, 0, 1).unsqueeze(0)
            
            with torch.no_grad():
                latent_vec = self.conv_encoder(image_obs).squeeze().cpu().numpy()
            
            nav_obs = observation[1]
            processed_observation = np.concatenate([latent_vec, nav_obs])
        else:
            # If the encoder model is not loaded, create a zero vector for the latent space
            latent_vec = np.zeros(self.latent_dim)
            nav_obs = observation[1]
            processed_observation = np.concatenate([latent_vec, nav_obs])
            
        return processed_observation