"""
Created on Thu Sep 14 15:32:22 2023

@author: BhargavBade
"""

from ccbdl.network.base import BaseNetwork
import torch
from torch import nn

class MyVarAutoEncoder_Yahoo(BaseNetwork):
    
    def __init__(self, name : str, act_function : torch.nn.modules.activation,
                  hidden_size : int, debug = False):
    
        super().__init__(name, debug)
        
        self.encoder = nn.Sequential(
            # nn.Linear(120,64),
            # act_function(),
            nn.Linear(50,32),
            act_function(),
          
        )
        
        self.fc_mu = nn.Linear(32, hidden_size)
        self.fc_logvar = nn.Linear(32, hidden_size)
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 32),
            act_function(),
            nn.Linear(32, 50),
            # act_function(),
            # nn.Linear(64, 120),
            # nn.ReLU(),
            # act_function(),
        )  

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z_reparametrized = mu + eps * std
        return z_reparametrized
    
    def kl(self,latent_mean,latent_logvar):
        return -0.5 * torch.mean(1 + latent_logvar -
                                    latent_mean.pow(2) -latent_logvar.exp())


           
    def forward(self,x):
            x = self.encoder(x)
            mu = self.fc_mu(x)
            logvar = self.fc_logvar(x)
            z = self.reparameterize(mu, logvar)
            decoded = self.decoder(z)
            self.kl_value = self.kl(mu,logvar)
            return decoded
    
    
if __name__ == '__main__':
    inp = torch.rand(32,1,50)   
    net = MyVarAutoEncoder_Yahoo("Vartest", torch.nn.LeakyReLU, 20)
    out = net(inp)    
    
    