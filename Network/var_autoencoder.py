
from ccbdl.network.base import BaseNetwork
import torch
from torch import nn

class MyVarAutoEncoder(BaseNetwork):
    
    def __init__(self, name : str, act_function : torch.nn.modules.activation, 
                 input_size:int, hidden_size : int, debug = False):
    
        super().__init__(name, debug)
        
        self.encoder = nn.Sequential(
            # nn.Linear(120,100),
            # act_function(),
            nn.Linear(input_size,80),
            # act_function(),    
            nn.Linear(80,64),
            # act_function(),
            nn.Linear(64,32),
            # act_function(),          
            )
        
        self.en_mu = nn.Linear(32, hidden_size)
        self.en_logvar = nn.Linear(32, hidden_size)
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 32),
            # act_function(),
            nn.Linear(32, 64),
            # act_function(),
            nn.Linear(64, 80),
            # act_function(),
            )
        
        self.de_mu = nn.Linear(80, input_size)
        self.de_logvar =  nn.Linear(80, input_size)
            
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
            latent_mu = self.en_mu(x)
            latent_logvar = self.en_logvar(x)
            z = self.reparameterize(latent_mu, latent_logvar)
            decoded = self.decoder(z)
            recon_mu =  self.de_mu(decoded)
            recon_logvar = self.de_logvar(decoded)
            self.kl_value = self.kl(latent_mu,latent_logvar)
            rec_z = self.reparameterize(recon_mu, recon_logvar)
            return rec_z
    
            
if __name__ == '__main__':
    inp = torch.rand(32,1,120)   
    net = MyVarAutoEncoder("Vartest", torch.nn.LeakyReLU, 120, 20)
    out = net(inp)    
    
    