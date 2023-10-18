from ccbdl.network.base import BaseNetwork
import torch
from torch import nn


class MyAutoEncoder(BaseNetwork):
    def __init__(self, name : str, act_function : torch.nn.modules.activation, input_size:int, hidden_size : int, debug = False):
        super().__init__(name, debug)
        self.encoder = torch.nn.Sequential(nn.Linear(input_size,32),
                                            # act_function(),                                        
                                           nn.Linear(32, hidden_size),
                                           )
        
        self.decoder = torch.nn.Sequential(nn.Linear(hidden_size, 32),
                                            # act_function(),
                                           nn.Linear(32, input_size),
                                           )
        
    def forward(self, x):
        latent_features = self.encoder(x)
        return self.decoder(latent_features)

if __name__ == '__main__':
    inp = torch.rand(64,1,50)
    
    net = MyAutoEncoder("test", torch.nn.ReLU, 50, 10)
    out = net(inp)
    print(out.shape)
    print(net)
    