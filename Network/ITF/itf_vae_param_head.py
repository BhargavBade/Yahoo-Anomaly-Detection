import torch
import torch.nn as nn
from collections import OrderedDict

class VAEParameterHead(nn.Module):
    def __init__(self, inp_size: int, num_parameters: int):
        super().__init__()

        self.head_parameters = nn.Sequential(OrderedDict([
            ('h_pa_lin_1', nn.Linear(inp_size, inp_size//2)),
        ]))

        self.head_parameters1 = nn.Sequential(OrderedDict([
            ('h_pa_lin_2', nn.Linear(inp_size//2, num_parameters)),
            ('h_pa_sig_1', nn.Sigmoid()) 
        ]))

        self.head_parameters2 = nn.Sequential(OrderedDict([
            ('h_pa_lin_3', nn.Linear(inp_size//2, num_parameters)),
            ('h_pa_sig_2', nn.Sigmoid()) 
        ]))
        
    # input is backbone -> output are parameters [0,1] 
    def forward(self, inp: torch.tensor):
        x = self.head_parameters(inp)
        head_parm_mu = self.head_parameters1(x)
        head_parm_logvar = self.head_parameters2(x)
        return head_parm_mu, head_parm_logvar
    
    
if __name__ == '__main__':

    # generate data
    data = torch.rand(64, 1, 100)

    # get model
    net = VAEParameterHead(100, 5)

    out = net(data)

    print(net)
    print(data.shape)
    print(out[0].shape)
    print(out[1].shape)