import torch
import torch.nn as nn
from collections import OrderedDict

class ParameterHead(nn.Module):
    def __init__(self, inp_size: int, num_parameters: int):
        super().__init__()

        self.head_parameters = nn.Sequential(OrderedDict([
            ('h_pa_lin_1', nn.Linear(inp_size, inp_size//2)),
            ('h_pa_relu_1', nn.ReLU()),
            ('h_pa_lin_2', nn.Linear(inp_size//2, num_parameters)),
            ('h_pa_sig_2', nn.Sigmoid())]))
        
    # input is backbone -> output are parameters [0,1] 
    def forward(self, inp: torch.tensor):
        return self.head_parameters(inp)
    
    
if __name__ == '__main__':

    # generate data
    data = torch.rand(64, 1, 100)

    # get model
    net = ParameterHead(100, 5)

    out = net(data)

    print(net)
    print(data.shape)
    print(out.shape)
    print(out[0])