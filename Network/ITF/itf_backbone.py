import torch
import torch.nn as nn
from collections import OrderedDict


class Backbone(nn.Module):
    def __init__(self, inp_size: int):
        super().__init__()
        self.out_size = inp_size//4
        
        self.backbone = nn.Sequential(OrderedDict([
                                      ('b_lin_1',   nn.Linear(inp_size, inp_size*3)),
                                      ('b_relu_1',  nn.ReLU()),
                                      ('b_lin_2',   nn.Linear(inp_size*3, inp_size)),
                                      ('b_relu_2',  nn.ReLU()),
                                      ('b_lin_3',   nn.Linear(inp_size, self.out_size)),
                                      ('b_relu_3',  nn.Sigmoid())]))
   
    # input is data -> output is input//4
    def forward(self, inp: torch.tensor):
        return self.backbone(inp)
    
    
if __name__ == '__main__':
    data = torch.rand(64,1,100)
    
    backbone = Backbone(100)
    
    out = backbone(data)
    
    print(data.shape)
    print(out.shape)
    print(backbone)