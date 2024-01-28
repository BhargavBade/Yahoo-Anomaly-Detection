import torch
import torch.nn as nn
from Network.ITF.CustomModels import GetTopK, CustomHardlim, Ones
from ccbdl.utils import DEVICE


class FunctionHead(nn.Module):
    def __init__(self, 
                 inp_size:int, 
                 num_functions: int,
                 attention:str = "hard",
                 k: int = 1):
        super().__init__()
    
        assert k <= num_functions, "More Functions selected than given Pool"
    
        self.head_functions = nn.Sequential()
        self.head_functions.add_module('h_fu_lin_1',    nn.Linear(inp_size, inp_size//2))
        self.head_functions.add_module('h_fu_relu_1',   nn.ReLU())
        self.head_functions.add_module('h_fu_lin_2',    nn.Linear(inp_size//2, num_functions))


        if attention == "hard":
            self.head_functions.add_module("hard",
                                           nn.Sequential(nn.Sigmoid(),
                                                         GetTopK(k),
                                                         CustomHardlim(0.2)))
        elif attention == "soft":
            self.head_functions.add_module("soft",
                                           nn.Sequential(nn.Sigmoid(),
                                                         GetTopK(k))) 
            
        elif attention == "softmax":
            self.head_functions.add_module("softmax", 
                                           nn.Sequential(nn.Softmax(dim=-1),
                                                         GetTopK(k))) 
            
        elif attention == "real_soft":
            self.head_functions.add_module("real_soft",
                                           nn.Sequential(nn.Sigmoid()))
            
        elif attention == "real_softmax":
            self.head_functions.add_module("real_softmax",
                                           nn.Sequential(nn.Softmax(dim=-1))) 
            
        elif attention == "all":
            self.head_functions.add_module("All",
                                           nn.Sequential(Ones())) 
        else:
            print("Wrong attention method selected: " + attention)
        

    # input is backbone -> output is k selected functions
    def forward(self, inp):
        return self.head_functions(inp)
    
    
if __name__ == '__main__':

    # generate data
    data = torch.rand(64, 1, 50, device=DEVICE)

    # get model
    net = FunctionHead(50, 10, "hard", 3).to(DEVICE)

    out = net(data)

    print(net)
    print(data.shape)
    print(out.shape)
    print(out[0])
