
import torch
import torch.nn as nn
from collections import OrderedDict
# from Network.ITF.itf_backbone import Backbone
from Network.ITF.itf_function_head import FunctionHead
from Network.ITF.itf_param_head import ParameterHead

from ccbdl.utils import DEVICE


class VarEncoder(nn.Module):
    def __init__(self,
                 inp_size: int,
                 num_functions: int,
                 num_parameters: int,
                 attention: str = "soft",
                 k: int = 1,
                 pass_z: bool = False):

        super().__init__()
        self.pass_z = pass_z
        
        # create backbone
        
        # self.backbone = Backbone(inp_size)
        
        self.out_size = inp_size//4
        
        self.backbone = nn.Sequential(OrderedDict([
                                      ('b_lin_1',   nn.Linear(inp_size, inp_size*3)),
                                      ('b_relu_1',  nn.ReLU()),
                                      ('b_lin_2',   nn.Linear(inp_size*3, inp_size)),
                                      ('b_relu_2',  nn.ReLU()),
                                      ('b_lin_3',   nn.Linear(inp_size, self.out_size)),
                                      ('b_relu_3',  nn.Sigmoid())]))
        
        # define inputs for all heads
        if self.pass_z:
            parameter_head_input = self.out_size + num_functions 
            
        else:
            parameter_head_input = self.out_size

        # create function head
        self.function_head = FunctionHead(self.out_size,
                                          num_functions,
                                          attention,
                                          k)
        
        # create parameter head
        self.parameter_head = ParameterHead(parameter_head_input,
                                            num_parameters)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z_reparametrized = mu + eps * std
        return z_reparametrized    


    def forward(self, inp: torch.tensor):
        # backbone
        out_b = self.backbone(inp)

        # function head
        zf = self.function_head(out_b)

        if self.pass_z:
            out_b = torch.cat((out_b, zf), dim=-1)
        
        # parameter head
        latent_mu = out_b
        latent_logvar = out_b
        z = self.reparameterize(latent_mu, latent_logvar)
        zp = self.parameter_head(z)

        return zf, zp


if __name__ == '__main__':

    # generate data
    data = torch.rand(64, 1, 100, device=DEVICE)

    # get model
    window_length = 100
    num_functions = 10
    num_parameters = 12
    k = 4
    attention = "soft"
    pass_z = False

    net = VarEncoder(window_length,
                 num_functions,
                 num_parameters,
                 attention,
                 k,
                 pass_z).to(DEVICE)

    zf,  zp = net(data)

    print(net)
    print("Input Size: \t\t", data.shape)
    print("Function Selection:\t", zf.shape)
    print(zf[0].detach())
    print("Parameters:\t\t\t", zp.shape)
    print(zp[0].detach())
