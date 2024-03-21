
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.distributions
from Network.ITF.itf_vae_function_head import VAEFunctionHead
from Network.ITF.itf_vae_param_head import VAEParameterHead
from torch.distributions import Normal, kl_divergence
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
        self.N = torch.distributions.Normal(0, 1)    
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
        self.function_head = VAEFunctionHead(self.out_size,
                                          num_functions,
                                          attention,
                                          k)
        
        # create parameter head
        self.parameter_head = VAEParameterHead(parameter_head_input,
                                            num_parameters)
    

    def dkl_normal(mu, sigma):
        return -0.5* (1+ torch.log(sigma**2) - mu**2 - sigma**2)

    def forward(self, inp: torch.tensor):
        # backbone
        out_b = self.backbone(inp)

        # function head
        zf = self.function_head(out_b)
        
        if self.pass_z:
            out_b = torch.cat((out_b, zf), dim=-1)
        
        # parameter head
        zp_lat_mu, zp_lat_logvar = self.parameter_head(out_b)
        zp = zp_lat_mu + zp_lat_logvar *self.N.sample(zp_lat_mu.shape).to(DEVICE) 
        
        #kl div loss
        self.kl_zf =  VarEncoder.dkl_normal(zp_lat_mu, zp_lat_logvar)
        
        return zf, zp, self.kl_zf 


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

    zf, zp, kl_loss = net(data)

    print(net)
    print("Input Size: \t\t", data.shape)
    print("Function Selection:\t", zf.shape)
    print(zf[0].detach())
    print("Parameters:\t\t\t", zp.shape)
    print(zp[0].detach())
