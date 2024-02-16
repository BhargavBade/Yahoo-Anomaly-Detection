
import torch
import torch.nn as nn
import Network.ITF.functions as f
from ccbdl.utils import DEVICE


def get_params(idx:int, count:int, data:torch.tensor):
    if count == 1:
        return data[...,idx:idx+1],
    elif count == 2:
        return data[...,idx:idx+1], data[...,idx+1:idx+2]
    elif count == 3:
        return data[...,idx:idx+1], data[...,idx+1:idx+2], data[...,idx+2:idx+3]
    else:
        assert False ,"Count of parameters not valid"

class VarDecoder(nn.Module):
    def __init__(self, function_pool: list, window_length: int = 100):
        super().__init__()
        self.function_pool = function_pool
        self.x = torch.arange(0, window_length, 1, device=DEVICE)

        # check format of function pool
        assert type(function_pool[0]) == tuple, "No Parameters Given in Pool Definition"

    def decode(self, 
               zf: torch.tensor, 
               zp: torch.tensor):

        out = torch.zeros(1, requires_grad=True, device=DEVICE)
        p_idx = 0
        # other functions
        for i in range(0,len(self.function_pool)):
            out = out + self.function_pool[i][0](*get_params(p_idx,self.function_pool[i][1],zp),
                                              rescale=self.function_pool[i][2],
                                              TMAX = len(self.x))(self.x) * zf[..., i:i+1]
            p_idx += self.function_pool[i][1]

        return out

    def forward(self, zf, zp):
        return self.decode(zf,zp)


if __name__ == '__main__':

    parameters_all = 16
    functions = 6
    function_pool = [(f.Const,1, True), 
                     (f.Sin,3, True), 
                     (f.Gaus,3, True), 
                     (f.Sqrt,3, True), 
                     (f.Exp_Sat,3, True), 
                     (f.Exp_Decay,3, True)]
    
    from CustomModels import CustomHardlim
    # Hard 1,0
    zf = torch.rand(64, 1, functions, device=DEVICE)
    zf = CustomHardlim(0.5)(zf)
    
    # Soft 0-1
    zp = torch.rand(64, 1, parameters_all, device=DEVICE)

    net = VarDecoder(function_pool,100).to(DEVICE)
    out = net(zf, zp)

    # input
    print("Input Size: \t\t", zf.shape)
    print("Function Selection:\t", zf.shape)
    print("Parameters:\t\t\t", zp.shape)

    # output
    print("Output Size: \t\t", out.shape)

    
    import matplotlib.pyplot as plt
    plt.plot(out[0,0].detach().cpu())

    plt.show()
