import torch
from ccbdl.network.base import BaseNetwork

from itf_vae_encoder import VarEncoder
from itf_vae_decoder import VarDecoder

from ccbdl.utils import DEVICE
from Network.ITF import functions as f


def get_params(idx:int, count:int, data:torch.tensor):
    if count == 1:
        return data[...,idx:idx+1],
    elif count == 2:
        return data[...,idx:idx+1], data[...,idx+1:idx+2]
    elif count == 3:
        return data[...,idx:idx+1], data[...,idx+1:idx+2], data[...,idx+2:idx+3]
    else:
        assert False ,"Count of parameters not valid"

def get_info(out_zf, out_zp, function_pool):
    info = "Information about BOF parameters and probabilities\n"
    # each batch
    batches = out_zf.shape[0]
    channels = out_zf.shape[1]
    
    for batch in range(batches):
        info += "\n" + "Batch: " + str(batch) + "\t"
        # each channel
        for ch in range(channels):
            body=""
            if ch!=0:
                body+="\t\t\t"
            body+= "Channel: " + str(ch) + "\t"
            #each func
            p_idx = 0
            for func in range(len(function_pool)):
                
                # get func
                temp = function_pool[func][0](*get_params(p_idx,function_pool[func][1],out_zp),
                                              True,
                                              TMAX = 100).__str__((batch,ch)).strip()
                p_idx += function_pool[func][1]
                
                # add probability 
                if out_zf.shape[0] == 1 and out_zf.shape[1] == 1:
                    temp = str(float(out_zf[0,0,func]))[:6] + "*(" + temp + ")"
                else:
                    temp = str(float(out_zf[batch,ch,func]))[:6] + "*(" + temp + ")"
                
                body += temp
                body += "\n"
                if func+1 != len(function_pool):
                    body += "\t\t\t\t\t\t"
            info += body  
    return info


class VarAutoencoder(BaseNetwork):
    def __init__(self,
                 inp_size: int,
                 function_pool:list,
                 attention: str = "hard",
                 k: int = 1,
                 pass_z: bool = False):
        super().__init__("Interpretable TimeSeries Feature (IFT-AE)", False)
        self.function_pool = function_pool

        # get infos from function pool
        num_functions = len(function_pool)
        num_parameters = sum([i[1] for i in function_pool])
                 
        self.encoder = VarEncoder(inp_size,
                                   num_functions,
                                   num_parameters,
                                   attention,
                                   k,
                                   pass_z)

        self.decoder = VarDecoder(function_pool, inp_size)

    def encode(self, inp: torch.tensor):
        return self.encoder(inp)

    def decode(self, ls: torch.tensor):
        return self.decoder(*ls)
       

    def forward(self, inp: torch.tensor, test=False, get_ls = False):
        # encoder
        ls = self.encode(inp)
        out = self.decode(ls)
        
        if test:
            info = get_info(*ls, self.function_pool)
            return out, info
        
        if get_ls:
            return out, (self.function_pool,*ls)
        
        return out


if __name__ == '__main__':

    # params
    bs = 10
    window_length = 100
    pass_z = True
    k = 2
    attention = "hard"
    function_pool = [(f.Sin, 3, True), (f.Const, 1, True), (f.Gaus, 3, True)]

    # generate data
    data = torch.rand(bs, 1, window_length).to(DEVICE)

    # get model
    net = VarAutoencoder(window_length, 
                      function_pool,
                      attention,
                      k,
                      pass_z).to(DEVICE)

    out = net(data)

    print(net)
    print(data.shape)
    print(out.shape)

    import matplotlib.pyplot as plt
    plt.plot(out[0, 0].detach().cpu())
    plt.show()
    
    
    zf, zp = net.encode(data)
    print(zf)
    print(zp)
    
    # get info
    text = get_info(zf,zp,function_pool)
    print(text)
    
    
    
    
    
