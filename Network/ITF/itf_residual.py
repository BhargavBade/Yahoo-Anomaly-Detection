import torch
from ccbdl.network.base import BaseNetwork

from Network.ITF.itf_autoencoder import Autoencoder

from ccbdl.utils import DEVICE
from Network.ITF import functions as f

class ITFResidualAutoencoder(BaseNetwork):
    def __init__(self, network_config: dict):
        super().__init__("Interpretable TimeSeries Feature (IFT-RES-AE)", False)
        
        # RESIDUAL
        stages = int(network_config["stages"])
        
        # AE
        window_length = network_config["window_length"] 
        function_pool = network_config["function_pool"] 
        attention = network_config["attention"]
        k = network_config["k"]
        pass_z = network_config["pass_z"]
                
        self.layers = torch.nn.ModuleList()
        for i in range(stages):
            self.layers.append(Autoencoder(window_length, 
                                           function_pool, 
                                           attention,
                                           k,
                                           pass_z))

        
        
    def forward(self, inp: torch.tensor, test=False, get_ls=False, get_noise=False):
        rec_lst=[] 

        # get latent space and output
        if get_ls:
            ls = []
            for layer in self.layers:
                rec, ls_layer = layer(inp, get_ls=True)
                rec_lst.append(rec)
                inp = inp - rec
                ls.append(ls_layer)
            out = sum(rec_lst)
            return out, ls

        if test:
            rest_lst=[]
            info_lst = []
            for layer in self.layers:
                rec, info = layer(inp, test=True)
                inp = inp - rec
                rec_lst.append(rec)
                rest_lst.append(inp)
                info_lst.append(info)
            out = sum(rec_lst)

            return out, rest_lst, rec_lst, info_lst
        
        for layer in self.layers:
            rec = layer(inp)
            inp = inp - rec
            rec_lst.append(rec)
            
        return sum(rec_lst)


if __name__ == '__main__':

    # params  
    network_config={"stages": 3,
                     "window_length": 100,
                     "function_pool": [(f.Sin,3,True), (f.Const,1,True), (f.Gaus,3,True)],
                     "attention": "soft",
                     "k": 3,
                     "pass_z":True,
                     "noise_pattern": "white"}
    
    
    
    # generate data
    data = torch.rand(64, 1, network_config["window_length"]).to(DEVICE)

    # get model
    net = ITFResidualAutoencoder(network_config).to(DEVICE)

    out = net(data)

    print(net)
    print(data.shape)
    print(out.shape)


    import matplotlib.pyplot as plt
    plt.plot(out[0,0].detach().cpu())
    plt.show()



