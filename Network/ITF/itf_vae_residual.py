import torch
from ccbdl.network.base import BaseNetwork

from Network.ITF.itf_var_autoencoder import VarAutoencoder
from ccbdl.utils import DEVICE
from Network.ITF import functions as f

class ITFResidualVarAutoencoder(BaseNetwork):
    
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
            
            self.VAE = VarAutoencoder(window_length, 
                                               function_pool, 
                                               attention,
                                               k,
                                               pass_z)
            
            self.layers.append(self.VAE)

    def kl_zf(self):
        kl = 0
        for i in range(0, len(self.layers)):
            kl += self.layers[i].kl_zf()
        return kl/len(self.layers) 
    
    def latent_plotting(self, inp):
        zp_agg = torch.zeros_like(self.layers[0].zp_latent(inp))  # Initialize with zeros
        for layer in self.layers:
            zp_agg += layer.zp_latent(inp)  # Aggregate the values
        return zp_agg / len(self.layers)
       
           
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
                rec = rec                                                
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
    net = ITFResidualVarAutoencoder(network_config).to(DEVICE)

    out, zp  = net(data)

    print(net)
    print(data.shape)
    print(out.shape)
    print(zp.shape)

    import matplotlib.pyplot as plt
    plt.plot(out[0,0].detach().cpu())
    plt.show()



