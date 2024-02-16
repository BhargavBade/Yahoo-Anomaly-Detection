import torch
from ccbdl.network.base import BaseNetwork
from ccbdl.utils import DEVICE
from Network.ITF import functions as f
# from Network.ITF.itf_var_autoencoder import VarAutoencoder
from Network.ITF.itf_vae_encoder import VarEncoder
from Network.ITF.itf_vae_decoder import VarDecoder


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
       
        self.function_pool = function_pool
        self.encoder_layers = torch.nn.ModuleList()
        self.decoder_layers = torch.nn.ModuleList()

        self.layers = torch.nn.ModuleList()
        for i in range(stages):
            
            self.encoder = VarEncoder(window_length, 
                                len(function_pool),
                                sum([f[1] for f in function_pool]),
                                attention,
                                k,
                                pass_z)
            
            self.decoder = VarDecoder(function_pool, window_length)
            
            self.encoder_layers.append(self.encoder)
            self.decoder_layers.append(self.decoder)

    def forward(self, inp: torch.tensor, test=False, get_ls=False, get_noise=False):
        rec_lst=[] 
        encoded_lst = []
        # get latent space and output
        if get_ls:
            ls = []
            for encoder, decoder in zip(self.encoder_layers, self.decoder_layers):
                encoded = encoder(inp)
                encoded_lst.append(encoded)
                rec = decoder(*encoded)
                rec_lst.append(rec)
                inp = inp - rec
                ls.append((self.function_pool, *encoded))
            out = sum(rec_lst)
            return out, ls

        if test:
            rest_lst=[]
            info_lst = []
            for encoder, decoder in zip(self.encoder_layers, self.decoder_layers):
                encoded = encoder(inp)
                encoded_lst.append(encoded)
                rec = decoder(*encoded)
                inp = inp - rec
                rec_lst.append(rec)
                rest_lst.append(inp)
                info = get_info(*encoded, self.function_pool)
                info_lst.append(info)
            out = sum(rec_lst)
            return out, rest_lst, rec_lst, info_lst
        
        for encoder, decoder in zip(self.encoder_layers, self.decoder_layers):
            encoded = encoder(inp)
            encoded_lst.append(encoded)
            zf, zp = encoded
            rec = decoder(zf, zp)
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

    out = net(data)

    print(net)
    print(data.shape)
    print(out.shape)


    import matplotlib.pyplot as plt
    plt.plot(out[0,0].detach().cpu())
    plt.show()
    