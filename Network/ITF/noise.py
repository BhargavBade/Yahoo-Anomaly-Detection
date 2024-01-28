import torch
from ccbdl.utils import DEVICE

def white_noise(shape:tuple, scale:torch.Tensor = 1, mean:torch.Tensor = 0):
    return torch.randn(shape, device=DEVICE) * scale.unsqueeze(1) + mean.unsqueeze(1)


def red_noise(shape:tuple , scale:torch.Tensor = 1, mean:torch.Tensor = 0, r:torch.Tensor=0.5):
    # assert not False in (r<1), "R vale >= 1"
    
    w = torch.randn(shape, device=DEVICE)
    x = torch.empty(shape, device=DEVICE)
    
    length = shape[-1]
    
    x[...,0] = w[...,0].clone()
    
    for i in range(0,length-1):
        x[...,i+1] = r*x[...,i].clone() + torch.pow(1-torch.pow(r,2),0.5) * w[...,i+1]
    
    
    
    return x * scale.unsqueeze(1) + mean.unsqueeze(1)



def unif_noise(shape:tuple, scale:torch.Tensor = 1, mean:torch.Tensor = 0):
    return torch.rand(shape, device=DEVICE) * scale.unsqueeze(1) + mean.unsqueeze(1)



if __name__ == '__main__':
    bs=1
    wl=10
    
    scale = torch.ones(bs, device=DEVICE)
    mean = torch.zeros(bs, device=DEVICE)
    r = torch.rand(bs, device = DEVICE)
    
    w_noise = white_noise((bs,wl), scale, mean)
    u_noise = unif_noise((bs,wl),scale, mean)
    r_noise = red_noise((bs,wl),scale, mean, r)
    print(r_noise)
    
    import matplotlib.pyplot as plt
    
    plt.plot(r_noise[0].detach().cpu(), label = "Red")
    plt.plot(w_noise[0].cpu(), label = "White")
    plt.plot(u_noise[0].cpu(), label = "Uniform")   
    plt.legend()
    plt.show()    
    
    # r_noise.mean().backward()    
    
    # import torchviz
    # import os
    # os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
    # torchviz.make_dot(r_noise.mean()).view()
    
    
    
    
    
    
    
    
    
    