import torch
import os
import matplotlib.pyplot as plt
from ccbdl.utils import DEVICE
from ccbdl.storages import storages

def visualize_latent_space(path,train_data,network,learner):
    
    figure_storage = storages.FigureStorage(path, dpi=300, types=("png", "pdf"))
    latent_space = [] 
        
    with torch.no_grad():
        for _, (inp, labels) in enumerate(train_data):
            inputs = inp.to(DEVICE)
            enc = learner._encode(inputs)
            mu = network.fc_mu(enc)
            logvar = network.fc_logvar(enc)
            z = network.reparameterize(mu, logvar)
            latent_space.append(z)
        
    latent_space = torch.cat(latent_space, dim=0)
    latent_space = latent_space.reshape(latent_space.shape[0],-1)
    
    # Plot latent space dimensions
    num_plots = latent_space.shape[-1]
    figs = []
    names = []
    for i in range(num_plots-1):
        fig = plt.figure(figsize=(10,10))
        name = f"Figure {i + 1}"
        plt.scatter(latent_space[:,i], latent_space[:,i+1], cmap="viridis")
        plt.xlabel(f"dim {i+1}")
        plt.ylabel(f"dim {i+2}")
        plt.title(f' (Dimensions {i+1} and {i+2})')
        plt.colorbar()      
        figs.append(fig)
        names.append(os.path.join("Latent_Space","Lt_sp_" + name))
    figure_storage.store_multi(figs, names, folder="", dpis=False)    
