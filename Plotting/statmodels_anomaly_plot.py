import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from ccbdl.storages import storages

def testdata_plotting(path,
                      test_data, 
                      test_labels, 
                      pred_labels, 
                      figure_storage,
                      random_seed=42):

    figss = []
    namess = []  
    path = path
    testdata = test_data
    testdata_labels = test_labels
    predicted_labels = pred_labels
    # fig_storage = figure_storage
    fig_storage = storages.FigureStorage(path, dpi=300, types=("png", "pdf"))
 
#----------------------------------------------------------------------------------------------------- 
    #Converting array into tensor
    num_sequences = 100
    no_of_samples = len(testdata) // (num_sequences)
    testdata_tensor = torch.Tensor(testdata).view(no_of_samples, 1, num_sequences)
    testlab_tensor = torch.Tensor(testdata_labels).view(no_of_samples, 1, num_sequences)
    predlab_tensor = torch.Tensor(predicted_labels).view(no_of_samples, 1, num_sequences)
    
    #converting tensors to arrays
    testdata = testdata_tensor.cpu().numpy()
    testdata_labels = testlab_tensor.cpu().numpy()
    predicted_labels = predlab_tensor.cpu().numpy()
#-----------------------------------------------------------------------------------------------------     
    
    # Set the number of plots, sequences, and figures
    figures = 60
    subplots_per_figure = 1
    
    np.random.seed(random_seed)  # Set the random seed
    random_indices = np.random.choice(testdata.shape[0], size=figures * subplots_per_figure, 
                                      replace=False)
     
    for f in range(figures):
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.suptitle(f"sample {f + 1}")
        name = f"{f + 1}"

        for s in range(subplots_per_figure):
            idx = random_indices[f * subplots_per_figure + s] % testdata.shape[0]
            
            data = testdata[idx]
            true_labels = testdata_labels[idx]
            pred_labels = predicted_labels[idx]

            flat_data = data.flatten()
            flat_true_labels = true_labels.flatten()
            flat_pred_labels = pred_labels.flatten()

            true_label_indices = np.where(flat_true_labels == 1)[0]
            true_label_values = flat_data[true_label_indices]

            pred_label_indices = np.where(flat_pred_labels == 1)[0]
            pred_label_values = flat_data[pred_label_indices]

            ax.plot(flat_data, label='data')
            
            # Scatter Plot
            ax.scatter(true_label_indices, true_label_values, color='red', 
                       marker='o', label='true anm', alpha=0.5, s=100)
            ax.scatter(pred_label_indices, pred_label_values, color='green', 
                       marker='x', label='pred anm', s=150)

            ax.legend(fontsize='small', loc="best")

            plt.tight_layout()

        figss.append(fig)
        namess.append(os.path.join("Test", "Sample_" + name))
    
    fig_storage.store_multi(figss, namess, folder="", dpis=False)
 