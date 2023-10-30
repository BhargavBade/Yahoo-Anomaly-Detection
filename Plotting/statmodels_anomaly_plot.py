import matplotlib.pyplot as plt
import os
import numpy as np

def testdata_plotting(path,
                      test_data, 
                      test_labels, 
                      pred_labels, 
                      figure_storage):

    figss = []
    namess = []
    
    path = path
    testdata = test_data
    testdata_labels = test_labels
    predicted_labels = pred_labels
    fig_storage = figure_storage
    
    test_anm_indices = np.where(testdata_labels == 1)[0] 
    pred_anm_indices = np.where(predicted_labels == 1.0)[0]
    
    # Set the number of plots, sequences, and figures
    num_plots = 120
    num_sequences = 100
    num_figures = 20
    
    # Generate random indices for the sequences
    random_indices = np.random.choice(testdata.shape[0] - num_sequences, num_plots, replace=False)
    
    # Create the figures and subplots
    for figure_num in range(num_figures):
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20,15))
        fig.suptitle(f"Figure {figure_num+1}")
        name = f"Figure {figure_num+1}"
        
        # Generate random indices for the subplots
        random_sub_indices = np.random.choice(num_plots, 6, replace=False)
        
        # Plot the selected sequences
        for i, ax in enumerate(axes.flat):
            index = random_indices[random_sub_indices[i]]
            sequence = testdata[index:index+num_sequences]
            indices = np.arange(index, index+num_sequences)
            
            # Check if anomaly indices exist in the current sequence
            anomaly_mask = np.isin(indices, test_anm_indices)
            pred_anomaly_mask = np.isin(indices, pred_anm_indices)
            
            # Plot the sequence
            ax.plot(indices, sequence)
            
            # Highlight anomaly points as scatter points
            ax.scatter(indices[anomaly_mask], sequence[anomaly_mask], c='red', marker='o', label='true_labels', alpha = 0.6)
            ax.scatter(indices[pred_anomaly_mask], sequence[pred_anomaly_mask], c='green', label='pred_labels', marker='x')
            
            ax.set_title(f"Sequence {i+1}")
            ax.legend(loc="best")
            
        plt.tight_layout()
    
        # plt.show()            
        figss.append(fig)                               
        namess.append(os.path.join("Test", "Sample_" + name))        
    fig_storage.store_multi(figss, namess, folder="", dpis=False)
