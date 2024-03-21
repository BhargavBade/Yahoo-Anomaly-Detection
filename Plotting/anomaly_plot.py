import matplotlib.pyplot as plt
import os
import numpy as np
from ccbdl.storages import storages

def testdata_plotting(path,
                      test_data_tensor, 
                      testdata_rec, 
                      test_labels, 
                      pred_labels, 
                      figure_storage,
                      random_seed=42):
                        
    figs = []
    names = []   
    path = path
    data_tensor = test_data_tensor.cpu()
    rec_data_tensor = testdata_rec.cpu()
    actual_labels = test_labels.cpu()
    predicted_labels = pred_labels.cpu()
    fig_storage = storages.FigureStorage(path, dpi=300, types=("png", "pdf"))
    
    testdata = data_tensor.numpy()
    rec_testdata = rec_data_tensor.numpy()
    testdata_labels = actual_labels.numpy()
    predicted_labels = predicted_labels.numpy()
    
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
            rec_data = rec_testdata[idx]
            true_labels = testdata_labels[idx]
            pred_labels = predicted_labels[idx]

            flat_data = data.flatten()
            flat_rec_data = rec_data.flatten()
            flat_true_labels = true_labels.flatten()
            flat_pred_labels = pred_labels.flatten()

            true_label_indices = np.where(flat_true_labels == 1)[0]
            true_label_values = flat_data[true_label_indices]

            pred_label_indices = np.where(flat_pred_labels == 1)[0]
            pred_label_values = flat_data[pred_label_indices]

            ax.plot(flat_data, label='org data')
            ax.plot(flat_rec_data, label='rec data')
            
            # Scatter Plot
            ax.scatter(true_label_indices, true_label_values, color='red', 
                       marker='o', label='true anm', alpha=0.5, s=100)
            ax.scatter(pred_label_indices, pred_label_values, color='green', 
                       marker='x', label='pred anm', s=150)
            
            ax.legend(fontsize='small', loc="best")

            plt.tight_layout()

        figs.append(fig)
        names.append(os.path.join("AnmTest", "Figure_" + name))

    fig_storage.store_multi(figs, names, folder="", dpis=False)
