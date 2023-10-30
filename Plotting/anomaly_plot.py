import matplotlib.pyplot as plt
import os
import numpy as np
from ccbdl.storages import storages

def testdata_plotting(path,
                      test_data_tensor, 
                      testdata_rec, 
                      test_labels, 
                      pred_labels, 
                      figure_storage):
                        
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

    random_indices = np.random.choice(testdata.shape[0], size=120, replace=False)

    figures = 20
    subplots_per_figure = 6

    for f in range(figures):
        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(20, 15))
        fig.suptitle(f"Figure {f + 1}")
        name = f"Figure {f + 1}"

        for s, ax in enumerate(axs.flat):
            idx = f * subplots_per_figure + s

            data = testdata[random_indices[idx]]
            rec_data = rec_testdata[random_indices[idx]]
            true_labels = testdata_labels[random_indices[idx]]
            pred_labels = predicted_labels[random_indices[idx]]

            flat_data = data.flatten()
            flat_rec_data = rec_data.flatten()
            flat_true_labels = true_labels.flatten()
            flat_pred_labels = pred_labels.flatten()

            true_label_indices = np.where(flat_true_labels == 1)[0]
            true_label_values = flat_data[true_label_indices]

            pred_label_indices = np.where(flat_pred_labels == 1)[0]
            pred_label_values = flat_data[pred_label_indices]

            ax.plot(flat_data)
            ax.plot(flat_rec_data)
            ax.scatter(true_label_indices, true_label_values, color='red', 
                       label='true_labels', alpha=0.6)
            ax.scatter(pred_label_indices, pred_label_values, color='green', 
                       marker='x', label='pred_labels', s=100)

            ax.legend(loc="best")

        plt.tight_layout()

        figs.append(fig)
        names.append(os.path.join("Test", "Random_Sample_" + name))

    fig_storage.store_multi(figs, names, folder="", dpis=False)