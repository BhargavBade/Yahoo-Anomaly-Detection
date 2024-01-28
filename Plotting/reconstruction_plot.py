import matplotlib.pyplot as plt
from abc import ABC
import os
from ccbdl.evaluation.plotting.base import GenericPlot


class ReconstructionPlot(GenericPlot, ABC):
    def __init__(self, learner, *args, **kwargs):
        super(ReconstructionPlot, self).__init__(learner, repeat=1)

    def consistency_check(self, *args, **kwargs):
        return True

    def preprocessing(self):
        pass

    # remove from data storage
    def postprocessing(self):
        self.learner.data_storage.reset_entry("Data")

    def plot(self, *args, **kwargs):
        figs = []
        names = []
        inp, rec = self.learner.data_storage.get_item("Data", batch=False)[0]
        inp = inp.cpu()
        rec = rec.cpu()
        
        # show 2 images and reconstruction
        f, axarr = plt.subplots(2,1)
        axarr[0].title.set_text("B1")
        axarr[0].plot(inp[0,0], label="Input")
        axarr[0].plot(rec[0,0], label="Reconstruction")
        axarr[0].legend(loc="upper right")
        axarr[1].title.set_text("B2")
        axarr[1].plot(inp[1,0], label="Input")
        axarr[1].plot(rec[1,0], label="Reconstruction")
        axarr[1].legend(loc="upper right")
        figs.append(f)
        names.append(os.path.join("Test", "reconst_plot_" + str(self.learner.epoch)))
        return figs, names

    def __repr__(self):
        return "reconstruction"
