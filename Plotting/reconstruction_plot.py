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
        
        
        # # show 2 images and reconstruction
        # f, axarr = plt.subplots(2,2)
        # axarr[0,0].imshow(inp[0,0], label="Input")
        # axarr[0,1].imshow(rec[0,0], label="Reconstruction")
        # axarr[1,0].imshow(inp[1,0], label="Input")
        # axarr[1,1].imshow(rec[1,0], label="Reconstruction")
        
        for i in range(5):
            
            f=plt.figure()
            plt.plot((inp[i,0].cpu()), 'r-', label='input')
            plt.plot((rec[i,0].cpu()), 'b-', label='reconstructed')
            plt.legend(loc='best')
            figs.append(f)
            plt.show()
        
        # plt.plot(x[0],  data[0,0].cpu().detach(), 'r-', label='input (sin)')  
        # plt.legend(loc='best')       
        # plt.show()        

        # figs.append(f)
        names.append(os.path.join("Test", "reconst_plot_" + str(self.learner.epoch)))
        return figs, names

    def __repr__(self):
        return "reconstruction"