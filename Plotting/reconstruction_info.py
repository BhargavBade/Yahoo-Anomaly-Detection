from abc import ABC
import os
from ccbdl.evaluation.plotting.base import GenericPlot


class ReconstructionInfo(GenericPlot, ABC):
    def __init__(self, learner, *args, **kwargs):
        super(ReconstructionInfo, self).__init__(learner, repeat=1)

    def consistency_check(self, *args, **kwargs):
        if len(self.learner.data_storage.get_item("Info", batch=False)) == 0:
            return False
        else:
            return True

    def preprocessing(self):
        pass

    # remove from data storage
    def postprocessing(self):
        self.learner.data_storage.reset_entry("Info")

    def plot(self, *args, **kwargs):

        info = self.learner.data_storage.get_item("Info", batch=False)[0]
        name = "Test\\reconst_info" + str(self.learner.epoch) + '.txt'

        info_text = ''
        for i, txt in enumerate(info):
            info_text += "Stage" + str(i) + '\n\n'
            info_text += txt

        with open(os.path.join(self.learner.path, 'png', name), 'w') as f:
            f.write(info_text)

        return [], []

    def __repr__(self):
        return "info"
