import torch
from ccbdl.utils import DEVICE
from ccbdl.utils.datatypes import TaskPool
from ccbdl.learning.auto_encoder import BaseAutoEncoderLearning
from ccbdl.evaluation.plotting.reconstruct import ReconstructionLosses
from ccbdl.evaluation.plotting.reconstruct import StaticReconstructions
from Plotting.reconstruction_info import ReconstructionInfo
from Plotting.reconstruction_plot import ReconstructionPlot

class LearnAutoEncoder(BaseAutoEncoderLearning):
    """
    
    Abstractmethods
    ---------------
    _train_epoch:
        training loop of one epoch.
    _test_epoch:
        testing loop of epoch.
    _validate_epoch:
        validation loop of one epoch.
    evaluate:
        evaluation (ploting + saving) of important metrics.
    learn:
        training, validating, testing and evaluating.
    _save:
        saving e.g. network and storages.
    """
    def __init__(self,
                 trial_path: str,
                 trial,
                 network,
                 train_data,
                 test_data,
                 val_data,
                 config: dict,
                 task: TaskPool,
                 debug=False,
                 logging=False):
        super().__init__(train_data, test_data, val_data,
                         path=trial_path, config=config, task=task, debug=debug, logging=logging)
        self.network = network
        self.trial = trial
        
        self.criterion = getattr(torch.nn, self.criterion)()
        self.optimizer = getattr(torch.optim, self.optimizer)(
            self.network.parameters(), lr=self.learning_rate)

        self.scheduler = getattr(torch.optim.lr_scheduler, self.scheduler)(
            self.optimizer, self.scheduler_step, gamma=self.gamma)

        self.plotter.register_custom_plot(ReconstructionPlot(self))
        self.plotter.register_custom_plot(ReconstructionInfo(self))
        self.plotter.register_default_plot(ReconstructionLosses(self))
        self.plotter.register_default_plot(StaticReconstructions(self, num=16))

    def _encode(self, data):
        data = data.to(torch.float32)
        return self.network(data)

    def _decode(self, rec):
        rec = rec.to(torch.float32)
        return rec

    def _train_epoch(self, train=True):
        self.network.train()
        losses = 0
        for _, (inp, _) in enumerate(self.train_data):
            for param in self.network.parameters():
                param.grad = None
            # get data
            inp = inp.to(torch.float32)
            inp = inp.to(DEVICE)

            # network
            reconstructions = self._decode(self._encode(inp))

            # loss
            self.loss = self.criterion(reconstructions, inp)
            self.loss.backward()
            
            losses += self.loss
            self.optimizer.step()
            self.data_storage.store(
                [self.epoch, self.batch, self.loss, self.test_loss])

            self.batch += 1
        self.train_loss = losses / len(self.train_data)
        self.scheduler.step()

    def _test_epoch(self):
        self.network.eval()
        loss = 0
        with torch.no_grad():
            for i, (inp, _) in enumerate(self.test_data):
                inp = inp.to(torch.float32)
                inp = inp.to(DEVICE)               
                reconstructions = self.network(inp)
                loss += self.criterion(reconstructions, inp).item()
                
                # get info of last batch
                if i >= len(self.test_data)-1:
                    reconstructions, _, _, info = self.network(inp, test = True)
                    
        self.test_loss = loss / len(self.test_data)

        self.data_storage.dump_store("Data", (inp, reconstructions))
        self.data_storage.dump_store("Info", info)
        
    def _hook_every_epoch(self):
        pass

    def _validate_epoch(self):
        pass

    def _update_best(self):
        if self.test_loss < self.best_values["TestLoss"]:
            self.best_values["TestLoss"] = self.test_loss
            self.best_values["TrainLoss"] = self.train_loss
            self.best_values["Batch"] = self.batch
            self.best_values["Epoch"] = self.epoch

            self.best_state_dict = self.network.state_dict()

    def evaluate(self):
        self.data_storage.store([self.epoch, self.batch, self.loss, self.test_loss],
                                force=self.batch)
        self._hook_every_epoch()

    def _save(self):

        # save current Net
        torch.save({'epoch': self.epoch,
                    'test_loss': self.test_loss,
                    'model_state_dict': self.network.state_dict()},
                   self.net_save_path)

        # save best Net
        torch.save({'epoch': self.best_values["Epoch"],
                    'test_loss': self.best_values["TestLoss"],
                    'model_state_dict': self.best_state_dict},
                   self.best_save_path)

        self.parameter_storage.store(self.best_values, "best_values")
        self.parameter_storage.store(str(self.network), "Network")
        
        # save best values but not model dict
        self.parameter_storage.store({k: self.best_values[k] for k in self.best_values.keys() - {'model_state_dict'}}, header="Best Values")