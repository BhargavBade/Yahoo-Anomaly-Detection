import torch
from ccbdl.utils import DEVICE
from ccbdl.utils.datatypes import TaskPool
from ccbdl.learning.auto_encoder import BaseAutoEncoderLearning
from ccbdl.evaluation.plotting.reconstruct import ReconstructionLosses, StaticReconstructions
from Plotting.reconstruction_plot import ReconstructionPlot
from torch.distributions import Normal, kl_divergence

class LearnVarAutoEncoder(BaseAutoEncoderLearning):
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
                 **kwargs):
        super().__init__(train_data, test_data, val_data, 
                          data_storage_names=["epoch", "batch",
                                              "train_loss", "kl_loss","test_loss"],
                         path=trial_path, config=config, task=task, debug=debug)
        self.network = network
        self.trial = trial
        
        self.parameter_storage.equal_signs()
        self.criterion = getattr(torch.nn, self.criterion)()  #For LAE Network
        self.optimizer = getattr(torch.optim, self.optimizer)(
            self.network.parameters(), lr=self.learning_rate)

        self.scheduler = getattr(torch.optim.lr_scheduler, self.scheduler)(
            self.optimizer, self.scheduler_step, gamma=self.gamma)
        
        self.plotter.register_custom_plot(ReconstructionPlot(self))
        self.plotter.register_custom_plot(StaticReconstructions(self, 6, **kwargs))  #__added_line__
        self.plotter.register_default_plot(ReconstructionLosses(self))
        
    # @tracer
    def _encode(self, ins):
        ins = ins.to(torch.float32)
        return self.network(ins)

    # @tracer
    def _decode(self, rec):
        rec = rec.to(torch.float32)
        return rec

    # @tracer
    def _train_epoch(self, train=True):
        self.network.train() 
        self.prior = Normal(0, 1)
        kl_weight = 0.2
        losses = 0               
        for _, (inp, _)  in enumerate(self.train_data):
            
            inp = inp.to(torch.float32)
            inp = inp.to(DEVICE)           
            for param in self.network.parameters(): 
                param.grad = None
                
            # network
            enc = self.network.encoder(inp)
            latent_mu = self.network.en_mu(enc)
            latent_logvar = self.network.en_logvar(enc)
            latent_stddev = torch.exp(0.5 * latent_logvar)            
            latent_dist = Normal(latent_mu, latent_stddev)                   
            z = self.network.reparameterize(latent_mu, latent_logvar)
            decoded = self.network.decoder(z)
            rec_z = self.network.de_mu(decoded)
            
            self.rec_loss = self.criterion(rec_z, inp)
            kl = kl_divergence(latent_dist, self.prior).mean()

            self.kl_divergence_loss = kl
            self.loss = self.rec_loss + kl_weight*self.kl_divergence_loss
            self.loss.backward() 
            losses+= self.loss
            self.optimizer.step()
            
            self.data_storage.store(
                [self.epoch, self.batch, self.loss, self.kl_divergence_loss, self.test_loss])

            self.batch += 1
        self.train_loss = losses / len(self.train_data)
        self.scheduler.step()
       
    # # @tracer
    def _test_epoch(self):
        self.network.eval()
        self.prior = Normal(0, 1)
        kl_weight = 0.2
        loss = 0 
        with torch.no_grad():              
            for i, (inp, _)  in enumerate(self.test_data):
                
                inp = inp.to(torch.float32)
                inp = inp.to(DEVICE)              
                    
                ## network
                enc = self.network.encoder(inp)
                latent_mu = self.network.en_mu(enc)
                latent_logvar = self.network.en_logvar(enc)
                latent_stddev = torch.exp(0.5 * latent_logvar)            
                latent_dist = Normal(latent_mu, latent_stddev)        
                z = self.network.reparameterize(latent_mu, latent_logvar)
                decoded = self.network.decoder(z)               
                rec_z = self.network.de_mu(decoded)
                
                self.rec_loss = self.criterion(rec_z, inp).item()
                kl = kl_divergence(latent_dist, self.prior).mean()
        
                self.kl_divergence_loss = kl  
                self.loss = self.rec_loss + kl_weight * self.kl_divergence_loss
                loss+= self.loss    
                
                            
        self.test_loss = loss / len(self.test_data)
        
        self.data_storage.dump_store("Data", (inp, rec_z))

    # @tracer
    def _hook_every_epoch(self):
        pass
    
    # @tracer
    def _validate_epoch(self):
        pass
        # return self._test_epoch()

    # @tracer
    def _update_best(self):
        if self.test_loss < self.best_values["TestLoss"]:
            self.best_values["TestLoss"] = self.test_loss
            self.best_values["TrainLoss"] = self.train_loss
            self.best_values["Batch"] = self.batch
            self.best_values["Epoch"] = self.epoch

            self.best_state_dict = self.network.state_dict()
            
    # @tracer
    def evaluate(self):
        self.data_storage.store([self.epoch, self.batch, self.loss, self.kl_divergence_loss, 
                                 self.test_loss], force=self.batch)
        self._hook_every_epoch() 

    # @tracer
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
        self.parameter_storage.write_tab("Network", str(self.network))
        
        # save best values but not model dict
        self.parameter_storage.store({k: self.best_values[k] for k in self.best_values.keys() - {'model_state_dict'}}, header="Best Values")
        