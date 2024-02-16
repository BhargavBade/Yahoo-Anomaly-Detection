
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
    # @tracer
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
        # self.criterion = getattr(torch.nn, self.criterion)(reduction = 'sum')
        self.optimizer = getattr(torch.optim, self.optimizer)(
            self.network.parameters(), lr=self.learning_rate)

        self.scheduler = getattr(torch.optim.lr_scheduler, self.scheduler)(
            self.optimizer, self.scheduler_step, gamma=self.gamma)
        # self.plotter.register_custom_plot(ReconstructionPlot(self))
        self.plotter.register_custom_plot(StaticReconstructions(self, 6, **kwargs))  #__added_line__
        self.plotter.register_default_plot(ReconstructionLosses(self))
              
    # @tracer
    def _encode(self, ins):
        ins = ins.to(torch.float32)
        return self.network(ins)

    # @tracer
    def _decode(self, ins):
        ins = ins.to(torch.float32)
        return ins

    # @tracer
    def _train_epoch(self, train=True):
        self.network.train() 
        self.L = 32
        self.prior = Normal(0, 1)
        losses = 0               
        for _, (inp, _)  in enumerate(self.train_data):
            
            inp = inp.to(torch.float32)
            inp = inp.to(DEVICE)           
            for param in self.network.parameters(): 
                param.grad = None
                
            # network
            encoded = self.network.encoder(inp)
            latent_mu = self.network.encoder.backbone(inp)
            latent_logvar = self.network.encoder.backbone(inp)
            latent_logvar = torch.exp(0.5 * latent_logvar)            
            latent_dist = Normal(latent_mu, latent_logvar)
            # z = latent_dist.rsample([self.L])  # shape:[self.L,batch_size,1,latent_size]
            # z = z.view(-1, z.size(2), z.size(3))  # shape:[self.L*batch_size,1,latent_size]                    
            decoded = self.network.decoder(*encoded)
            recon_mu = decoded
            # recon_mu = recon_mu.view(self.L, *inp.shape)
            recon_logvar = decoded
            recon_logvar = torch.exp(0.5 * recon_logvar)
            # recon_logvar = recon_logvar.view(self.L, *inp.shape)
            # rec_z = self.network.reparameterize(recon_mu, recon_logvar)
            log_lik = Normal(recon_mu, recon_logvar).log_prob(inp).mean()
            kl = kl_divergence(latent_dist, self.prior).mean()
             
            self.rec_loss = abs(log_lik)
            self.kl_divergence_loss = kl
            self.loss = self.rec_loss + self.kl_divergence_loss
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
        self.L = 16
        self.prior = Normal(0, 1)
        loss = 0 
        with torch.no_grad():              
            for i, (inp, _)  in enumerate(self.test_data):
                
                inp = inp.to(torch.float32)
                inp = inp.to(DEVICE)              
                    
                ## network
                encoded = self.network.encoder(inp)
                latent_mu = self.network.encoder.backbone(inp)
                latent_logvar = self.network.encoder.backbone(inp)
                latent_logvar = torch.exp(0.5 * latent_logvar)            
                latent_dist = Normal(latent_mu, latent_logvar)
                # z = latent_dist.rsample([self.L])  # shape:[self.L,batch_size,1,latent_size]
                # zf_encoded, zp_encoded =  encoded # shape:[self.L*batch_size,1,latent_size]    
                # zf_encodedd =  latent_dist.rsample([self.L]) 
                # zp_encodedd =  latent_dist.rsample([self.L]) 
                # zp_encoded =  zf_encodedd.view(-1, zf_encoded.size(2), zf_encoded.size(3)) 
                # zp_encoded =  zp_encodedd.view(-1, zp_encoded.size(2), zp_encoded.size(3))                 
                decoded = self.network.decoder(*encoded)
                recon_mu = decoded
                # recon_mu = recon_mu.view(self.L, *inp.shape)
                recon_logvar = decoded
                recon_logvar = torch.exp(0.5 * recon_logvar)
                # recon_logvar = recon_logvar.view(self.L, *inp.shape)
                rec_z = self.network.encoder.reparameterize(recon_mu, recon_logvar)
                log_lik = Normal(recon_mu, recon_logvar).log_prob(inp).mean()
                kl = kl_divergence(latent_dist, self.prior).mean()

                self.rec_loss = abs(log_lik)        
                self.kl_divergence_loss = kl  
                self.loss = self.rec_loss + self.kl_divergence_loss
                loss+= self.loss      
                
                # get info of last batch
                if i >= len(self.test_data)-1:
                    reconstructions, _, _, info = self.network(inp, test = True)
                            
        self.test_loss = loss / len(self.test_data)
        
        self.data_storage.dump_store("Data", (inp, rec_z))
        self.data_storage.dump_store("Info", info)

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
        # self.best_state_dict = self.network.state_dict()
        # save best values but not model dict
        self.parameter_storage.store({k: self.best_values[k] for k in self.best_values.keys() - {'model_state_dict'}}, header="Best Values")
    # @tracer
    def evaluate(self):
        self.data_storage.store([self.epoch, self.batch, self.loss, self.kl_divergence_loss, 
                                 self.test_loss], force=self.batch)
        self._hook_every_epoch() 
        
        # # Pass the best_state_dict back to the MyOptimizer class
        # return self.best_state_dict        
        