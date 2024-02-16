import optuna
import datetime
import os
from Data.prepare_data import prepare_data
# from Network.auto_encoder_lin import MyAutoEncoderLin
# from Learning.learn_bofautoencoder import LearnAutoEncoder
from Learning.learn_bofvarautoencoder import LearnVarAutoEncoder
from ccbdl.parameter_optimizer.optuna_base import BaseOptunaParamOptimizer
from Plotting.latentspace_plotting import visualize_latent_space
from ccbdl.utils import DEVICE
from ccbdl.storages import storages
from ccbdl import NBPATH
from ccbdl.evaluation.additional import notebook_handler
import torch
import time
import Network
from Testing.lae_anm_detc import LAEAnomalyDetection
from Testing.vae_anm_detc import VAEAnomalyDetection

class MyOptimizer(BaseOptunaParamOptimizer):
    def __init__(self,
                 study_config: dict,
                 optimize_config: dict,
                 network_config: dict,
                 data_config: dict,
                 learner_config: dict,
                 study_path: str,
                 comment: str = "",
                 config_path: str = "",
                 debug: bool = False,
                 logging: bool = False):
        # get sampler and pruner for parent class
        if "sampler" in study_config.keys():
            if hasattr(optuna.samplers, study_config["sampler"]["name"]):
                sampler = getattr(
                    optuna.samplers, study_config["sampler"]["name"])()
        else:
            sampler = optuna.samplers.TPESampler()

        if "pruner" in study_config.keys():
            if hasattr(optuna.pruners, study_config["pruner"]["name"]):
                pruner = getattr(
                    optuna.pruners, study_config["pruner"]["name"])()
        else:
            pruner = None

        super().__init__(study_config["direction"], study_config["study_name"], study_path,
                         study_config["number_of_trials"], data_config["task"], comment, study_config["optimization_target"], sampler, pruner, config_path, debug, logging)

        self.optimize_config = optimize_config
        self.network_config = network_config
        self.data_config = data_config
        self.learner_config = learner_config
        self.study_config = study_config
        self.best_state_dict = None
        self.best_model = None
        self.parameter_storage = storages.ParameterStorage(study_path)

        optuna.logging.disable_default_handler()
        self.create_study()

        
        
    def create_study(self):
        self.study = optuna.create_study(
            direction=self.direction, sampler=self.sampler, study_name=self.study_name)

    def _objective(self, trial):
        print("\n\n\n******* Start Trial: " + str(trial.number+1) +
              "/" + str(self.number_of_trials)+" *******")
        # folder creation
        trial_start = time.strftime('%Y-%m-%d__%H-%M-%S')
        trial_path = os.path.join(self.study_path, str(
            trial.number).zfill(2) + "_" + trial_start)

        # suggest parameters
        suggested = self._suggest_parameters(self.optimize_config, trial)
        self.learner_config["learning_rate"] = suggested["learning_rate"]
        self.network_config["stages"] = suggested["stages"]

        print("### GET DATA ####")
        train_data, test_data, val_data = prepare_data(self.data_config)


        # AutoEncoder
        network = getattr(Network,self.network_config["name"] )(self.network_config).to(DEVICE)

        print("\n\n******* Start Train AutoEncoder *******")
        # self.learner = LearnAutoEncoder(trial_path,
        self.learner = LearnVarAutoEncoder(trial_path,                                
                                            trial,
                                            network,
                                            train_data,
                                            test_data,
                                            val_data,
                                            self.learner_config,
                                            task=self.task,
                                            logging=self.logging)
        
        self.learner.parameter_storage.store(suggested, header = "suggested_parameters")
        self.learner.fit(test_epoch_step=self.learner_config["testevery"])
        print("\n******* Train AutoEncoder Done *******")
        
        #---------------------------------------------------------------------------------- 
        self.store_config(self.data_config, self.optimize_config,
                          self.study_config, self.network_config,
                          self.learner_config, trial_path)
        
        save_path = os.path.join(trial_path, "best_state.pt")
        torch.save({'epoch': self.learner.best_values["Epoch"],
                    'test_loss': self.learner.best_values["TestLoss"],
                    'model_state_dict': self.best_state_dict},
                     save_path)
    
        save_model_path = os.path.join(trial_path, "best_model.pt")
        torch.save(network, save_model_path)
    
        self.parameter_storage.write_tab("Best Network", str(self.best_model))
        
        # anomaly_detector = LAEAnomalyDetection(trial_path, self.data_config)
        anomaly_detector = VAEAnomalyDetection(trial_path, self.data_config)
        anomaly_detector.find_threshold()   
        anomaly_detector.find_anomalies()
        #----------------------------------------------------------------------------------- 
        
        return self.learner.best_values[self.optimization_target]

    def start_study(self):
        print("******* Start " + os.path.basename(__file__) +
              ": " + self.study_name + " *******")
        self.study.optimize(self._objective,
                            n_trials=self.number_of_trials,
                            callbacks=[self.trial_end_callback])
        duration = sum((t.duration for t in self.study.trials),
                       datetime.timedelta())
        return duration

    def eval_study(self):
       
        eval_jnb = 'study_eval.ipynb'
        notebook_handler(os.path.join(NBPATH, eval_jnb),
                         execute=True, result_dir=self.study_path, direction=self.study_config["direction"]).save_as_html(self.study_path, self.study_name)
