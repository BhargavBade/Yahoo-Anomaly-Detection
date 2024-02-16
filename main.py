""" 
This script trains an Mnist AutoEncoder using the ccb_dl_package
"""
# append parent folder sys.path for imports
from ParamOptimizer.optuna_param_optimizer_bof import MyOptimizer
# from ParamOptimizer.optuna_param_optimizer import MyOptimizer
import ccbdl
import os
import sys
import inspect
from Network.ITF import functions as f
from Testing.lae_anm_detc import LAEAnomalyDetection
from Testing.vae_anm_detc import VAEAnomalyDetection

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(os.path.dirname(currentdir)))
sys.path.insert(0, parentdir)

if __name__ == '__main__':

    # Get Config
    config_loader = ccbdl.config_loader.loaders.ConfigurationLoader()
    # config_path = os.path.join(os.getcwd(), "Configuration", "config_sinewaves.yaml")
    config_path = os.path.join(os.getcwd(), "Configuration", "config_sinewaves_bof.yaml")
    config = config_loader.read_config(config_path)


    # Get Configurations
    network_config = config["Network"]
    optimizer_config = config["Optimized"]
    data_config = config["Data"]
    learner_config = config["Learning"]
    study_config = config["Study"]

    # Study Config
    study_path = ccbdl.storages.storages.generate_train_folder(name="__" + study_config["study_name"],
                                                               generate=True,
                                                               location=os.path.dirname(os.path.realpath(__file__)))
    # Define Parameter Optimizer
    opti = MyOptimizer(study_config,
                       optimizer_config,
                       network_config,
                       data_config,
                       learner_config,
                       study_path,
                       comment="Study for Testing",
                       config_path = config_path,
                       debug=False,
                       logging=True)
    
    study = study_config["study_name"]
    
    if "lae" in study.lower():   
        anomaly_detector = LAEAnomalyDetection(study_path, data_config)
    
    elif "var" in study.lower():
        anomaly_detector = VAEAnomalyDetection(study_path, data_config)
        
    elif "bof" in study.lower():
        # additional stuff, definition of possible functions in the bag
        network_config["function_pool"] = [
                                            # (f.Gaus, 3, True), 
                                            (f.Cos, 3, True), 
                                            (f.Sin, 3, True), 
                                            (f.Lin, 2, True), 
                                            (f.Exp_Sat, 3, True)]
        # anomaly_detector = LAEAnomalyDetection(study_path, data_config)
    
    else:
        print("study not found")     

    # Run Parameter Optimizer
    opti.start_study()

    # Summarize Study
    opti.summarize_study()

    # Compare Runs
    opti.eval_study()
    
    # # Testing
    # # Finding threshold and anomalies
    # anomaly_detector.find_threshold()   
    # anomaly_detector.find_anomalies()