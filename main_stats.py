""" 
This script trains an Mnist AutoEncoder using the ccb_dl_package
"""
# append parent folder sys.path for imports
import ccbdl
import os
import sys
import inspect
from StatsTesting.arma_anm_detc import AnomalyDetection_ARMA
from StatsTesting.isofor_anm_detc import AnomalyDetection_ISOFOR
from StatsTesting.ocsvm_anm_detc import AnomalyDetection_OCSVM

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(os.path.dirname(currentdir)))
sys.path.insert(0, parentdir)

if __name__ == '__main__':

    # Get Config
    config_loader = ccbdl.config_loader.loaders.ConfigurationLoader()
    config_path = os.path.join(os.getcwd(), "Configuration", "config_stats.yaml")
    config = config_loader.read_config(config_path)

    # Get Configurations
    data_config = config["Data"]
    study_config = config["Study"]

    # Study Config
    study_path = ccbdl.storages.storages.generate_train_folder(name="__" + study_config["study_name"],
                                                               generate=True,
                                                               location=os.path.dirname(os.path.realpath(__file__)))
    
    study = study_config["study_name"]
    
    
    if "isofor" in study.lower():
        rec_error = AnomalyDetection_ISOFOR(study_path, study_config, data_config)
        
    elif "arma" in study.lower():
        rec_error = AnomalyDetection_ARMA(study_path, study_config, data_config)
        
    elif "ocsvm" in study.lower():  
        rec_error = AnomalyDetection_OCSVM(study_path, study_config, data_config)
        
    else:
        print("study not found")

    # Testing
    # Finding threshold reconstruction error and anomalies
    rec_error.learning()
    rec_error.find_threshold()   
    rec_error.find_anomalies()
