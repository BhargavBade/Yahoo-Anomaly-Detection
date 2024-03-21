import torch
import numpy as np
import torch.nn as nn
import os
from ccbdl.utils import DEVICE
from ccbdl.storages import storages
from Data.prepare_data import prepare_data
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report, average_precision_score
import matplotlib.pyplot as plt
from sklearn import metrics
from Plotting.anomaly_plot import testdata_plotting
from torch.distributions import Normal

class VAEAnomalyDetection():
        
    def __init__(self, path, data_config: dict):    
                  
        self.path = path
        self.threshold = None
        self.pred_labels = None
        self.testt_dataa = None
        self.data_config = data_config
                                                      
        # get data
        train_data, test_data, val_data = prepare_data(self.data_config)

        self.train_data = train_data                     
        self.test_data = test_data
        self.val_data  = val_data
        
        # For storage of results and plots    
        self.parameter_storage = storages.ParameterStorage(path)
        self.parameter_storage.write("This file is automatically generated by ccbdl.learning.base.BaseLearning")
        self.figure_storage = storages.FigureStorage(path, dpi=300, types=("png", "pdf"))
        
        # Path to retrive the best model
        self.path_best = os.path.join(self.path,"best_state.pt")
        self.path_bestmodel =  os.path.join(self.path,"best_model.pt")

#----------------------------------------------------------------------------------------------                        
    def find_threshold(self):        
        
        print("\n******** Finding threshold reconstruction error of the Data ********\n")                                

        # Loading the saved best model along with hidden size and weights
        self.model = torch.load(self.path_bestmodel)
        self.model.eval() 
        
        self.L = 16              
        val_data = []
        val_labels = []
        valdata_rec = []    
        rec_latent_probabilities = [] 
 
        # Assuming self.test_data is your test data
        with torch.no_grad():
            for _, (inp, labels) in enumerate(self.test_data):
                                
                inp = inp.to(torch.float32)
                inp = inp.to(DEVICE)
                test_reconstructions = self.model(inp)
                
                #network
                enc = self.model.encoder(inp)
                latent_mu = self.model.en_mu(enc)                          
                latent_logvar = self.model.en_logvar(enc)                    
                z = self.model.reparameterize(latent_mu, latent_logvar)
                decoded = self.model.decoder(z)                  
                recon_mu = self.model.de_mu(decoded)
                recon_logvar = self.model.de_logvar(decoded)                              
                recon_stddev = torch.exp(0.5 * recon_logvar)
                recon_dist = Normal(recon_mu, recon_stddev)

                rec_latent_prob_density = recon_dist.log_prob(inp).exp()
                rec_latent_probabilities.append(rec_latent_prob_density)
                                
                val_data.append(inp) 
                val_labels.append(labels)
                valdata_rec.append(test_reconstructions) 
            
            v_concatenated_prob = torch.cat(rec_latent_probabilities, dim=0).to(DEVICE)           
            v_prob_array = v_concatenated_prob.cpu().numpy() 
            self.val_labels = torch.cat(val_labels)
            
            # Finding the best possible Threshold Value for Anomaly Detection              
            best_f1_score = 0.0
            best_threshold = 0.0
            
            # Iterate over different values of y
            start = 0.001
            end = 0.3
            step = 0.001
            
            probability = start
            while probability <= end:
                # print(current_value)
                threshold = float(probability)
                val_anomaly = (v_concatenated_prob < threshold).to(DEVICE)
                v_pred_labels_tensor = torch.where(val_anomaly, torch.tensor(1).to(DEVICE), torch.tensor(0).to(DEVICE))
                v_ground_truth_tensor_1d = self.val_labels.view(-1)
                v_preds_tensor_1d = v_pred_labels_tensor.view(-1)
                v_ground_truth = v_ground_truth_tensor_1d.cpu().numpy()
                v_final_preds = v_preds_tensor_1d.cpu().numpy()
                val_f1_score = metrics.f1_score(v_ground_truth, v_final_preds)               
                probability += step
                
                if val_f1_score > best_f1_score:
                    best_f1_score = val_f1_score
                    best_threshold = threshold

            self.threshold = best_threshold
            print("Best F1 Score from val dataset:", best_f1_score, '\n')
            print("Best Threshold probability from Val Data:", best_threshold, '\n')
            threshold_txt = "Best Threshold probability from Val Data: " + str(self.threshold)
            self.parameter_storage.write_tab("00", str(threshold_txt))
            
        return best_threshold
    
#------------------------------------------------------------------------------------------------- 

#Testing Phase
 
    def find_anomalies(self):
        
        self.model = torch.load(self.path_bestmodel)
        self.model.eval()   
        
        self.L = 16              
        test_data = []
        test_labels = []
        testdata_rec = []                                          
        rec_latent_probabilities = [] 
        with torch.no_grad():
            for _, (inp, labels) in enumerate(self.test_data):
                                
                inp = inp.to(torch.float32)
                inp = inp.to(DEVICE)
                test_reconstructions = self.model(inp)      
                
                # network
                enc = self.model.encoder(inp)
                latent_mu = self.model.en_mu(enc)                          
                latent_logvar = self.model.en_logvar(enc)                   
                z = self.model.reparameterize(latent_mu, latent_logvar)
                decoded = self.model.decoder(z)                  
                recon_mu = self.model.de_mu(decoded)
                recon_logvar = self.model.de_logvar(decoded)                              
                recon_stddev = torch.exp(0.5 * recon_logvar)
                recon_dist = Normal(recon_mu, recon_stddev)
 
                rec_latent_prob_density = recon_dist.log_prob(inp).exp()
                rec_latent_probabilities.append(rec_latent_prob_density)
                
                test_data.append(inp) 
                test_labels.append(labels)
                testdata_rec.append(test_reconstructions) 
            
            self.testdata_rec = torch.cat(testdata_rec, dim = 0).to(DEVICE)    
            t_concatenated_prob = torch.cat(rec_latent_probabilities, dim=0).to(DEVICE)   
                    
            v_prob_array = t_concatenated_prob.cpu().numpy()                                
            self.test_data_tensor = torch.cat(test_data, dim = 0)
            self.test_labels = torch.cat(test_labels)                     
                       
            threshold = self.threshold                             
            anomaly = (t_concatenated_prob < threshold).to(DEVICE)    
            # 1 = anomaly, 0 = normal            
            pred_labels_tensor = torch.where(anomaly, torch.tensor(1).to(DEVICE), torch.tensor(0).to(DEVICE))                         
            self.pred_labels = pred_labels_tensor
                       
        # ---------------------------------------------------------------------------------        
        #Evaluating the test data based on some metrics
        
            actual_labels_tensor = self.test_labels
            
            # Reshape the original labels tensor into a 1D array
            ground_truth_tensor_1d = actual_labels_tensor.view(-1)
            
            # Count the number of zeros and ones
            normal_count = torch.sum(ground_truth_tensor_1d == 0)
            print("no of actual normal data points are:", normal_count, '\n')
            true_anomaly_count = torch.sum(ground_truth_tensor_1d == 1)                                
            print("no of actual anomaly data points are:", true_anomaly_count,'\n' ) 
            true_anm_txt = "no of actual anomaly data points are:" + str(true_anomaly_count)
            
            ground_truth = ground_truth_tensor_1d.cpu().numpy()
                        
            # Reshape the predictions label tensor into a 1D array
            preds_tensor_1d = pred_labels_tensor.view(-1)
            
            normal_count = torch.sum(preds_tensor_1d == 0)
            print("no of predicited normal data points are:", normal_count, '\n') 
            pred_anomaly_count = torch.sum(preds_tensor_1d == 1)                                           
            print("no of predicted anomaly data points are:", pred_anomaly_count ,'\n' ) 
            pred_anm_txt = "no of predicted anomaly data points are:" + str(pred_anomaly_count)
            
            final_preds = preds_tensor_1d.cpu().numpy()
            
            roc_auc = roc_auc_score(ground_truth, final_preds)         
            print("AUROC Score is:", roc_auc)
            roc_str = "AUROC Score is: " + str(roc_auc)
                       
            fpr, tpr, thresholds = metrics.roc_curve(ground_truth, final_preds)          
            fig = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='roc_auc_curve')
            fig.plot()
            plt.show()                   
            report = classification_report(ground_truth, final_preds, digits=4)
            print(report,'\n')

            #Calculatimg AUPRC
            auprc = average_precision_score(ground_truth, final_preds)
            print('AUPRC Score is', auprc)
            prc_str = "AUPRC Score is: " + str(auprc)        
            
            self.parameter_storage.write_tab("Classification Report", str(report))
            self.parameter_storage.write_tab("00", str(true_anm_txt))
            self.parameter_storage.write_tab("00", str(pred_anm_txt))
            self.parameter_storage.write_tab("00", str(roc_str))
            self.parameter_storage.write_tab("00", str(prc_str))
            
            
            # Plotting the data and visualizing anomalies
            testdata_plotting(path = self.path,
                              test_data_tensor = self.test_data_tensor,   
                              testdata_rec = self.testdata_rec, 
                              test_labels = self.test_labels, 
                              pred_labels =self.pred_labels, 
                              figure_storage = self.figure_storage)
                        
        return actual_labels_tensor, pred_labels_tensor
