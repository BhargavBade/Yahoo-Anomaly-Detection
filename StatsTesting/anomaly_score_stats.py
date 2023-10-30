import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report, average_precision_score

def anomaly_score_stats(ground_truth, final_preds, param_storage):
    
    ## Finding Accuracy Scores for the model        
    final_preds = np.array(final_preds)
    ground_truth = np.array(ground_truth)
    parameter_storage = param_storage
        
    roc_auc = roc_auc_score(ground_truth, final_preds)         
    print("AUROC Score is:", roc_auc)
    roc_str = "AUROC Score is: " + str(roc_auc)
                                 
    report = classification_report(ground_truth, final_preds, digits=4)
    print(report,'\n')
    
    #Calculatimg AUPRC
    auprc = average_precision_score(ground_truth, final_preds)
    print('AUPRC Score is', auprc)
    prc_str = "AUPRC Score is: " + str(auprc)        
    
    parameter_storage.write_tab("Classification Report", str(report))
    parameter_storage.write_tab("00", str(roc_str))
    parameter_storage.write_tab("00", str(prc_str))
    