import torch
from Data.prepare_data import prepare_data
from ccbdl.utils import DEVICE

class BaseAnomalyStats():
 # def __init__(self, path, network):
  def __init__(self, data_config: dict):    
               
     self.data_config = data_config

     # get data
     self.train_data, self.test_data, self.val_data = prepare_data(self.data_config)
     
     # Train Data
     train_tensor = []
     train_lab_tensor = []
     for _, (inp, labels) in enumerate(self.train_data):                           
         # get data
         inp = inp.to(torch.float32)
         inp = inp.to(DEVICE)                        
         train_tensor.append(inp)
         train_lab_tensor.append(labels)
     
     concatenated_tensor1 = torch.cat(train_tensor, dim=0).to(DEVICE)
     concatenated_tensor2 = torch.cat(train_lab_tensor, dim=0).to(DEVICE)
     self.train_array = concatenated_tensor1.reshape(-1).cpu().numpy()
     self.train_lab_array = concatenated_tensor2.reshape(-1).cpu().numpy()                              
     
     # Val Data
     val_tensor = []
     val_lab_tensor = []
     for _, (inp, labels) in enumerate(self.val_data):                           
         # get data
         inp = inp.to(torch.float32)
         inp = inp.to(DEVICE)                        
         val_tensor.append(inp)
         val_lab_tensor.append(labels)
     
     concatenated_tensor1 = torch.cat(val_tensor, dim=0).to(DEVICE)
     concatenated_tensor2 = torch.cat(val_lab_tensor, dim=0).to(DEVICE)
     self.val_array = concatenated_tensor1.reshape(-1).cpu().numpy()
     self.val_lab_array = concatenated_tensor2.reshape(-1).cpu().numpy()   
     
     # Test Data
     test_tensor = []
     test_lab_tensor = []
     for _, (inp, labels) in enumerate(self.test_data):                           
         # get data
         inp = inp.to(torch.float32)
         inp = inp.to(DEVICE)                        
         test_tensor.append(inp)
         test_lab_tensor.append(labels)
     
     concatenated_tensor1 = torch.cat(test_tensor, dim=0).to(DEVICE)
     concatenated_tensor2 = torch.cat(test_lab_tensor, dim=0).to(DEVICE)
     self.test_array = concatenated_tensor1.reshape(-1).cpu().numpy()
     self.test_lab_array = concatenated_tensor2.reshape(-1).cpu().numpy()
     
     return self.train_array, self.train_lab_array, self.val_array, self.val_lab_array, self.test_array, self.test_lab_array
                                                                             