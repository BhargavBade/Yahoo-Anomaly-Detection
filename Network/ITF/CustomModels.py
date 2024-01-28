import torch
import torch as nn
from ccbdl.utils import DEVICE

#%% Pass Input
class Pass(torch.nn.Module):
    def __init__(self):
        super(Pass, self).__init__()
    def forward(self,x):
        return x
    def __repr__(self):
        return "{}".format(self.__class__.__name__)

#%% Top K
class GetTopK(torch.nn.Module):
    def __init__(self, k):
        super(GetTopK, self).__init__()
        self.k=k
    def forward(self, x):
        # top K
        values, indices = torch.topk(x, self.k, dim=-1, largest=True, sorted=False)
        # set rest to zero
        res = torch.zeros(x.shape,device=DEVICE,requires_grad=True)
        res = res.scatter(-1, indices, values)
        return res
    def __repr__(self):
        return "{}(k={})".format(self.__class__.__name__, self.k)

#%% Remove 0 zfs
class Remove0s(torch.nn.Module):
    def __init__(self,definition):
        super(Remove0s, self).__init__()
        self.definition = definition
        self.param_lengths = [x[1] for x in definition]
        
    def forward(self, zf, zd):
        zero_functions = torch.repeat_interleave(zd == 0, torch.tensor(self.param_lengths,device=DEVICE), dim=-1)
        indices = (zero_functions == True).nonzero(as_tuple=True)
        zf.data[indices]=torch.zeros(1,device=DEVICE)
        return zf
    
#%% Reshape
class Reshape(torch.nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)  
    
    def __repr__(self):
        return "{}(Shape={})".format(self.__class__.__name__, self.shape)

#%% Flatten to 2D
class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)
    def __repr__(self):
        return "{}(Shape=bs,-1)".format(self.__class__.__name__)
        
#%% CustomHardlim
class CustomHardlim(torch.nn.Module):
    def __init__(self,threshold):
        super(CustomHardlim, self).__init__()
        self.customHardlim = Hardlim.apply 
        self.threshold = threshold
    def forward(self, x):
        x = self.customHardlim(x,self.threshold)
        return x
    def __repr__(self):
        return "{}(Threshold={})".format(self.__class__.__name__, self.threshold)
    
#%%Hardlim
class Hardlim(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, const):
        x = (x>const).float()
        return x
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None
    
    
    
class CustomWhere():
    def __init__(self,a1,a2):
        self.a1=a1
        self.a2=a2
        self.customWhere = MyWhere.apply
    def __call__(self, x):
        x = self.customWhere(x,self.a1,self.a2)
        return x

    
    
class MyWhere(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a1, a2):
        x = torch.where((x < a2), torch.zeros(1,device=DEVICE), a1)
        return x
    @staticmethod
    def backward(ctx, grad_output):
        # print("123")
        return None, grad_output, grad_output
        
        
        
class Ones(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dump = nn.Sequential(nn.Linear(1,1))    
    def forward(self, data):
        return torch.ones(data.shape, device=DEVICE)