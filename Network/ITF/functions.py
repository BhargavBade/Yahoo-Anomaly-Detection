import torch
import inspect
import matplotlib.pyplot as plt
import math as m
from ccbdl.utils import DEVICE
from .CustomModels import CustomWhere

#%% base class for all functions
class Function(object):
    def __init__(self, func, *params):
        self.func = func
        self.parameters = params
        self.len_params = len(params)
        
    def __call__(self,x):
        return self.func(x)
        
    def __str__(self,value=(None,None)):
        func_str = inspect.getsourcelines(self.func)[0][0][25:]
        func_str = func_str.replace("torch.", "")
        func_str = func_str.replace("\n","")
        return self._replace_params(func_str,value)
    
    def _replace_params(self,func_str,value=(None,None)):    
        if not any(map(lambda x: x is None, value)):
            for param in reversed(range(self.len_params)):
                func_str = func_str.replace("a"+str(param+1), "{:.4f}".format(self.parameters[param][value].item()))
        return func_str
    
    def get_plot(self,x,value=(0,0)):
        fig=plt.figure(figsize=(5,5))
        plt.plot(x.cpu(),self.__call__(x)[value].cpu())
        plt.title(self.__str__(value))
        name = "Batch_" + str(value[0]) + " CH_" + str(value[1]) 
        return fig, name
    
    def show(self,x,value=(0,0)):
        plot, name = self.get_plot(x,value)
        plt.show()



#%% Const    
class Const(Function):
    def __init__(self,a1,rescale=False,TMAX=100):
        if rescale:
            inp_min, inp_max=0,1
            self.a1_min, self.a1_max = (-1, 1)
            a1 = self.a1_min+(self.a1_max-self.a1_min)*(a1-inp_min)/(inp_max-inp_min)   # offset
        func = lambda x: a1.repeat(1,1,x.shape[-1])
        super().__init__(func, a1)        
    def __str__(self,value=(None,None)):
        func_str = super().__str__(value)
        return func_str.replace(".repeat(1,1,x.shape[-1])","")


#%% Periodic
class Sin(Function):
    def __init__(self,a1,a2,a3,rescale=False,TMAX=100):
        if rescale:
            inp_min, inp_max = (0,1)
            self.a1_min, self.a1_max = (0, 1)                               # amplitude
            self.a2_min, self.a2_max = (0.2*(2*m.pi)/TMAX, 10*(2*m.pi)/TMAX) # frequency number of oscillations (cycles) that occur each second of time.
            self.a3_min, self.a3_max = (0, 2*m.pi)                          # phase
            a1 = self.a1_min+(self.a1_max-self.a1_min)*(a1-inp_min)/(inp_max-inp_min)
            a2 = self.a2_min+(self.a2_max-self.a2_min)*(a2-inp_min)/(inp_max-inp_min)
            a3 = self.a3_min+(self.a3_max-self.a3_min)*(a3-inp_min)/(inp_max-inp_min)
        func = lambda x: a1*torch.sin(a2*x+a3)
        super().__init__(func, a1, a2, a3)


class Cos(Function):
    def __init__(self,a1,a2,a3,rescale=False,TMAX=100):
        if rescale:
            inp_min, inp_max = (0,1)
            self.a1_min, self.a1_max = (-1, 1)                               # amplitude
            self.a2_min, self.a2_max = (0.2*(2*m.pi)/TMAX, 10*(2*m.pi)/TMAX) # frequency number of oscillations (cycles) that occur each second of time.
            self.a3_min, self.a3_max = (0, 2*m.pi)                          # phase
            a1 = self.a1_min+(self.a1_max-self.a1_min)*(a1-inp_min)/(inp_max-inp_min)
            a2 = self.a2_min+(self.a2_max-self.a2_min)*(a2-inp_min)/(inp_max-inp_min)
            a3 = self.a3_min+(self.a3_max-self.a3_min)*(a3-inp_min)/(inp_max-inp_min)
        func = lambda x: a1*torch.cos(a2*x+a3)
        super().__init__(func, a1, a2, a3)




class Sin_Abs(Function):
    def __init__(self,a1,a2,a3,rescale=False,TMAX=100):
        if rescale:
            inp_min, inp_max = (0,1)
            self.a1_min, self.a1_max = (0, 1)                               # amplitude
            self.a2_min, self.a2_max = (0.1*(2*m.pi)/TMAX, 10*(2*m.pi)/TMAX) # frequency number of oscillations (cycles) that occur each second of time.
            self.a3_min, self.a3_max = (0, 2*m.pi)                          # phase
            a1 = self.a1_min+(self.a1_max-self.a1_min)*(a1-inp_min)/(inp_max-inp_min)
            a2 = self.a2_min+(self.a2_max-self.a2_min)*(a2-inp_min)/(inp_max-inp_min)
            a3 = self.a3_min+(self.a3_max-self.a3_min)*(a3-inp_min)/(inp_max-inp_min)
        func = lambda x: torch.abs(a1*torch.sin(a2*x+a3))
        super().__init__(func, a1, a2, a3)

class Square_Wave(Function):
    def __init__(self,a1,a2,a3,rescale=False,TMAX=100):
        if rescale:
            inp_min, inp_max = (0,1)
            self.a1_min, self.a1_max = (0, 1)                               # amplitude
            self.a2_min, self.a2_max = (0.2*(2*m.pi)/TMAX, 5*(2*m.pi)/TMAX) # frequency number of oscillations (cycles) that occur each second of time.
            self.a3_min, self.a3_max = (0, 2*m.pi)                          # phase
            a1 = self.a1_min+(self.a1_max-self.a1_min)*(a1-inp_min)/(inp_max-inp_min)
            a2 = self.a2_min+(self.a2_max-self.a2_min)*(a2-inp_min)/(inp_max-inp_min)
            a3 = self.a3_min+(self.a3_max-self.a3_min)*(a3-inp_min)/(inp_max-inp_min)
        func = lambda x: a1*torch.sin(a2*x+a3)
        super().__init__(func, a1, a2, a3)

class Triangle_Wave(Function):
    def __init__(self,a1,a2,a3,rescale=False,TMAX=100):
        if rescale:
            inp_min, inp_max = (0,1)
            self.a1_min, self.a1_max = (0, 1)                               # amplitude
            self.a2_min, self.a2_max = (0.2*(2*m.pi)/TMAX, 5*(2*m.pi)/TMAX) # frequency number of oscillations (cycles) that occur each second of time.
            self.a3_min, self.a3_max = (0, 2*m.pi)                          # phase
            a1 = self.a1_min+(self.a1_max-self.a1_min)*(a1-inp_min)/(inp_max-inp_min)
            a2 = self.a2_min+(self.a2_max-self.a2_min)*(a2-inp_min)/(inp_max-inp_min)
            a3 = self.a3_min+(self.a3_max-self.a3_min)*(a3-inp_min)/(inp_max-inp_min)
        func = lambda x: a1*torch.sin(a2*x+a3)
        super().__init__(func, a1, a2, a3)
        
class Sawtooth_Wave(Function):
    def __init__(self,a1,a2,a3,rescale=False,TMAX=100):
        if rescale:
            inp_min, inp_max = (0,1)
            self.a1_min, self.a1_max = (0, 2)                # amplitude 
            self.a2_min, self.a2_max = (1*TMAX, (1/10)*TMAX) # frequency 
            self.a3_min, self.a3_max = (0, 1)                # offset
            a1 = self.a1_min+(self.a1_max-self.a1_min)*(a1-inp_min)/(inp_max-inp_min)
            a2 = self.a2_min+(self.a2_max-self.a2_min)*(a2-inp_min)/(inp_max-inp_min)
            a3 = self.a3_min+(self.a3_max-self.a3_min)*(a3-inp_min)/(inp_max-inp_min)
        func = lambda x: 2*a1*((x/a2)-torch.floor(0.5+(x/a2)))+a3
        super().__init__(func, a1, a2, a3)
        
class Cycloid(Function):
    def __init__(self,a1,a2,a3,rescale=False,TMAX=100):
        super().__init__(lambda x: a1+a2+a3, a1, a2, a3)
        raise NotImplementedError ("Funktion : " + type(self).__name__ +  " not implemented")
      
class si(object):
    def __init__(self,a1,a2,a3):
        self.a1=a1
        self.a2=a2
        self.a3=a3
    def __call__(self,x):
        erg =  (torch.sin(self.a2*x+self.a3)/(x*self.a2+self.a3))  
        # remove nands to 1
        erg[erg != erg] = 1
        if torch.any(erg != erg):
            print("Problem in SI detected")
        erg *= self.a1 
        return erg
    def __str__(self,value=(None,None)): 
        func_str = "a1*(sin(a2*x+a3)/(x*a2+a3))"        
        return func_str  
     
class Si(Function):
    def __init__(self,a1,a2,a3, rescale=False,TMAX=100):
        if rescale:
            inp_min, inp_max = (0,1)
            self.a1_min, self.a1_max = (-1, 1)                          # amplitude at 0
            self.a2_min, self.a2_max = (0, 5*(2*m.pi)/TMAX )            # frequency
            self.a3_min, self.a3_max = (-6*(2*m.pi), 1*(2*m.pi))        # phase
            a1 = self.a1_min+(self.a1_max-self.a1_min)*(a1-inp_min)/(inp_max-inp_min)
            a2 = self.a2_min+(self.a2_max-self.a2_min)*(a2-inp_min)/(inp_max-inp_min)
            a3 = self.a3_min+(self.a3_max-self.a3_min)*(a3-inp_min)/(inp_max-inp_min) 
        func = si(a1, a2, a3)
        super().__init__(func, a1, a2, a3) 
        
    def __str__(self,value=(None,None)):
        func_str = self.func.__str__(value)
        return self._replace_params(func_str,value)
          
#%% Trend   
class Lin(Function):
    def __init__(self,a1,a2, rescale=False,TMAX=100):
        if rescale:
            inp_min, inp_max = (0,1)
            self.a1_min, self.a1_max = (-1, 1)              # offset
            self.a2_min, self.a2_max = (-2/TMAX, 2/TMAX)    # slope
            a1 = self.a1_min+(self.a1_max-self.a1_min)*(a1-inp_min)/(inp_max-inp_min)
            a2 = self.a2_min+(self.a2_max-self.a2_min)*(a2-inp_min)/(inp_max-inp_min)
        func = lambda x: a1+a2*x 
        super().__init__(func, a1, a2)

class Exp_Sat(Function):
    def __init__(self,a1,a2,a3, rescale=False,TMAX=100):
        if rescale:
            inp_min, inp_max = (0,1)
            self.a1_min, self.a1_max = (-1, 1)          # offset
            self.a2_min, self.a2_max = (0, 1)           # start height
            self.a3_min, self.a3_max = (0, TMAX*0.333)  # tau
            a1 = self.a1_min+(self.a1_max-self.a1_min)*(a1-inp_min)/(inp_max-inp_min)
            a2 = self.a2_min+(self.a2_max-self.a2_min)*(a2-inp_min)/(inp_max-inp_min)
            a3 = self.a3_min+(self.a3_max-self.a3_min)*(a3-inp_min)/(inp_max-inp_min) 
            e = 1e-5*TMAX
        func = lambda x: a1+a2*(1-torch.exp(-x/(a3+e)))
        super().__init__(func, a1, a2, a3)
        
class Exp_Decay(Function):
    def __init__(self,a1,a2,a3, rescale=False,TMAX=100):
        if rescale:
            inp_min, inp_max = (0,1)
            self.a1_min, self.a1_max = (-1, 1)          # offset
            self.a2_min, self.a2_max = (0, 1)           # start height
            self.a3_min, self.a3_max = (0, TMAX*0.333)  # tau
            a1 = self.a1_min+(self.a1_max-self.a1_min)*(a1-inp_min)/(inp_max-inp_min)
            a2 = self.a2_min+(self.a2_max-self.a2_min)*(a2-inp_min)/(inp_max-inp_min)
            a3 = self.a3_min+(self.a3_max-self.a3_min)*(a3-inp_min)/(inp_max-inp_min) 
            e = 1e-5*TMAX
        func = lambda x: a1+a2*(torch.exp(-x/(a3+e)))
        super().__init__(func, a1, a2, a3)

class Exp_Mixed(Function):
    def __init__(self,a1,a2,a3, rescale=False,TMAX=100):
        if rescale:
            inp_min, inp_max = (0,1)
            self.a1_min, self.a1_max = (-1, 1)          # offset
            self.a2_min, self.a2_max = (-1, 1)           # end height
            self.a3_min, self.a3_max = (0, TMAX*0.333)  # tau
            a1 = self.a1_min+(self.a1_max-self.a1_min)*(a1-inp_min)/(inp_max-inp_min)
            a2 = self.a2_min+(self.a2_max-self.a2_min)*(a2-inp_min)/(inp_max-inp_min)
            a3 = self.a3_min+(self.a3_max-self.a3_min)*(a3-inp_min)/(inp_max-inp_min) 
            e = 1e-5*TMAX
        func = lambda x: a1+a2*(1-torch.exp(-x/(a3+e)))
        super().__init__(func, a1, a2, a3)
  
class Sqrt(Function):
    def __init__(self,a1,a2,a3, rescale=False,TMAX=100):
        if rescale:
            inp_min, inp_max = (0,1)
            self.a1_min, self.a1_max = (-1, 1)                              # offset
            self.a2_min, self.a2_max = (-2/m.sqrt(TMAX), 2/m.sqrt(TMAX))    # scale
            self.a3_min, self.a3_max = (0, TMAX/4 )                         # phase
            a1 = self.a1_min+(self.a1_max-self.a1_min)*(a1-inp_min)/(inp_max-inp_min)
            a2 = self.a2_min+(self.a2_max-self.a2_min)*(a2-inp_min)/(inp_max-inp_min)
            a3 = self.a3_min+(self.a3_max-self.a3_min)*(a3-inp_min)/(inp_max-inp_min) 
        func = lambda x: a1+a2*(torch.sqrt(x+a3))
        super().__init__(func, a1, a2, a3)  

class Log10(Function):
    def __init__(self,a1,a2,a3, rescale=False,TMAX=100):
        if rescale:
            inp_min, inp_max = (0,1)
            self.a1_min, self.a1_max = (-1, 1)            # offset
            self.a2_min, self.a2_max = (-2, 2)            # end value *2
            self.a3_min, self.a3_max = (0.01, TMAX/4)     # phase
            a1 = self.a1_min+(self.a1_max-self.a1_min)*(a1-inp_min)/(inp_max-inp_min)
            a2 = self.a2_min+(self.a2_max-self.a2_min)*(a2-inp_min)/(inp_max-inp_min)
            a3 = self.a3_min+(self.a3_max-self.a3_min)*(a3-inp_min)/(inp_max-inp_min) 
        func = lambda x: a1+ a2*torch.log10(x+a3)    
        super().__init__(func, a1, a2, a3)         
             
class Tanh(Function):
    def __init__(self,a1,a2,a3, rescale=False,TMAX=100):
        if rescale:
            inp_min, inp_max = (0,1)
            self.a1_min, self.a1_max = (0, 1)                   # amplitude
            self.a2_min, self.a2_max = (-50/TMAX, 50/TMAX)      # slope
            self.a3_min, self.a3_max = (0, TMAX)                # phase
            a1 = self.a1_min+(self.a1_max-self.a1_min)*(a1-inp_min)/(inp_max-inp_min)
            a2 = self.a2_min+(self.a2_max-self.a2_min)*(a2-inp_min)/(inp_max-inp_min)
            a3 = self.a3_min+(self.a3_max-self.a3_min)*(a3-inp_min)/(inp_max-inp_min) 
        func = lambda x: a1*torch.tanh((x-a3)*a2)  
        super().__init__(func, a1, a2, a3)         
    
class Sig(Function):
    def __init__(self,a1,a2,a3, rescale=False,TMAX=100):
        if rescale:
            inp_min, inp_max = (0,1)
            self.a1_min, self.a1_max = (0, 1)                    # amplitude
            self.a2_min, self.a2_max = (-50/TMAX, 50/TMAX)       # slope
            self.a3_min, self.a3_max = (-TMAX, 0)                # phase
            a1 = self.a1_min+(self.a1_max-self.a1_min)*(a1-inp_min)/(inp_max-inp_min)
            a2 = self.a2_min+(self.a2_max-self.a2_min)*(a2-inp_min)/(inp_max-inp_min)
            a3 = self.a3_min+(self.a3_max-self.a3_min)*(a3-inp_min)/(inp_max-inp_min) 
        func = lambda x: a1/(1+torch.exp((-x-a3)*a2)) 
        super().__init__(func, a1, a2, a3)        
 
 
#%% Event
class Gaus(Function):
    def __init__(self,a1,a2,a3, rescale=False,TMAX=100):
        if rescale:
            inp_min, inp_max = (0,1)
            self.a1_min, self.a1_max = (-1, 1)              # gaus heigth
            self.a2_min, self.a2_max = (0, 2)               # gaus width
            self.a3_min, self.a3_max = (0, TMAX)   # gaus position
            a1 = self.a1_min+(self.a1_max-self.a1_min)*(a1-inp_min)/(inp_max-inp_min)
            a2 = self.a2_min+(self.a2_max-self.a2_min)*(a2-inp_min)/(inp_max-inp_min)
            a3 = self.a3_min+(self.a3_max-self.a3_min)*(a3-inp_min)/(inp_max-inp_min) 
        func = lambda x: a1*torch.exp(-torch.abs(a2*(x-a3)**2))
        super().__init__(func, a1, a2, a3)
        
class Step(Function):
    def __init__(self,a1,a2, rescale=False,TMAX=100):
        if rescale:
            inp_min, inp_max = (0,1)
            self.a1_min, self.a1_max = (0, 1)                  # amplitude
            self.a2_min, self.a2_max = (0,TMAX)                # step time
            a1 = self.a1_min+(self.a1_max-self.a1_min)*(a1-inp_min)/(inp_max-inp_min)
            a2 = self.a2_min+(self.a2_max-self.a2_min)*(a2-inp_min)/(inp_max-inp_min)
            
        func = CustomWhere(a1,a2)
        
        super().__init__(func, a1, a2) 
        
    def __str__(self,value=(None,None)):
        func_str = "step(amplitude: a1, time: a2)"
        return self._replace_params(func_str,value)
    
class Rect(Function):
    def __init__(self,a1,a2,a3, rescale=False,TMAX=100):
        if rescale:
            inp_min, inp_max = (0,1)
            self.a1_min, self.a1_max = (-1, 1)               # amplitude
            self.a2_min, self.a2_max = (0,TMAX)              # rise point
            self.a3_min, self.a3_max = (0, 1)                # fall point at remaining percentage
            a1 = self.a1_min+(self.a1_max-self.a1_min)*(a1-inp_min)/(inp_max-inp_min)
            a2 = self.a2_min+(self.a2_max-self.a2_min)*(a2-inp_min)/(inp_max-inp_min)
            a3 = self.a3_min+(self.a3_max-self.a3_min)*(a3-inp_min)/(inp_max-inp_min) 
        func = lambda x: torch.where(torch.logical_or(x < a2 , x > a2+(TMAX-a2)*a3), torch.zeros(1,device=DEVICE),a1)
        super().__init__(func, a1, a2, a3) 
        
    def __str__(self,value=(None,None)):
        func_str = "rect(amplitude: a1, rise: a2, fall: a3)"
        return self._replace_params(func_str,value)
    
    
      
if __name__ ==  "__main__":
    data=torch.rand(2,2,100, device=DEVICE)
    a1 = torch.tensor([1.], device=DEVICE, requires_grad=True)
    a2 = torch.tensor([1.], device=DEVICE, requires_grad=True)
    a3 = torch.tensor([0.2], device=DEVICE, requires_grad=True)

    
    x = torch.linspace(0,100,10000, device=DEVICE)
    
    F = Sawtooth_Wave(a1,a2,a3, rescale=True, TMAX=100) 
    # F2 = Sin(a1,a2,a3, rescale=True, TMAX=10) 
    out = F(x)
   
    out.sum().backward()
    
    # t = torch.linspace(0,10,10000)
    # r = 10
    # # y = r * (t- torch.sin(t))
    
    # y=torch.abs(torch.cos(t)*r)

    
    import matplotlib.pyplot as plt
    plt.plot(x.cpu().detach(),out.cpu().detach())
    plt.show()
    
    #123