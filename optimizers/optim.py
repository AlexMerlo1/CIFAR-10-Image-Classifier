import torch
from torch.optim import Optimizer

class csi5140Adam(Optimizer):
    def __init__(self, params, alpha, beta_1, beta_2, epsilon):
        super().__init__(params)
        

#build initial gradient decenst to understand how pytorch tracks / updates variables.
class csi5140GD(Optimizer):
    """
    custom implementation of gradient descent leveraging pytorch (built to understand the methodology)
    params are pulled from pytorch
    hyperparamters:
    learning rate (lr)
    """
    def __init__(self, params, lr=0.02):
        defaults = dict(lr=lr)
        super(csi5140GD, self).__init__(params, defaults)
    def step(self):
        for group in self.param_groups:
            lr = group['lr'] #update hyperparameters
            #work through individual parameters
            for param in group['params']:
                #check for gradient tracking on each parameters
                if param.grad is None: 
                    continue #ignore if not tracking gradients
                #get gradient from paramter
                grad = param.grad.data
                #update parameter (w = w - (lr* grad))
                param.data.add_(grad, alpha=-lr)    


#gradient descent with momentum
class csi5140GDM(Optimizer):
    """
    custom implementation of gradient descent with momentum leveraging pytorch
    params are pulled from pytorch
    hyperparamters:
    learning rate (lr)
    momentum (momentum)
    """
    def __init__(self, params, momentum=0.9, lr=0.02):
        defaults = dict(lr=lr, momentum=momentum)
        super(csi5140GDM, self).__init__(params, defaults)
    def step(self):
        for group in self.param_groups:
            lr = group['lr'] #update hyperparameters
            momentum = group['momentum']
            #work through individual parameters
            
            for param in group['params']:
                #check for gradient tracking on each parameters
                if param.grad is None: 
                    continue #ignore if not tracking gradients

                #get gradient & state from paramter
                grad = param.grad.data
                state = self.state[param] #momentum buffer

                #check state is not empty
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(param.data) #zeros_like stays on device, zeros freaks out and tries to access CPU

                #velocity calc
                vel = state['momentum_buffer']
                vel.mul_(momentum).add_(grad)

                #update parameter (w = w - (lr* velocity))
                param.data.add_(vel, alpha=-lr)    
