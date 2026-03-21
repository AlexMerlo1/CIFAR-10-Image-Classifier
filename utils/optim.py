import torch
from torch.optim import Optimizer
import math
from utils.regularization import csi5140_l2

#build initial gradient decenst to understand how pytorch tracks / updates variables.
class csi5140GD(Optimizer):
    """
    custom implementation of gradient descent leveraging pytorch (built to understand the methodology)
    params are pulled from pytorch
    hyperparamters:
    learning rate def(0.01)
    """
    def __init__(self, params, lr=0.01, weight_decay=0.1):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(csi5140GD, self).__init__(params, defaults)
    def step(self):
        for group in self.param_groups:

            #update hyperparameters
            lr = group['lr'] 
            weight_decay = group.get('weight_decay', 0)

            #work through individual parameters
            for param in group['params']:
                #check for gradient tracking on each parameters
                if param.grad is None: 
                    continue #ignore if not tracking gradients
                
                #get gradient from paramter
                grad = param.grad.data

                #l2
                if weight_decay != 0:
                    csi5140_l2(param, weight_decay)


                #update parameter (w = w - (lr* grad))
                param.data.add_(grad, alpha=-lr)    


#gradient descent with momentum
class csi5140GDM(Optimizer):
    """
    custom implementation of gradient descent with momentum leveraging pytorch
    params are pulled from pytorch
    hyperparamters:
    learning rate def(0.01)
    momentum def(0.9)
    """
    def __init__(self, params, momentum=0.9, lr=0.01, weight_decay=0.1):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super(csi5140GDM, self).__init__(params, defaults)
    def step(self):
        for group in self.param_groups:
            #update hyperparameters
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group.get('weight_decay', 0)
            #work through individual parameters
            
            for param in group['params']:
                #check for gradient tracking on each parameters
                if param.grad is None: 
                    continue #ignore if not tracking gradients
                
                #get gradient & state from paramter
                grad = param.grad.data
                state = self.state[param] #momentum buffer

                #L2
                if weight_decay != 0:
                    csi5140_l2(param, weight_decay)

                #check state is not empty
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(param.data) #zeros_like stays on device, zeros freaks out and tries to access CPU

                #velocity calc
                vel = state['momentum_buffer']
                vel.mul_(momentum).add_(grad)

                #update parameter (w = w - (lr* velocity))
                param.data.add_(vel, alpha=-lr)    


#ADAM Optimizer
class csi5140Adam(Optimizer):
    """
    custom implementation of ADAM leveraging pytorch
    params are pulled from pytorch
    hyperparamters:
    learning rate def(0.001)
    beta 1 def(0.9)
    beta 2 def(0.99)
    eps def(1e-8)

    something seems wrong....review this class
    """
    def __init__(self, params, lr=0.001, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.1):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(csi5140Adam, self).__init__(params, defaults)
    def step(self):
        loss = None
        for group in self.param_groups:

            #update hyperparameters
            lr = group['lr'] 
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group.get('weight_decay', 0)

            #work through individual parameters
            for param in group['params']:

                #check for gradient tracking on each parameters
                if param.grad is None: 
                    continue #ignore if not tracking gradients

                #get gradients and states
                grad = param.grad.data
                state = self.state[param]
                               
                #setup state values
                if len(state) == 0:
                    state['step'] =0

                    #exp moving average of grad values 
                    state['exp_avg'] = torch.zeros_like(param.data)

                    #exp moving average of squared grads
                    state['exp_avg_sq'] = torch.zeros_like(param.data)

                exp_avg = state['exp_avg']
                exp_avg_squared = state['exp_avg_sq']
                state['step'] += 1

                #calculate ADAM
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_squared.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                #bias corrections
                bias_corr1 = 1 - beta1 ** state['step']
                bias_corr2 = 1 - beta2 ** state['step']

                #step size calc
                step_size = lr * (math.sqrt(bias_corr2) / bias_corr1)

                #update parameters
                bottom = exp_avg_squared.sqrt().add_(eps)
                param.data.addcdiv_(exp_avg, bottom, value=-step_size)

                #L2 is this the correct time in the sequence to update the weights?
                if weight_decay != 0:
                    csi5140_l2(param, weight_decay)
        
        return loss