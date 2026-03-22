import torch
import torch.nn as nn
import math

def csi5140_l2(param, l2_lambda):
    """
    simple L2 regularization (weight decay) implementation
    provide parameter.data (AFTER CONFIRMING .GRAD is populated)

    param
    l2_lambda
    """
    param.grad.add_(param.data, alpha=l2_lambda)


class csi5140DDropout(nn.Module):
    """
    overloads nn.Module, creates another layer object
    hyperparameter:
    p: probability of neuron dropping out of network
    """
    def __init__(self, p=0.7):
        super(csi5140DDropout, self).__init__()
        self.p = p
    def forward(self, x):
        #activate only if training 
        if self.training:
            #create random number distribution - neurons are kept alive with p (1-p)
            #this may over/under activate on a single pass but the total distribution will follow the probability
            #without this, there is an inherit bias to the dropout since neuron A will die, neuron B must live messing with the probability

            r_mask = torch.rand(x.shape, device=x.device) > self.p
            return r_mask * x / (1.0-self.p)
        else:
            #exit routine if not training
            return x

def csi5140_cosine_learning_rate_decay(optimizer, step_now, step_max, lr_initial, lr_min):
    """
    calculates current epoch learning rate based on cosine decay, 
    updates optimizer.param learning rates.
    integrate into training loop

    optimizer = optimizer object used in model (custom models work too as we inherit and override the optimizer class)
    step_now = current epoch
    step_max = # of epochs to modify learning rate over. Set to total number of epochs to modify rate over entire batch
    lr_initial = starting learning rate
    lr_min = final learning rate
    """

    #check if we need to modify learning rate - allows for customization of how many epochs to decrease the rate over
    if step_now > step_max:
        #stop applying the algorithm
        lr = lr_min
    else:
        #calculate cosine decay
        cosine_decay = 0.5 * (1+math.cos(math.pi * step_now/step_max))
        #calc learning rate
        lr = lr_min + (lr_initial - lr_min) * cosine_decay
    #update learning rate in optimizer
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    #return learning rate for ablation  study
    return lr

 
class csi5140StepDecay():
    pass