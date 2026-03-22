import torch
import torch.nn as nn

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

