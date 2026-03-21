import torch

def csi5140_l2(param, l2_lambda):
    """
    simple L2 regularization (weight decay) implementation
    provide parameter.data (AFTER CONFIRMING .GRAD is populated)

    param
    l2_lambda
    """
    param.grad.add_(param.data, alpha=l2_lambda)

