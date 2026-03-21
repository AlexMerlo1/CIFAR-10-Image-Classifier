import torch

class csi5140Adam(torch.optim.optimizer):
    def __init__(self, params, alpha, beta_1, beta_2, epsilon):
        super().__init__(params)
        
