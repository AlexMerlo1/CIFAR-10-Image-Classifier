import torch
import torch.nn as nn
import math
import numpy as np

class Conv2d(nn.Module):
    """
    2D Convolutional layer.
    Input X:  (batch_size, in_channels, H, W)
    Output: (batch_size, out_channels, H_out, W_out)

    Forward

    FOR EVERY IMAGE
    Slide filter matrix across X[i]
    Element-wise multipication
    Sum all elements and assign value in output matrix add bias
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # Initialize image/filter weights to small random values
        self.weights = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size) *
            math.sqrt(2 / (in_channels * kernel_size * kernel_size))
        ) # (out_channels, in_channels, kernel_size, kernel_size) 
        self.bias = nn.Parameter(torch.zeros(out_channels))
    def forward(self, x):
        batch_size, in_channels, H, W = x.shape
        H_out = math.floor((H + 2*self.padding - self.kernel_size) / self.stride) + 1
        W_out = math.floor((W + 2*self.padding - self.kernel_size) / self.stride) + 1
        
        
        # https://docs.pytorch.org/docs/stable/generated/torch.nn.Unfold.html
        # Extract sliding blocks from a batched tensor into final matrix
        all_patches = torch.nn.functional.unfold(
            x,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride
        ) # (Batch size, patch size, # of patches)
        W = self.weights.flatten(start_dim=1)   # Reshape so can multiply input img with patches
        # Dot prod of W & all patches
        Z_l = torch.matmul(W, all_patches)  # (batch_size, output_channels, multiplied matrix)
        Z_l = Z_l.reshape(batch_size, self.out_channels, H_out, W_out) # unflatten Z_l
        Z_l = Z_l + self.bias.reshape(1, -1, 1, 1) # Reshape bias to (Batch_size, output_channels,H_out,W_out) for addition
        return Z_l


class Flatten:
    """
    Flattens spatial dimensions into a 1D feature vector per sample.
    Input:  (batch_size, C, H, W)
    Output: (batch_size, C * H * W)
    """
    def forward(self, x):
        pass

    def backward(self, grad):
        pass


class Linear(nn.Module):
    """
    Fully connected layer: output = x @ W + b
    Input:  (batch_size, in_features)
    Output: (batch_size, out_features)
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias_enabled = bias

        #nn parameters
        self.weights = nn.Parameter(torch.randn((self.out_features, self.in_features))*0.01)

        #check bias
        if self.bias_enabled:
            self.bias = nn.Parameter(torch.zeros((self.out_features)))
        else:
            self.register_parameter('bias', None)


    def forward(self, x):
        #calculate Z
        Z = torch.matmul(x, self.weights.t())
        #add bias if enabled
        if self.bias is not None:
            Z += self.bias
        return Z


class ReLU(nn.Module):
    """
    ReLU activation: f(x) = max(0, x)
    Input/Output: same shape as input
    Using zeros_like to keep tensors on the same device (GPU/CPU)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        #this feels suspicously simple
        return torch.maximum(torch.zeros_like(x), x)

    #def backward(self, grad):
    #    pass


class Softmax:
    """
    Softmax activation: f(x_i) = exp(x_i) / sum(exp(x))
    Applied along class dimension (axis=1).
    Input/Output: (batch_size, num_classes)
    """

    def forward(self, x):

        pass

    def backward(self, grad):
        pass


class Dropout:
    """
    Randomly zeroes out neurons during training with probability `rate`.
    Scales remaining activations by 1 / (1 - rate) to preserve expected values.
    Input/Output: same shape as input
    """

    def __init__(self, rate=0.0):
        self.rate = rate
        self.mask = None

    def forward(self, x, training=True):
        pass

    def backward(self, grad):
        pass