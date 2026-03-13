# Convolutional Layer
import torch.nn as nn
"""
Input: Normalized X matrix

# Conv2D Parameters
in_channels (3): 1 for grayscale, 3 for RGB, ...
out_channels: number of our choosing - each filter learns something different
    ex: edges, corners, textures, shapes, ...
kernel_size (3x3): Imagine 3x3 grid sliding across all pixels
Stride (default=1): Moves x pixels at a time (default 1 pixel) 
Padding: Padding added to all 4 sides (Default with 0 padding) 

N = height in or width in
P = Padding
K = Kernal Size
S = Stride

output size = floor((n + 2P - K) / S) + 1

# Other parameters can be included but optional
"""
#activations
class ReLU:
    def __init__(self, Z):
        self.Z = Z

class SoftMax:
    def __init__(self, Z):
        self.Z = Z

class Flatten:
    def __init__(self, Z):
        self.Z = Z
    
#convolutional layer
class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, H_in, W_in):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.H_in = H_in
        self.W_in = W_in

#linear
class Linear:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features




# Conv2D hyperparameters
in_channels = 3
out_channels = 32
kernel_size = 3
stride = 1
padding = 1
H_in = 32 # Num of pixels height
W_in = 32 # Num of pixels width
layer1_conv = nn.Conv2d(
    in_channels=in_channels,
    out_channels=out_channels,
    kernel_size=kernel_size,
    stride=stride,
    padding=padding
)
# ReLU activation layer
"""
input: previous layers Z
No parameters required
"""
layer2_relu = nn.ReLU()
# Flatten Layer
"""
input: Torch tensor (Batch size, All output features of previous layer)
ex usage
t = torch.tensor([[[1, 2],
                   [3, 4]],
                  [[5, 6],
                   [7, 8]]])
torch.flatten(t)
tensor([1, 2, 3, 4, 5, 6, 7, 8])
torch.flatten(t, start_dim=1)
tensor([[1, 2, 3, 4],
        [5, 6, 7, 8]])
"""

layer3_flatten = nn.flatten()
# Fully-Connected Layer
import math
"""
Input: Flattened Layer

Parameters:
in_features: size of each input sample 
out_features: size of each output sample
- Change depending on layer - if final layer should be the logit for every class
- if final layer we want output of 10 for 10 classes
bias: True/False to include or exclude bias

"""
# Size: H*W*F
# Output Height & Width of convolutional layer
H_out = math.floor((H_in + 2*padding - kernel_size) / stride) + 1
W_out = math.floor((W_in + 2*padding - kernel_size) / stride) + 1


in_features = out_channels * H_out * W_out # Flattened Size
FC_output_features = 10
include_bias = True

layer4_FC = nn.Linear(in_features=in_features, 
                      out_features=FC_output_features,
                      )  

# Softmax
"""
Input: Class logits

Parameters:
dim: Dim 1 ensures row sums to 1, 0 down columns
"""


layer5_softmax = nn.Softmax(dim=1)

def get_pytorch_layers():
    layers = [layer1_conv, layer2_relu, layer3_flatten, layer4_FC, layer5_softmax]
    return layers