from model import PrjModel

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


class Linear(PrjModel):
    """
    Fully connected layer: output = x @ W + b
    Input:  in activations (A-1)
    Output: out activations (A)
    """

    def __init__(self, input, layer_name, no_neurons, activation):
        #may need to know if it is the output layer, input layer, or hidden layer. Not sure. 
        #discuss best method to grab model values (batch_size, others)
        self.input = input
        self.no_neurons = no_neurons
        self.activation = activation
        self.name = name
        self.curr_method = curr_method
        self.weights = None
        self.bias = None


    def forward(self, x):
        #initialize containers
        A = None #update with correct shape
        Z = None #update with correct shape
        pass

    def backward(self, grad):
        #most likely will change to reflect autograd capability
        dw = None
        db = None
        da = None
        db = None
        pass

    


class Conv2d:
    """
    2D Convolutional layer.
    Input:  (batch_size, in_channels, H, W)
    Output: (batch_size, out_channels, H_out, W_out)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weights = None       # (out_channels, in_channels, kernel_size, kernel_size) - He initialization
        self.bias = None          # (out_channels,)
        self.grad_weights = None
        self.grad_bias = None

    def forward(self, x):
        pass

    def backward(self, grad):
        pass




