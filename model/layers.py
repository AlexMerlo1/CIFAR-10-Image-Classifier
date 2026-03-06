class Model:
  def __init__(self, layers, optimizer=None, regularization=None, augmentation=None):
    self.layers = layers
    # self.optimizer = optimizer if optimizer else SGD(lr=0.01)
    # self.regularization = regularization
    # self.augmentation = augmentation

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


class Linear:
    """
    Fully connected layer: output = x @ W + b
    Input:  (batch_size, in_features)
    Output: (batch_size, out_features)
    """

    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.bias_enabled = bias
        self.weights = None       # (in_features, out_features) - Random initialization
        self.bias = None          # (1, out_features)
        self.grad_weights = None
        self.grad_bias = None

    def forward(self, x):
        pass

    def backward(self, grad):
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


class ReLU:
    """
    ReLU activation: f(x) = max(0, x)
    Input/Output: same shape as input
    """

    def forward(self, x):
        pass

    def backward(self, grad):
        pass


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