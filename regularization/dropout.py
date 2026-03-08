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