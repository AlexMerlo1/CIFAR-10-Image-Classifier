"""
Baseline Model Design:

Architecture:
- Input: (batch_size, 3, 32, 32)
- ReLU after every hidden layer
- Softmax on output layer

Layer Classes:
- Flatten()
- Linear(in_features, out_features, bias=True)
- Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
- ReLU()
- Softmax()
- Dropout(rate=0.0)

Baseline (off by default):
- Optimizer: SGD(lr=0.01)
- No dropout
- No L2 regularization
- Constant learning rate

Optimizers:
SGD:
- lr (default: 0.01)

Adam:
- lr (default: 0.001)
- beta1 (default: 0.9)
- beta2 (default: 0.999)

GD Momentum:
- lr (default: 0.01)
- momentum (default: 0.9)

Learning Rate Schedules (wraps an optimizer):
CosineDecay:
- optimizer (e.g. SGD, Adam)
- T (total steps/epochs)

StepDecay:
- optimizer
- gamma (decay factor, default: 0.1)
- step_size (decay interval, default: 10)

Regularization:
L2:
- weight_decay (default: 0.0 / off)

Data Augmentation:
Augment:
- horizontal_flip (default: False)
- rotation_degrees (default: 0)
- crop (default: False)

Usage:
# Baseline
Model(
    layers=[
        CNN(in_channels=3, out_channels=32, kernel_size=3, padding=1), 
        ReLU(), 
        A Flatten(), 
        Linear(in_features=32*32*32, out_features=256),
        ReLU(),
        Linear(in_features=256, out_features=10),
        Softmax()
    ]
)

# Full configuration
Model(
    layers=[
        Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
        ReLU(),
        Flatten(),
        Linear(in_features=32*32*32, out_features=256),
        ReLU(),
        Dropout(rate=0.3),
        Linear(in_features=256, out_features=10),
        Softmax()
    ],
    optimizer=Adam(lr=0.001),
    regularization=L2(weight_decay=0.001),
    augmentation=Augment(horizontal_flip=True, rotation_degrees=15)
)
"""
import torch.nn as nn
from model.layers import Conv2d as csi5140_Conv2d
from model.layers import Linear as csi5140_Linear
from model.layers import ReLU as csi5140_ReLU
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            csi5140_ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            csi5140_ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            csi5140_Linear(64 * 8 * 8, 256),
            csi5140_ReLU(),
            csi5140_Linear(256, 10)
        )

    def forward(self, x):
        # step through each layer of the network
        return self.network(x)
