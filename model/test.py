import torch
import torchvision
import torchvision.transforms as transforms

# transform → converts image to tensor
transform = transforms.ToTensor()

train_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)
from layers import Conv2d
from model import Model

model = Model()
for x, y in train_loader:
    print("x shape:", x.shape)  # (32, 3, 32, 32)
    print("y shape:", y.shape)  # (32,)
    break
for x, y in train_loader:
    output = model(x)

    print("output shape:", output.shape)  # (32, 10)
    break
import torch.nn as nn
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1):  # just 1 epoch for testing
    for x, y in train_loader:
        optimizer.zero_grad()

        output = model(x)
        loss = criterion(output, y)

        loss.backward()
        optimizer.step()

    print("loss:", loss.item())