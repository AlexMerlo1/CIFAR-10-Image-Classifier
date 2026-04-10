import torch
from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from evaluations import plot_metrics
from utils.util import get_device
import os
import torch.nn as nn
from onnxruntime.quantization import quantize_static, CalibrationDataReader

def check_accuracy(loader, model, device):
    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            scores = model(x)
            _, predictions = scores.max(1)

            num_correct += (predictions == y).sum().item()
            num_samples += predictions.size(0)
    
    acc = num_correct / num_samples * 100

    model.train()
    return acc  

# CIFAR-10 normalization stats
cifar_mean = [0.485, 0.456, 0.406]
cifar_std  = [0.229, 0.224, 0.225]


# Training transforms (with augmentation + normalization)
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=cifar_mean, std=cifar_std)
])


# Test transforms (NO augmentation, but WITH normalization)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=cifar_mean, std=cifar_std)
])


training_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=train_transform
)

test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=test_transform
)

#dynamic worker allocation
workers = 0
print(f"Setting number of workers to available CPU cores: {workers}")

from torch.utils.data import DataLoader
batch_size = 128
train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, persistent_workers=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, persistent_workers=False)

device = get_device()


class CSI5140_final_model(nn.Module):
    def __init__(self):
        super(CSI5140_final_model, self).__init__()
        self.cn1 = nn.Conv2d(3, 32, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(32)
        self.ReLU = nn.ReLU()
        self.MaxPool = nn.MaxPool2d(2)

        self.cn2 = nn.Conv2d(32, 64, 3, padding=1)
        self.norm2 = nn.BatchNorm2d(64)
        #nn.ReLU(),
        #nn.MaxPool2d(2),

        self.flatten = nn.Flatten()
        self.an1 = nn.Linear(64 * 8 * 8, 256)
        #nn.ReLU(),
        #self.dropout = nn.Dropout()
        self.an2 = nn.Linear(256, 10)
    def forward(self, x):
        #cn1
        x = self.cn1(x)
        x = self.norm1(x)
        x = self.ReLU(x)
        x = self.MaxPool(x)

        #cn2
        x = self.cn2(x)
        x = self.norm2(x)
        x = self.ReLU(x)
        x = self.MaxPool(x)

        #an1

        x = self.flatten(x)
        x = self.an1(x)
        x = self.ReLU(x)
        #x = self.dropout(x)
        
        #an2

        x = self.an2(x)
        
        return x

#initialize model
TheModel = CSI5140_final_model().to(device)
epochs = 3
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
            TheModel.parameters(),
            lr=0.001,
            betas=(0.9, 0.99),
            weight_decay=1e-4
        )
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=0.0001
        )
train_accs = []
test_accs = []
train_costs = []
iteration = 0
for epoch in range(epochs):
    TheModel.train()
    correct = 0
    total = 0
    epoch_loss = 0
    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad() # Set gradients back to 0

        outputs = TheModel(images)
        loss = criterion(outputs, labels)

        loss.backward() # Backprop
        optimizer.step() # Update weights

        train_costs.append((iteration,round(loss.item(), 4)))
        epoch_loss += loss.item()
        iteration += 1
        
        _, predicted = torch.max(outputs, 1) # Get best class for every sample
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total
    test_acc = check_accuracy(test_loader, TheModel, device)
    avg_loss = epoch_loss / len(train_loader)

    train_accs.append(round(train_acc, 4))
    test_accs.append(round(test_acc, 4))

    scheduler.step()

    print(f"Epoch {epoch+1}: Loss {avg_loss:.4f}, Train {train_acc:.2f}%, Test {test_acc:.2f}%")

TheModel.eval()
TheModel.to("cpu")
inp_tensor = torch.randn(1, 3, 32, 32).to("cpu")

# SAVE BASELINE MODEL
os.makedirs("saved_models", exist_ok=True)
torch.save(TheModel.state_dict(), "saved_models/baseline_model.pth")
print("Baseline model saved.")
# SAVE METRICS
metrics = {
    "train_accs": train_accs,
    "test_accs": test_accs,
    "train_costs": train_costs
}
torch.save(metrics, "saved_models/baseline_metrics.pth")
print("Metrics saved.")

try:
    torch.onnx.export(
        TheModel, 
        inp_tensor, 
        "rpi_model/csi5140_rpi_model_low_epochs.onnx",
        export_params=True, 
        opset_version=12, # Opset 12 is highly compatible with RPi
        # do_constant_folding=True,
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}, 
        dynamo=False
    )
except Exception as e:
    print (f"exporting model failed: {e}")
print("complete.")
