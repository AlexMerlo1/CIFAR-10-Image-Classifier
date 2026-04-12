import torch
from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from utils.util import get_device
import os
import torch.nn as nn
from onnxruntime.quantization import quantize_static, QuantFormat, QuantType, quant_pre_process
from utils.util import OnnxDataLoaderTorch
import psutil
from pathlib import Path
from evaluations import check_accuracy
from utils.util import test_diff_prune_models

force_training = False #set this to true to force the model to retrain, otherwise if model parameters exist it will use those.

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
        
if __name__ == "__main__":
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
    total_mem = psutil.virtual_memory().total
    total_mem = total_mem / (1024**3)
    print(f"Total RAM: {total_mem:.2f} GB")
    workers = os.cpu_count()
    print(f"Available CPU Cores: {workers}")
    if workers > 8 and total_mem > 17:
        print(f"Setting Worker Count To: {workers}")
    elif workers > 8 and total_mem < 17:
        print("Not Enough Memory, Fixing Worker Count To 4")
        workers = 4
    else:
        print("System Does Not Meet Requirements For Worker Optimization, Fixing Worker Count to 1")
        workers = 1

    from torch.utils.data import DataLoader
    batch_size = 128
    train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, persistent_workers=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, persistent_workers=True)

    device = get_device()

    #initialize model
    TheModel = CSI5140_final_model().to(device)
    epochs = 100
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
    #check for trained model data 
    torch_metrics_folder = "pytorch_model_metrics"
    torch_metrics_path = Path(torch_metrics_folder)
    torch_metrics_path.mkdir(exist_ok=True)
    existing_model = Path(torch_metrics_folder + "/baseline_model.pth")
    if existing_model.is_file() and not(force_training):
        print(f"model data exists in file: {str(existing_model)}, skipping training")
    else:
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

        try:
            torch.save(TheModel.state_dict(), (torch_metrics_folder + "/baseline_model.pth"))
            metrics = {
                "train_accs": train_accs,
                "test_accs": test_accs,
                "train_costs": train_costs
            }
            torch.save(metrics, (torch_metrics_folder + "/baseline_metrics.pth"))
            print(f"Baseline PyTorch Model & Baseline Metrics Saved To: {torch_metrics_folder}")
        except Exception as e:
            print(f"Failed to save torch model: {e}")
    
    #test pruning
    df, best_model, best_config = test_diff_prune_models(
        CSI5140_final_model,
        device,
        train_loader,
        test_loader,
        torch_metrics_folder
    )

    print(df.sort_values(by="test_accuracy", ascending=False).head(5))
    print("\nBEST CONFIG:")
    print(best_config)

        #TODO: to handle the best model, push to ONNX

    TheModel.eval()
    TheModel.to("cpu") #forcing model back to CPU to prevent issues with ONNX export.
    inp_tensor = torch.randn(1, 3, 32, 32).to("cpu")

    #export ONNX models 
    onnx_export_folder = "rpi_model"
    onnx_export_path = Path(onnx_export_folder)
    onnx_export_path.mkdir(exist_ok=True)

    #export baseline uncompressed model
    uncompressed_model_path = (onnx_export_folder + "/csi5140_rpi_model.onnx")
    try:
        torch.onnx.export(
            TheModel, 
            inp_tensor, 
            uncompressed_model_path,
            export_params=True, 
            opset_version=12, # Opset 12 is highly compatible with RPi
            do_constant_folding=True,
            input_names=['input'], 
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}, 
            dynamo=False
        )
        print(f"Baseline ONNX Model Exported to: {uncompressed_model_path}")
    except Exception as e:
        print (f"exporting model failed: {e}")

    #quantize model
    #preprocess quantization corrects shape errors, merge conv & batchnorm layers...etc
    preprocess_model= (onnx_export_folder + "/csi5140_rpi_model_prep.onnx")
    try:
        quant_pre_process(
            input_model=uncompressed_model_path,
            output_model_path=preprocess_model,
            skip_optimization=False,
            auto_merge=False
        )
        print(f"Preprocessed ONNX Quantization Model Exported to: {preprocess_model}")
    except Exception as e:
        print (f"unable to pre-process model: error: {e}")

    CalibrationData = OnnxDataLoaderTorch(training_data)
    quantized_model_path = (onnx_export_folder + "/csi5140_rpi_model_8bit.onnx")
    try:
        quantize_static(
            model_input=preprocess_model, 
            model_output=quantized_model_path,
            calibration_data_reader=CalibrationData, 
            quant_format=QuantFormat.QDQ, 
            activation_type=QuantType.QInt8, 
            weight_type=QuantType.QInt8
        )
        print(f"8Bit Int ONNX Quantized Model Exported to: {quantized_model_path}")
    except Exception as e:
        print(f"unable to quantize model, error: {e}")
    print("complete.")
