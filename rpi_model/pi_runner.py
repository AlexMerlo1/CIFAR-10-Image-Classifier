import onnxruntime as ort
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import os
import warnings

#quiet warnings
warnings.filterwarnings("ignore")
#get data

# CIFAR-10 normalization stats
cifar_mean = [0.485, 0.456, 0.406]
cifar_std  = [0.229, 0.224, 0.225]

# Test transforms (NO augmentation, but WITH normalization)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=cifar_mean, std=cifar_std)
])

test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=test_transform
)

from torch.utils.data import DataLoader
batch_size = 128
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

#optimizations for onnx runner
options = ort.SessionOptions()
options.intra_op_num_threads = os.cpu_count()
options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
options.add_session_config_entry("session.intra_op.allow_spinning", "1")


### LOAD MODEL TO PI VIA ONNX ###
#note path to model
model_run = ort.InferenceSession("model/csi5140_rpi_model.onnx", sess_options=options, providers=['CPUExecutionProvider'])
input_name = model_run.get_inputs()[0].name

#run model
correct = 0
total = 0
latencies = []
for images, labels in test_loader:
    #confirm no gradient attached
    img_np = images.detach().cpu().numpy()

    #timer
    start_time = time.perf_counter()

    #run onnx session
    ort_inputs = {input_name: img_np}
    ort_outs = model_run.run(None, ort_inputs)

    end_time = time.perf_counter()
    duration_ms = (end_time - start_time) * 1000
    latencies.append(duration_ms)

    #predictions
    pred = np.argmax(ort_outs[0], axis=1)
    correct += (pred == labels.numpy()).sum()
    total += labels.size(0)

print(f"RPI Deployred Accuracy: {100 * correct / total:.2f}%")

avg_latency = np.mean(latencies)
fps = 1000 / avg_latency if avg_latency > 0 else 0

print(f"Average Latency: {avg_latency:.2f} ms")
print(f"Frames Per Second (FPS): {fps:.2f}")