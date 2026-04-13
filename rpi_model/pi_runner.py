import onnxruntime as ort
import numpy as np
import torch
from torchvision import datasets, transforms
import time
import os
import warnings
import pandas as pd

#quiet warnings
warnings.filterwarnings("ignore")

def model_run(model_path):

    #optimizations for onnx runner
    options = ort.SessionOptions()
    options.intra_op_num_threads = os.cpu_count() # use all cores
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL #set model back to sequential execution mode
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL #https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html
    options.add_session_config_entry("session.intra_op.allow_spinning", "1") #increases power consumption but keeps threads alive to improve latency. disable for lower power consumption at the cost of slight performance loss.

    ### LOAD MODEL TO PI VIA ONNX ###
    #note path to model
    model_run = ort.InferenceSession(model_path, sess_options=options, providers=['CPUExecutionProvider'])
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
    print(model_path)
    acc = round(100 * correct / total, 2)
    print(f"RPI Deployed Accuracy: {acc}%")
    avg_latency = round(np.mean(latencies), 2)
    fps = round(1000 / avg_latency, 2) if avg_latency > 0 else 0

    print(f"Average Latency: {avg_latency} ms")
    print(f"Frames Per Second (FPS): {fps}")    

    records = {
        "name" : model_path,
        "acc_ms" : acc,
        "avg_latency_ms" : avg_latency,
        "fps" : fps
    }

    return records

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

if __name__ == "__main__":
    uncompressed_model = "rpi_model/csi5140_rpi_model.onnx"
    quantized_8bit = "rpi_model/csi5140_rpi_model_8bit.onnx"
    pruned_uncompressed_model = "rpi_model/csi5140_rpi_model_pruned.onnx"
    pruned_8bit_model = "rpi_model/csi5140_rpi_model_8bit_pruned.onnx"
    try:
        print("running models")
        #run models, get stats
        rpi_model_stats = []
        rpi_model_stats.append(model_run(uncompressed_model))
        rpi_model_stats.append(model_run(quantized_8bit))
        rpi_model_stats.append(model_run(pruned_uncompressed_model))
        rpi_model_stats.append(model_run(pruned_8bit_model))

        #print stats datadframe
        stats_file="rpi_onnx_model_results.csv"
        print(f"Printing model stats to: {stats_file}")
        df = pd.DataFrame(rpi_model_stats)
        df.to_csv("rpi_onnx_model_results.csv", index=False)
        print("complete")
    except Exception as e:
        print ("failed to generate stats: {e}")
