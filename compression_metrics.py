import onnxruntime as ort
from evaluations import plot_metrics
from onnx_tool import model_profile
import torch
import os

# Accuracy data 
metrics = torch.load("saved_models/baseline_metrics.pth", map_location="cpu")
# Uncompressed model
onnx_model_path = "rpi_model/csi5140_rpi_model_low_epochs.onnx"
# Macs = model_profile("rpi_model/csi5140_rpi_model.onnx", dynamic_shapes={"input": [1, 3, 32, 32]})
session = ort.InferenceSession(onnx_model_path)

plot_metrics(metrics["train_accs"], metrics["test_accs"], metrics["train_costs"], "Baseline Model")
model_size = os.path.getsize("rpi_model/csi5140_rpi_model.onnx") / (1024 * 1024)
print(f"Model Size: {model_size:.2f} MB")
# Tune model & put on pi

# Test just quantization vs uncompressed

# Pruned & quantized vs uncompressed
# Do_constant_folding vs uncompressed
# All enabled features vs uncompressed
# Explore the effects of pruning on specific layers 