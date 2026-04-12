import onnxruntime as ort
from evaluations import plot_metrics
from thop import profile
import torch
import os
from export_final_model import CSI5140_final_model
import numpy as np
import time

# ===== MODEL CONFIG =====
models = {
    "baseline": {
        "pth": "saved_models/baseline_model.pth",
        "onnx": "rpi_model/csi5140_rpi_model_low_epochs.onnx"
    },
    "folding": {
        "pth": "saved_models/baseline_model.pth",
        "onnx": "rpi_model/csi5140_rpi_model_low_epochs_folding.onnx"
    }
    # "pruned": {
    #     "pth": "saved_models/pruned_model.pth",
    #     "onnx": "rpi_model/pruned_model.onnx"
    # },
    # "quantized_baseline": {
    #     "pth": None,  # no PyTorch version
    #     "onnx": "rpi_model/quantized_model.onnx"
    # },
    # "pruned_quantized": {
    #     "pth": None, # no PyTorch version
    #     "onnx": "rpi_model/pruned_quantized_model.onnx"
    # }
}

results = {}

# ===== LOOP =====
for name, paths in models.items():
  print(f"\n--- {name.upper()} ---")

  results[name] = {}

  # ===== FLOPs (ONLY if PyTorch model exists) =====
  if paths["pth"] is not None:
    model = CSI5140_final_model()
    model.load_state_dict(torch.load(paths["pth"], map_location="cpu"))
    model.eval()

    dummy = torch.randn(1, 3, 32, 32)
    macs, params = profile(model, inputs=(dummy,))

    results[name]["flops"] = macs * 2
    results[name]["params"] = params

    print(f"FLOPs: {macs * 2}")
  else:
    results[name]["flops"] = "N/A"

  ## ===== ONNX RUNTIME =====
  session = ort.InferenceSession(paths["onnx"])
  input_name = session.get_inputs()[0].name

  batch_size = 128
  dummy = np.random.randn(batch_size, 3, 32, 32).astype(np.float32)

  # Warmup (no timing)
  for _ in range(50):
      session.run(None, {input_name: dummy})

  # Timing
  latencies = []
  runs = 200

  for _ in range(runs):
      start = time.perf_counter()
      session.run(None, {input_name: dummy})
      end = time.perf_counter()

      latencies.append((end - start) * 1000)

  avg_latency = np.mean(latencies)

  # Throughput (correct FPS)
  fps = (batch_size * 1000) / avg_latency

  results[name]["latency_ms"] = avg_latency
  results[name]["fps"] = fps

  print(f"Latency (batch {batch_size}): {avg_latency:.3f} ms")
  print(f"FPS: {fps:.2f} images/sec")

  # ===== MODEL SIZE =====
  size = os.path.getsize(paths["onnx"]) / (1024 * 1024)
  results[name]["size_mb"] = size

  print(f"Size: {size:.2f} MB")

import matplotlib.pyplot as plt

# ===== PREP DATA =====
models = list(results.keys())

# Collect metrics (handle missing values like "N/A")
def get_metric(metric_name):
    values = []
    for m in models:
        val = results[m].get(metric_name, None)
        if isinstance(val, str) or val is None:
            values.append(0)  # for quantized FLOPs etc.
        else:
            values.append(val)
    return values

metrics_to_plot = ["flops", "latency_ms", "fps", "size_mb"]

# ===== PLOT EACH METRIC =====
for metric in metrics_to_plot:
  values = get_metric(metric)

  plt.figure()
  plt.bar(models, values)
  plt.title(metric.upper())
  plt.xlabel("Model")
  plt.ylabel(metric)

  # rotate labels if needed
  plt.xticks(rotation=30)

  # show values on bars
  for i, v in enumerate(values):
      plt.text(i, v, f"{v:.2f}", ha='center', va='bottom')

  plt.tight_layout()
  plt.show()