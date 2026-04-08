import torch
import matplotlib.pyplot as plt
from onnxruntime.quantization import CalibrationDataReader
import numpy as np

def show_random_images(dataset, rows=3, cols=3):
  labels_map = {
      0: "airplane",
      1: "automobile",
      2: "bird",
      3: "cat",
      4: "deer",
      5: "dog",
      6: "frog",
      7: "horse",
      8: "ship",
      9: "truck",
  }

  figure = plt.figure(figsize=(8, 8))

  for i in range(1, rows * cols + 1):
      sample_idx = torch.randint(len(dataset), (1,)).item()
      img, label = dataset[sample_idx]
      img = img.clamp(0, 1)
      figure.add_subplot(rows, cols, i)
      plt.title(labels_map[label])
      plt.axis("off")

      plt.imshow(img.permute(1, 2, 0))

  plt.show()

import torch

def get_device():
    if torch.cuda.is_available():
        print("NVIDIA CUDA detected")
        device = torch.device("cuda")

    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        print("Intel XPU detected")
        device = torch.device("xpu")

    elif torch.backends.mps.is_available():
        print("Apple Silicon MPS detected")
        device = torch.device("mps")

    else:
        print("No hardware accelerator found. CPU")
        device = torch.device("cpu")

    print(f"Set to use device: {device}")
    return device

class OnnxDataLoaderTorch(CalibrationDataReader):
    """
    wrap exisitng pytorch dataloader in Onnx calibration loader for static quantization
    dataloader = pytorch test/train
    input_name = input from onnx export, input typically in our case
    """
    def __init__(self, dataloader, input_name="input"):
        super().__init__()
        self.dataloader = iter(dataloader)
        self.input_name = input_name
    
    def get_next(self):
        try:
            batch = next(self.dataloader)
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            data = batch.detach().cpu().numpy()
            if len(data.shape) == 3:
                data = np.expand_dims(data, axis=0) # Now [1, C, H, W]
            return {self.input_name: data}
        except StopIteration:
            return None

