import torch
import matplotlib.pyplot as plt

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