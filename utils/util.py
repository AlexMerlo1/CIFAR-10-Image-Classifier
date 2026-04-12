import itertools

import torch
import matplotlib.pyplot as plt
from onnxruntime.quantization import CalibrationDataReader
import numpy as np
from torch.nn.utils import prune
import torch.nn as nn
from evaluations import check_accuracy
import pandas as pd
from torch.nn.utils import prune
import torch.nn as nn
import torch
import itertools
from thop import profile
from tqdm import tqdm



def get_model_size_mb(model, nonzero_params=None, dtype_bytes=4):
    """
    If nonzero_params is provided -> estimates compressed sparse size
    dtype_bytes:
        float32 = 4
        float16 = 2
        int8    = 1
    """

    if nonzero_params is None:
        # fallback (original behavior)
        param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    else:
        # compressed estimate
        param_size = nonzero_params * dtype_bytes

    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())

    size_all_mb = (param_size + buffer_size) / (1024**2)
    return size_all_mb

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


def test_diff_prune_models(csi_model, device, train_loader, test_loader, pytorch_model_path):
    amounts = [0,.3,.5,.7,.75,.8,.85,.9]
    length = len(amounts) ** 3

    records = []  # for DataFrame
    best_acc = -1
    best_model = None
    best_config = None

    for cn1, cn2, an1 in tqdm(itertools.product(amounts, repeat=3), total=length, mininterval=0.5, position=0, leave=True):

        name = f"cn1_{cn1}_cn2_{cn2}_an1_{an1}"

        # fresh model
        model = csi_model().to(device)
        model.load_state_dict(
            torch.load(pytorch_model_path + "/baseline_model.pth", map_location=device)
        )

        # collect layers
        conv_layers = []
        linear_layers = []

        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                conv_layers.append(module)
            elif isinstance(module, nn.Linear):
                linear_layers.append(module)

        # apply pruning
        if len(conv_layers) > 0 and cn1 > 0:
            prune.ln_structured(conv_layers[0], "weight", cn1, dim=0, n=float("-inf"))
            prune.remove(conv_layers[0], "weight")

        if len(conv_layers) > 1 and cn2 > 0:
            prune.ln_structured(conv_layers[1], "weight", cn2, dim=0, n=float("-inf"))
            prune.remove(conv_layers[1], "weight")

        if len(linear_layers) > 0 and an1 > 0:
            prune.ln_structured(linear_layers[0], "weight", an1, dim=0, n=float("-inf"))
            prune.remove(linear_layers[0], "weight")
        nonzero_params = 0
        total_params = 0

        for p in model.parameters():
            total_params += p.numel()
            nonzero_params += torch.count_nonzero(p).item()
        # evaluate
        train_acc = check_accuracy(train_loader, model, device)
        test_acc = check_accuracy(test_loader, model, device)
        effective_model_size_mb = get_model_size_mb(
            model,
            nonzero_params=nonzero_params,
            dtype_bytes=4
        )

        dummy = torch.randn(1, 3, 32, 32).to(device)
        macs, parameters = profile(model, inputs=(dummy,))
        sparsity = 1 - (nonzero_params / total_params)
        flops = macs * 2 
        sparsity_adjusted_flops = flops * (1 - sparsity)

        # store row
        records.append({
            "name": name,
            "cn1": cn1,
            "cn2": cn2,
            "an1": an1,
            "train_accuracy": round(train_acc,2),
            "test_accuracy": round(test_acc, 2),
            "nonzero_parameters": nonzero_params,
            "num_parameters": parameters, 
            "effective_model_size_mb": round(effective_model_size_mb,2),
            "sparsity_adjusted_flops": round(sparsity_adjusted_flops, 2)
        })

        # track best
        if test_acc > best_acc:
            best_acc = test_acc
            best_model = model
            best_config = {
                "name": name,
                "cn1": cn1,
                "cn2": cn2,
                "an1": an1,
                "train_accuracy": round(train_acc,2),
                "test_accuracy": round(test_acc,2),
                "nonzero_parameters": nonzero_params,
                "num_parameters": parameters, 
                "effective_model_size_mb": round(effective_model_size_mb,2),
                "sparsity_adjusted_flops": round(sparsity_adjusted_flops,2)
            }

    # create DataFrame
    df = pd.DataFrame(records)
    df.to_csv("pruning_study_results.csv", index=False)
    return df, best_model, best_config


def build_pruned_model_for_export(csi_model, device, pytorch_model_path):
    # fresh model
    model = csi_model().to(device)
    model.load_state_dict(
        torch.load(pytorch_model_path + "/baseline_model.pth", map_location=device)
    )

    # collect layers
    conv_layers = []
    linear_layers = []

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append(module)
        elif isinstance(module, nn.Linear):
            linear_layers.append(module)

    # apply pruning
    prune.ln_structured(conv_layers[0], "weight", 0, dim=0, n=float("-inf"))
    prune.remove(conv_layers[0], "weight")
    prune.ln_structured(conv_layers[1], "weight", 0, dim=0, n=float("-inf"))
    prune.remove(conv_layers[1], "weight")
    prune.ln_structured(linear_layers[0], "weight", 0.8, dim=0, n=float("-inf"))
    prune.remove(linear_layers[0], "weight")

    return model