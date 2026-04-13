import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path
from utils.util import get_model_size_mb
import numpy as np
def save_graph(filename, folder="plots/compression_metrics_plots", dpi=300):
    """
    Saves the current matplotlib figure.

    Args:
        filename (str): Name of the file (e.g., "plot.png")
        folder (str): Directory to save the plot
        dpi (int): Resolution
    """
    output_dir = Path(folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")

    print(f"Saved to: {output_path}")

df = pd.read_csv("pruning_study_results.csv")


# ======== Pruning vs Accuracy ======== 
df_simple = df[(df["cn1"] == 0) & (df["cn2"] == 0)]

plt.plot(
    df_simple["nonzero_parameters"],
    df_simple["test_accuracy"],
    marker="o"
)

# format with commas
formatter = FuncFormatter(lambda x, _: f"{int(x):,}")
plt.gca().xaxis.set_major_formatter(formatter)

plt.gca().invert_xaxis()

plt.xlabel("Number of Non-Zero Parameters")
plt.ylabel("Test Accuracy (%)")
plt.title("Effect of Linear Layer Pruning")

save_graph("an1_pruning_effect.png")

plt.show()

# ======== FLOPs vs Accuracy ======== 

plt.figure()

plt.plot(
    df_simple["sparsity_adjusted_flops"],
    df_simple["test_accuracy"],
    marker="o"
)

# format in millions
formatter = FuncFormatter(lambda x, _: f"{x/1e6:.1f}M")
plt.gca().xaxis.set_major_formatter(formatter)

plt.gca().invert_xaxis()

plt.xlabel("Sparsity-Adjusted FLOPs (Millions)")
plt.ylabel("Test Accuracy (%)")
plt.title("Effect of Linear Layer Pruning (FLOPs vs Accuracy)")

save_graph("an1_flops_vs_accuracy.png")

plt.show()

# ======== Quantized vs Pruned vs Baseline ======== 

# pruned (an1 = 0.8)
pruned_model = df[
    (df["cn1"] == 0) &
    (df["cn2"] == 0) &
    (df["an1"] == 0.8)
]
pruned_row = pruned_model.iloc[0]

# baseline (no pruning)
baseline_model = df[
    (df["cn1"] == 0) &
    (df["cn2"] == 0) &
    (df["an1"] == 0)
]
baseline_row = baseline_model.iloc[0]

# sizes
baseline_fp32 = get_model_size_mb(None, baseline_row["nonzero_parameters"], dtype_bytes=4)
baseline_int8 = get_model_size_mb(None, baseline_row["nonzero_parameters"], dtype_bytes=1)

pruned_fp32 = get_model_size_mb(None, pruned_row["nonzero_parameters"], dtype_bytes=4)
pruned_int8 = get_model_size_mb(None, pruned_row["nonzero_parameters"], dtype_bytes=1)

# plot
plt.figure()

labels = [
    "Baseline (FP32)",
    "Baseline (INT8)",
    "Pruned (FP32)",
    "Pruned (INT8)"
]

sizes = [
    baseline_fp32,
    baseline_int8,
    pruned_fp32,
    pruned_int8
]

plt.bar(labels, sizes)

plt.ylabel("Model Size (MB)")
plt.title("Model Size Comparison (Baseline vs Pruning vs Quantization)")

# value labels
for i, v in enumerate(sizes):
    plt.text(i, v, f"{v:.2f} MB", ha='center', va='bottom')

save_graph("model_size_comparison_full.png")

plt.show()


# ======== RPI Model Performance Comparison ======== 

df_models = pd.read_csv("rpi_onnx_model_results.csv")

# clean model names (optional, nicer labels)
df_models["Model"] = df_models["name"].apply(lambda x: 
    "Baseline" if "8bit" not in x and "pruned" not in x else
    "Quantized (INT8)" if "8bit" in x and "pruned" not in x else
    "Pruned" if "pruned" in x and "8bit" not in x else
    "Pruned + Quantized"
)
plt.figure()
plt.bar(df_models["Model"], df_models["avg_latency_ms"])
plt.ylabel("Latency (ms)")
plt.title("Model Latency Comparison")

for i, v in enumerate(df_models["avg_latency_ms"]):
    plt.text(i, v, f"{v:.2f}", ha='center', va='bottom')

save_graph("model_latency_comparison.png")
plt.show()
plt.figure()
plt.bar(df_models["Model"], df_models["acc_ms"])
plt.ylabel("Test Accuracy (%)")
plt.title("Model Accuracy Comparison")

for i, v in enumerate(df_models["acc_ms"]):
    plt.text(i, v, f"{v:.2f}%", ha='center', va='bottom')

save_graph("model_accuracy_comparison.png")
plt.show()
plt.figure()

plt.scatter(df_models["avg_latency_ms"], df_models["acc_ms"])

for i, row in df_models.iterrows():
    plt.text(row["avg_latency_ms"], row["acc_ms"], row["Model"])

plt.xlabel("Latency (ms)")
plt.ylabel("Test Accuracy (%)")
plt.title("Latency vs Accuracy Tradeoff")

save_graph("latency_vs_accuracy.png")
plt.show()

models = ["Baseline", "Pruned"]

fps_no_quant = [
    df[df["name"].str.contains("rpi_model.onnx")]["fps"].values[0],
    df[df["name"].str.contains("pruned.onnx") & ~df["name"].str.contains("8bit")]["fps"].values[0]
]

fps_quant = [
    df[df["name"].str.contains("8bit.onnx") & ~df["name"].str.contains("pruned")]["fps"].values[0],
    df[df["name"].str.contains("8bit_pruned")]["fps"].values[0]
]

x = np.arange(len(models))
width = 0.35

plt.figure()

plt.bar(x - width/2, fps_no_quant, width, label="FP32")
plt.bar(x + width/2, fps_quant, width, label="INT8 Quantized")

plt.xticks(x, models)
plt.ylabel("Frames Per Second (FPS)")
plt.title("Impact of Quantization on Throughput (FPS)")

# annotate
for i in range(len(models)):
    plt.text(x[i] - width/2, fps_no_quant[i], f"{fps_no_quant[i]:.2f}", ha='center', va='bottom')
    plt.text(x[i] + width/2, fps_quant[i], f"{fps_quant[i]:.2f}", ha='center', va='bottom')

plt.legend()

save_graph("quantization_fps_comparison.png")
plt.show()

# isolate ONLY conv1 pruning 
df_conv1 = df[
    (df["cn2"] == 0) &
    (df["an1"] == 0)
].sort_values("cn1")

plt.figure()

plt.plot(
    df_conv1["cn1"],
    df_conv1["test_accuracy"],
    marker="o"
)

plt.xlabel("Conv1 Pruning Amount")
plt.ylabel("Test Accuracy (%)")
plt.title("Effect of Conv1 Pruning on Accuracy")


save_graph("cn1_pruning_vs_accuracy.png")
plt.show()