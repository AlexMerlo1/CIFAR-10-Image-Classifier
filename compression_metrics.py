import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path
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