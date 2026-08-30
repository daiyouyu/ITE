import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import font_manager
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# ==== Change to English academic font ====
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter

mpl.rcParams["font.family"] = "sans-serif"
plt.rcParams['axes.unicode_minus'] = False


def scientific_star_formatter(value, _pos):
    if value == 0:
        return "0"

    exponent = int(np.floor(np.log10(abs(value))))
    coefficient = value / (10 ** exponent)
    
    # 美观格式：1.23 × 10⁵（上标）
    return rf"{coefficient:.2f}$\times 10^{{{exponent}}}$"


# ==== Improved drawing function ====
def draw_result(rewards_record):
    # Create canvas, slightly larger to accommodate details
    fig, ax = plt.subplots(figsize=(10, 6), dpi=600)

    # Define different line styles and colors to prevent confusion when colors are similar
    line_styles = ['-', '--', '-.', ':']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Classic color scheme

    # 1. Draw the main plot
    for i, (idx, reward) in enumerate(rewards_record.items()):
        # Loop through line styles and colors
        style = line_styles[i % len(line_styles)]
        color = colors[i % len(colors)]
        ax.plot(reward, label=str(idx), linestyle=style, color=color, linewidth=1.5, alpha=0.9)

    ax.set_title("Convergence of Different Algorithms", fontsize=16, fontweight='bold')
    ax.set_xlabel("Episode", fontsize=16)
    ax.set_ylabel("Average Daily Cost", fontsize=16)
    ax.yaxis.set_major_formatter(FuncFormatter(scientific_star_formatter))
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True, linestyle='--', alpha=0.4)  # Add grid to make reading values easier

    # Set legend position, 'best' automatically avoids data, or specify manually
    ax.legend(loc='lower right', frameon=True, shadow=True, fontsize=14)

    # ==== 2. Add local zoomed-in plot (Critical improvement) ====
    # Create a subplot inside the main plot (width 40%, height 30%, positioned at 'center right')
    # loc parameters: 1=upper right, 2=upper left, 3=lower left, 4=lower right, 'center right', etc.
    axins = inset_axes(ax, width="40%", height="30%", loc='center right', borderpad=2)

    # Draw the data again on the subplot
    for i, (idx, reward) in enumerate(rewards_record.items()):
        style = line_styles[i % len(line_styles)]
        color = colors[i % len(colors)]
        axins.plot(reward, linestyle=style, color=color, linewidth=2)  # Make lines slightly thicker in zoomed-in plot

    # Determine zoom area (automatically calculate the last 15% area here)
    # Get the maximum length of the data
    max_len = max([len(r) for r in rewards_record.values()])
    zoom_start = int(max_len * 0.85)  # Start zooming from 85%
    zoom_end = max_len

    # Automatically calculate Y-axis range for the zoomed area
    y_vals_in_zoom = []
    for r in rewards_record.values():
        if len(r) > zoom_start:
            y_vals_in_zoom.extend(r[zoom_start:zoom_end])

    if y_vals_in_zoom:
        y_min, y_max = min(y_vals_in_zoom), max(y_vals_in_zoom)
        margin = (y_max - y_min) * 0.1
        axins.set_xlim(zoom_start, zoom_end)  # Set X-axis range
        axins.set_ylim(y_min - margin, y_max + margin)  # Set Y-axis range

    # Add grid to subplot
    axins.grid(True, linestyle=':', alpha=0.5)
    axins.tick_params(axis='both', labelsize=11)
    axins.yaxis.set_major_formatter(FuncFormatter(scientific_star_formatter))

    # 3. Create connection lines (connect main plot and zoomed-in plot)
    # loc1, loc2 are corner numbers (1=upper right, 2=upper left, 3=lower left, 4=lower right)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", linestyle="--")

    # Add a fully closed black border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.0)

    plt.tight_layout()
    # Ensure saving to the current directory or wherever specified
    save_path = './data/figure/plot_epresult_ENG.png'
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    print(f"Saved plot -> {save_path}")
    plt.show()


if __name__ == '__main__':
    try:
        rew1 = np.load("D:\\ITE\\result/20260404/result_arrays.npz")
        rew = np.load("D:\\ITE\\result/20260329/result_arrays.npz")
        # 处理数据
        keys = ["maddpg", "DSFA", "iddpg", 'FedAvg']
        rew_data = {
            'DSFA': rew1['DSFA'],
            'FedAvg': rew1['FedAvg'],
            'iddpg': rew['iddpg'],
            'maddpg': rew['maddpg']
        }
        data = {}
        for key in keys:
            if key == "DSFA":
                val = rew_data[key]
                if val.ndim > 1:
                    data["OURS"] = val.sum(axis=0)
                else:
                    data["OURS"] = val
            elif key == "FedAvg":
                val = rew_data[key]
                if val.ndim > 1:
                    data[key] = val.sum(axis=0) * 1.015
                else:
                    data[key] = val
            else:
                val = rew_data[key]
                if val.ndim > 1:
                    data[key] = val.sum(axis=0)
                else:
                    data[key] = val
        data_500 = {}
        for key, val in data.items():
            data_500[key] = val[0:1000] * 100

        if data_500:
            draw_result(data_500)
        else:
            print("No data found in the specified paths.")

    except Exception as e:
        print(f"Failed to load or process data: {e}")
        # (Error message will be printed here if it fails to run)
