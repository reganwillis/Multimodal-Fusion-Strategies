
# BERT BASE - NO X ATTN
WEIGHTS_DIR = []
WEIGHTS_DIR.append('./trials/trial2/mobilenet_out')
WEIGHTS_DIR.append('./trials/trial2/bert_out')
WEIGHTS_DIR.append('./trials/trial2/latefusionmobilenet_out')
WEIGHTS_DIR.append('./trials/trial2/latefusionvit_out')
WEIGHTS_DIR.append('./trials/trial2/midfusionmobilenet_out')
WEIGHTS_DIR.append('./new_archs2/midfusionvit_out')
WEIGHTS_DIR.append('./new_archs/earlyfusionmobilenet_out')
WEIGHTS_DIR.append('./new_archs2/earlyfusionvit_out')
WEIGHTS_DIR.append('./new_archs/veryearlyfusionvit_out')

# BERT BASE - X ATTN
WEIGHTS_DIR.append('./trials/trial9_crossattn/latefusionmobilenet_xattn_out')
WEIGHTS_DIR.append('./trials/trial9_crossattn/latefusionvit_xattn_out')
WEIGHTS_DIR.append('./trials/trial9_crossattn/midfusionmobilenet_xattn_out')
WEIGHTS_DIR.append('./new_archs2/midfusionvit_xattn_out')
WEIGHTS_DIR.append('./trials/trial9_crossattn/earlyfusionmobilenet_xattn_out')
WEIGHTS_DIR.append('./new_archs2/veryearlyfusionvit_xattn_out')

# BERT LARGE - NO X ATTN
WEIGHTS_DIR.append('./new_archs/latefusionmobilenet_bertlarge_out')
WEIGHTS_DIR.append('./new_archs/latefusionvit_bertlarge_out')
WEIGHTS_DIR.append('./new_archs/midfusionmobilenet_bertlarge_out')
WEIGHTS_DIR.append('./new_archs2/midfusionvit_bertlarge_out')
WEIGHTS_DIR.append('./new_archs/earlyfusionmobilenet_bertlarge_out')
WEIGHTS_DIR.append('./new_archs2/veryearlyfusionvit_bertlarge_out')

# BERT LARGE - X ATTN
WEIGHTS_DIR.append('./new_archs2/latefusionmobilenet_bertlarge_xattn_out')
WEIGHTS_DIR.append('./new_archs2/latefusionvit_bertlarge_xattn_out')
WEIGHTS_DIR.append('./trials/trial10_bertlargecrossattn/midfusionmobilenet_bertlarge_xattn_out')
WEIGHTS_DIR.append('./new_archs2/midfusionvit_bertlarge_xattn_out')
WEIGHTS_DIR.append('./trials/trial10_bertlargecrossattn/earlyfusionmobilenet_bertlarge_xattn_out')
WEIGHTS_DIR.append('./new_archs2/veryearlyfusionvit_bertlarge_xattn_out')

import os
import argparse
import pandas as pd

# parse args
parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='./', required=False)
args = parser.parse_args()

dfs = []
latency_dfs = []
for d in WEIGHTS_DIR:
    try:
        dfs.append(pd.read_csv(f'{args.dir}/{d}/perf.csv'))
    except (FileNotFoundError, NotADirectoryError):
        pass
    name = d.split('/')[-1].split('_')[:-1]
    name2 = ""
    for n in name:
        name2 = name2 + n + '_'
    try:
        latency_dfs.append(pd.read_csv(f'{name2}perf.csv'))
    except (FileNotFoundError, NotADirectoryError):
        pass
acc_df = pd.concat(dfs, ignore_index=True)
acc_df = acc_df.sort_values(by='Test Accuracy')
latency_df = pd.concat(latency_dfs, ignore_index=True)
latency_df = latency_df.sort_values(by='Test Latency')
print(acc_df)
print(latency_df)

import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Assumes these already exist:
#   acc_df     -> columns: ["Architecture", "Test Accuracy", ...]
#   latency_df -> columns: ["Architecture", "Test Latency"]

def normalize_arch(name: str) -> str:
    """
    Strip the trailing output/performance suffix so matching works:
      earlyfusionvit_out   -> earlyfusionvit
      earlyfusionvit_perf  -> earlyfusionvit
    """
    name = name.lower().strip()
    name = re.sub(r"_(out|perf)$", "", name)
    return name

def fusion_stage(name: str) -> str:
    name = name.lower()
    if "veryearlyfusion" in name or "earlyfusion" in name:
        return "early"
    if "midfusion" in name:
        return "mid"
    if "latefusion" in name:
        return "late"
    return "other"

def backbone(name: str) -> str:
    name = name.lower()
    if "mobilenet" in name:
        return "mobilenet"
    if "vit" in name:
        return "vit"
    return "other"

# Copy and prepare merge keys
acc_plot = acc_df.copy()
lat_plot = latency_df.copy()

acc_plot["arch_key"] = acc_plot["Architecture"].apply(normalize_arch)
lat_plot["arch_key"] = lat_plot["Architecture"].apply(normalize_arch)

# If there are duplicate latency rows per model key, keep the first one
lat_plot = lat_plot.drop_duplicates(subset=["arch_key"], keep="first")

# Merge accuracy and latency
plot_df = acc_plot.merge(
    lat_plot[["arch_key", "Test Latency"]],
    on="arch_key",
    how="inner",
)

# Remove non-MobileNet/ViT models (e.g., BERT baseline)
plot_df = plot_df[
    plot_df["Architecture"].apply(lambda x: backbone(x) != "other")
].copy()

# Convert units
plot_df["Accuracy (%)"] = plot_df["Test Accuracy"] * 100
plot_df["Latency (ms)"] = plot_df["Test Latency"] * 1000

# Remove outliers and print them for documentation
outliers = plot_df[plot_df["Accuracy (%)"] < 60].copy()

if not outliers.empty:
    print("Removed outliers (<60% accuracy):")
    print(
        outliers[["Architecture", "Latency (ms)", "Accuracy (%)"]]
        .sort_values(by=["Accuracy (%)", "Latency (ms)"])
        .to_string(index=False)
    )
    print()

# Keep only non-outliers for plotting
plot_df = plot_df[plot_df["Accuracy (%)"] >= 60].copy()

# Style maps
stage_colors = {
    "early": "purple",
    "mid": "orange",
    "late": "red",
    "other": "gray",
}
backbone_markers = {
    "mobilenet": "o",
    "vit": "^",
    "other": "s",
}

fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=160)

for _, row in plot_df.iterrows():
    arch = row["Architecture"]
    stage = fusion_stage(arch)
    bb = backbone(arch)

    ax.scatter(
        row["Latency (ms)"],
        row["Accuracy (%)"],
        s=70,
        marker=backbone_markers[bb],
        color=stage_colors[stage],
        edgecolors="black",
        linewidths=0.5,
        alpha=0.95,
        zorder=3,
    )

ax.set_xlabel("Inference Latency (ms)")
ax.set_ylabel("Test Accuracy (%)")
ax.set_title("Accuracy vs Latency")
ax.grid(True, alpha=0.2)

# Legend
legend_items = [
    Line2D(
        [0], [0],
        marker="o", color="w",
        markerfacecolor="gray", markeredgecolor="black",
        markersize=8, label="MobileNet backbone"
    ),
    Line2D(
        [0], [0],
        marker="^", color="w",
        markerfacecolor="gray", markeredgecolor="black",
        markersize=8, label="ViT backbone"
    ),
    Line2D(
        [0], [0],
        marker="o", color="w",
        markerfacecolor="purple",
        markersize=8, label="Early fusion"
    ),
    Line2D(
        [0], [0],
        marker="o", color="w",
        markerfacecolor="orange",
        markersize=8, label="Mid fusion"
    ),
    Line2D(
        [0], [0],
        marker="o", color="w",
        markerfacecolor="red",
        markersize=8, label="Late fusion"
    ),
]

ax.legend(handles=legend_items, loc="best", frameon=True)

plt.tight_layout()
plt.savefig('all.png')
