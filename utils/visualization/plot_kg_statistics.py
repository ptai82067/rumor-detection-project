#!/usr/bin/env python3
"""
Plot D: KG Statistics Chart — PHEME dataset and KG v2 statistics
Output: docs/figures/kg_statistics_chart.png
"""

import os
import sys
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

OUTPUT_PATH = os.path.join("docs", "figures", "kg_statistics_chart.png")
DATA_PATH = os.path.join("data", "processed", "pheme_features_with_graph.csv")


def plot_label_distribution(ax, df):
    """Subplot 1: Pie chart of label distribution (Rumor vs Non-Rumor)."""
    label_counts = df["label"].value_counts()
    sizes = [label_counts.get(0, 0), label_counts.get(1, 0)]
    labels = [f"Non-Rumor (0)\n{sizes[0]:,}", f"Rumor (1)\n{sizes[1]:,}"]
    colors = ["#68D391", "#FC8181"]
    explode = (0, 0.05)

    ax.pie(
        sizes, labels=labels, autopct="%1.1f%%",
        colors=colors, explode=explode,
        startangle=90, shadow=False,
        textprops={"fontsize": 10, "fontweight": "bold"},
        pctdistance=0.6,
    )
    ax.set_title("Phân bố nhãn (Rumor vs Non-Rumor)", fontsize=12, fontweight="bold")


def plot_event_distribution(ax, df):
    """Subplot 2: Horizontal bar chart of tweets per event."""
    event_counts = df["event_id"].value_counts().sort_values(ascending=True)

    event_names = {
        "charliehebdo": "Charlie Hebdo",
        "sydneysiege": "Sydney Siege",
        "ferguson": "Ferguson",
        "ottawashooting": "Ottawa Shooting",
        "germanwings-crash": "Germanwings Crash",
    }
    labels = [event_names.get(str(eid).strip(), str(eid))
              for eid in event_counts.index]
    colors = ["#E53E3E", "#DD6B20", "#D69E2E", "#38A169", "#3182CE"]
    colors = colors[:len(event_counts)]

    bars = ax.barh(labels, event_counts.values, color=colors,
                   edgecolor="#333333", linewidth=0.8, height=0.6)

    for bar, val in zip(bars, event_counts.values):
        ax.text(val + 200, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("Số lượng tweets", fontsize=11)
    ax.set_title("Phân bố số tweet theo sự kiện", fontsize=12, fontweight="bold")
    ax.margins(x=0.15)
    ax.tick_params(axis="y", labelsize=9)


def plot_thread_size_distribution(ax, df):
    """Subplot 3: Histogram of thread_size with KDE overlay."""
    sizes = df["thread_size"].values

    # Cap at 99th percentile for better visualization
    cap = np.percentile(sizes, 99)
    sizes_clipped = np.clip(sizes, 0, cap)

    bins = np.linspace(0, cap, 40)
    n, bins_edges, patches = ax.hist(
        sizes_clipped, bins=bins, color="#4299E1",
        edgecolor="#2B6CB0", linewidth=0.5, alpha=0.8,
    )

    for patch, bin_edge in zip(patches, bins_edges):
        if bin_edge >= 20:
            patch.set_facecolor("#F6AD55")
            patch.set_edgecolor("#DD6B20")
            patch.set_alpha(0.8)

    # KDE overlay
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(sizes_clipped)
        x_range = np.linspace(0, cap, 200)
        kde_values = kde(x_range) * len(sizes_clipped) * (bins_edges[1] - bins_edges[0])
        ax.plot(x_range, kde_values, "r-", linewidth=2.0, alpha=0.7, label="KDE")
        ax.legend(fontsize=9)
    except ImportError:
        pass

    ax.set_xlabel("Thread size (số tweet)", fontsize=11)
    ax.set_ylabel("Số thread", fontsize=11)
    ax.set_title("Phân bố kích thước thread", fontsize=12, fontweight="bold")
    ax.tick_params(labelsize=9)


def plot_kg_summary(ax):
    """Subplot 4: Summary table of KG v2 statistics."""
    ax.axis("off")

    ax.text(
        0.5, 0.95, "Thống kê KG v2 tổng hợp",
        fontsize=12, fontweight="bold",
        ha="center", va="top", transform=ax.transAxes,
    )

    summary_data = [
        ("Tổng triples", "2,732,764"),
        ("Posts", "102,440"),
        ("  SourcePost", "5,802"),
        ("  ReplyPost", "96,638"),
        ("Users", "49,345"),
        ("Events", "5"),
        ("Threads", "5,802"),
        ("Reply edges", "58,070"),
        ("participatesInThread", "67,509"),
    ]

    col_labels = ["Chỉ số", "Giá trị"]
    rows = [(label, val) for label, val in summary_data]

    table = ax.table(
        cellText=[[r[0], r[1]] for r in rows],
        colLabels=col_labels,
        cellLoc="left",
        loc="center",
        colWidths=[0.55, 0.25],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)

    for i in range(len(rows) + 1):
        for j in range(2):
            cell = table[i, j]
            cell.set_edgecolor("#555555")
            cell.set_linewidth(0.5)
            if i == 0:
                cell.set_facecolor("#2B6CB0")
                cell.set_text_props(color="white", fontweight="bold")
            elif i in (1, 4, 6, 8) and j == 0:
                cell.set_facecolor("#EBF8FF")
                cell.set_text_props(fontweight="bold")
            elif i in (2, 3):
                cell.set_facecolor("#F7FAFC")
                cell.set_text_props(style="italic", fontsize=8.5)
            else:
                cell.set_facecolor("white")

    ax.set_position([0.05, 0.05, 0.9, 0.85])


def main():
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Data file not found: {DATA_PATH}")
        sys.exit(1)

    print("[INFO] Loading data...")
    df = pd.read_csv(DATA_PATH)

    label_counts = df["label"].value_counts()
    print(f"  Rumor (1): {label_counts.get(1, 0):,}")
    print(f"  Non-Rumor (0): {label_counts.get(0, 0):,}")
    print(f"  Total: {len(df):,}")

    print("[INFO] Creating figure with 4 subplots...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor("white")

    plot_label_distribution(axes[0, 0], df)
    plot_event_distribution(axes[0, 1], df)
    plot_thread_size_distribution(axes[1, 0], df)
    plot_kg_summary(axes[1, 1])

    fig.suptitle(
        "Thống kê Bộ Dữ Liệu PHEME và Knowledge Graph v2",
        fontsize=15, fontweight="bold", y=0.98,
    )

    plt.subplots_adjust(hspace=0.25, wspace=0.2)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] KG statistics chart saved: {OUTPUT_PATH}")
    print(f"     Size: {os.path.getsize(OUTPUT_PATH):,} bytes")
    print(f"     Label distribution - Rumor: {label_counts.get(1, 0):,}, "
          f"Non-Rumor: {label_counts.get(0, 0):,}")


if __name__ == "__main__":
    main()