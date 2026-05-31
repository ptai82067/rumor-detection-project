#!/usr/bin/env python3
"""
Plot C: KG Sample Subgraph — Real conversation thread from PHEME
Output: docs/figures/kg_sample_subgraph.png
"""

import os
import sys
import warnings
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

OUTPUT_PATH = os.path.join("docs", "figures", "kg_sample_subgraph.png")
DATA_PATH = os.path.join("data", "processed", "pheme_features_with_graph.csv")


def select_thread(df):
    """Select a suitable thread for visualization."""
    thread_sizes = df.groupby("thread_id").size()
    candidate_threads = thread_sizes[
        (thread_sizes >= 8) & (thread_sizes <= 15)
    ].index

    rumor_data = df[df["label"] == 1]

    for tid in candidate_threads:
        if tid not in rumor_data["thread_id"].values:
            continue
        thread_df = rumor_data[rumor_data["thread_id"] == tid]
        if thread_df["depth"].max() >= 3:
            return tid, thread_df

    for tid in candidate_threads:
        thread_df = df[df["thread_id"] == tid]
        if len(thread_df) >= 5:
            return tid, thread_df

    first_tid = df["thread_id"].iloc[0]
    return first_tid, df[df["thread_id"] == first_tid]


def build_reply_graph(thread_df):
    """Build a directed graph from the reply structure."""
    G = nx.DiGraph()

    # First pass: add all nodes
    for _, row in thread_df.iterrows():
        post_id = int(row["post_id"])
        post_id_short = str(post_id)[:8]
        depth = int(row["depth"])
        subtree_reply = int(row.get("subtree_reply_count", 0))
        user_rumor_ratio = float(row.get("user_prior_rumor_ratio", 0))
        is_source = bool(row.get("is_source", depth == 0))

        G.add_node(
            post_id_short,
            depth=depth,
            is_source=is_source,
            subtree_reply=subtree_reply,
            user_rumor_ratio=user_rumor_ratio,
            full_post_id=post_id,
        )

    # Second pass: add edges based on reply_to
    for _, row in thread_df.iterrows():
        post_id = int(row["post_id"])
        post_id_short = str(post_id)[:8]

        if pd.notna(row["reply_to"]):
            reply_to = int(row["reply_to"])
            reply_to_short = str(reply_to)[:8]

            if reply_to_short in G.nodes():
                G.add_edge(reply_to_short, post_id_short)
            elif reply_to in thread_df["post_id"].values:
                match_row = thread_df[thread_df["post_id"] == reply_to]
                if not match_row.empty:
                    match_short = str(int(match_row.iloc[0]["post_id"]))[:8]
                    G.add_edge(match_short, post_id_short)

    return G


def custom_hierarchical_layout(G):
    """Create a hierarchical layout without graphviz."""
    depths = {}
    for node, data in G.nodes(data=True):
        depths[node] = data.get("depth", 0)

    if not depths:
        return {}

    max_depth = max(depths.values())
    depth_groups = {}
    for node, d in depths.items():
        depth_groups.setdefault(d, []).append(node)

    pos = {}
    for depth, nodes_at_depth in depth_groups.items():
        n_nodes = len(nodes_at_depth)
        for i, node in enumerate(sorted(nodes_at_depth)):
            if n_nodes == 1:
                x = 0
            else:
                x = (i / (n_nodes - 1)) * 2 - 1
            # Use depth directly: root at top (high y)
            y = max_depth - depth + 1
            pos[node] = (x * 3.0, y * 2.0)

    return pos


def draw_sample_subgraph(G, thread_id, thread_df):
    """Draw the sample subgraph."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    fig.patch.set_facecolor("white")

    # Get layout
    pos = custom_hierarchical_layout(G)

    if not pos:
        ax.text(0.5, 0.5, "Empty graph", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    # Depth colors
    depth_colors = {
        0: "#E53E3E",
        1: "#90CDF4",
        2: "#63B3ED",
        3: "#4299E1",
    }
    default_depth_color = "#2B6CB0"

    # Draw edges
    for src, tgt in G.edges():
        if src in pos and tgt in pos:
            x1, y1 = pos[src]
            x2, y2 = pos[tgt]
            ax.annotate(
                "",
                xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="->",
                    color="#AAAAAA",
                    linewidth=1.2,
                    connectionstyle="arc3,rad=0.1",
                    shrinkA=10,
                    shrinkB=10,
                ),
                zorder=1,
            )

    # Draw nodes
    for node in G.nodes():
        if node not in pos:
            continue
        x, y = pos[node]
        data = G.nodes[node]
        depth = data.get("depth", 0)
        is_source = data.get("is_source", False)
        subtree_reply = data.get("subtree_reply", 0)
        user_rumor_ratio = data.get("user_rumor_ratio", 0)

        if is_source:
            size = 0.35
            color = "#E53E3E"
            label_text = f"SOURCE\n{node}\nd={depth}"
        else:
            base_size = 0.18
            bonus = min(subtree_reply / 30, 0.15)
            size = base_size + bonus
            color = depth_colors.get(depth, default_depth_color)
            label_text = f"{node}\nd={depth}"

        edge_color = "#CC3333" if user_rumor_ratio > 0.5 else "#333333"
        edge_width = 2.5 if user_rumor_ratio > 0.5 else 1.0

        if is_source:
            rect = mpatches.FancyBboxPatch(
                (x - size, y - size), size * 2, size * 2,
                boxstyle="round,pad=0.05",
                facecolor=color, edgecolor=edge_color,
                linewidth=edge_width, zorder=10,
            )
            ax.add_patch(rect)
        else:
            circle = plt.Circle(
                (x, y), size,
                facecolor=color, edgecolor=edge_color,
                linewidth=edge_width, zorder=10,
            )
            ax.add_patch(circle)

        text_color = "white" if (is_source or depth >= 3) else "#1A202C"
        ax.text(
            x, y, label_text,
            fontsize=6, ha="center", va="center",
            color=text_color, fontweight="bold" if is_source else "normal",
            zorder=11,
        )

    # Info box
    source_nodes = [n for n in G.nodes() if G.nodes[n].get("is_source", False)]
    if source_nodes and source_nodes[0] in pos:
        sx, sy = pos[source_nodes[0]]
    else:
        sx, sy = -4, 3

    n_posts = len(thread_df)
    max_depth = max(
        (d for _, d in G.nodes(data="depth") if d is not None), default=0
    )
    n_edges = G.number_of_edges()

    info_text = (
        f"Thread ID: {thread_id}\n"
        f"Label: RUMOR\n"
        f"Thread size: {n_posts} posts\n"
        f"Max depth: {max_depth}\n"
        f"Reply edges: {n_edges}"
    )
    ax.text(
        sx + 2.0, sy + 1.0, info_text,
        fontsize=10, fontfamily="DejaVu Sans",
        ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF5F5",
                  edgecolor="#E53E3E", linewidth=1.5),
        zorder=20,
    )

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="#E53E3E", edgecolor="#333333",
                       label="SourcePost (depth=0)"),
        plt.Circle((0, 0), 0.1, facecolor="#90CDF4", edgecolor="#333333",
                   label="ReplyPost (depth=1)"),
        plt.Circle((0, 0), 0.1, facecolor="#63B3ED", edgecolor="#333333",
                   label="ReplyPost (depth=2)"),
        plt.Circle((0, 0), 0.1, facecolor="#4299E1", edgecolor="#333333",
                   label="ReplyPost (depth=3+)"),
        plt.Line2D([0], [0], color="#CC3333", linewidth=2.5,
                   label="User rumor ratio > 0.5"),
    ]
    ax.legend(
        handles=legend_elements, loc="upper right",
        fontsize=9, framealpha=0.9, title="Ontology Classes",
    )

    ax.set_title(
        "Minh họa Knowledge Graph — Conversation Thread (Rumor)\n"
        "Cấu trúc reply tree và phân loại node theo ontology v2",
        fontsize=13, fontweight="bold", pad=20,
    )

    ax.set_xlim(-5, 6)
    ax.set_ylim(-1, max_depth * 2.5 + 1)
    ax.axis("off")
    plt.tight_layout()
    return fig


def main():
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Data file not found: {DATA_PATH}")
        sys.exit(1)

    print("[INFO] Loading data...")
    df = pd.read_csv(DATA_PATH)

    print("[INFO] Selecting thread for visualization...")
    thread_id, thread_df = select_thread(df)
    print(f"  Selected thread_id: {thread_id}")
    print(f"  Posts in thread: {len(thread_df)}")

    print("[INFO] Building reply graph...")
    G = build_reply_graph(thread_df)
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")

    print("[INFO] Drawing subgraph...")
    fig = draw_sample_subgraph(G, thread_id, thread_df)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] KG sample subgraph saved: {OUTPUT_PATH}")
    print(f"     Size: {os.path.getsize(OUTPUT_PATH):,} bytes")
    print(f"     Thread ID: {thread_id}")
    print(f"     Posts: {len(thread_df)}")
    print(f"     Edges: {G.number_of_edges()}")


if __name__ == "__main__":
    main()