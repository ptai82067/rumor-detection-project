#!/usr/bin/env python3
"""
Plot B: Ontology Full Diagram — Classes and Object Properties (PHEME v2)
Output: docs/figures/ontology_full_diagram.png
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx


OUTPUT_PATH = os.path.join("docs", "figures", "ontology_full_diagram.png")


def build_ontology_graph():
    """Build the ontology graph with classes and object properties."""
    G = nx.DiGraph()

    nodes = {
        "Post":                {"color": "#4A90D9"},
        "SourcePost":          {"color": "#2B6CB0"},
        "ReplyPost":           {"color": "#2B6CB0"},
        "User":                {"color": "#38A169"},
        "ConversationThread":  {"color": "#DD6B20"},
        "Event":               {"color": "#805AD5"},
        "VeracityLabel":       {"color": "#FC8181"},
    }

    for node, attrs in nodes.items():
        G.add_node(node, color=attrs["color"])

    edges = [
        ("Post", "User", "postedBy"),
        ("Post", "Event", "aboutEvent"),
        ("Post", "Post", "repliesTo"),
        ("Post", "ConversationThread", "inThread"),
        ("ConversationThread", "VeracityLabel", "hasVeracity"),
        ("ConversationThread", "SourcePost", "hasSourcePost"),
        ("ConversationThread", "Event", "belongsToEvent"),
        ("Post", "ReplyPost", "hasReply"),
        ("ReplyPost", "Post", "hasParentPost"),
        ("User", "ConversationThread", "participatesInThread"),
        ("SourcePost", "Post", "subClassOf"),
        ("ReplyPost", "Post", "subClassOf"),
    ]

    for src, tgt, label in edges:
        G.add_edge(src, tgt, label=label)

    return G


def draw_ontology_diagram(G):
    """Draw the full ontology diagram with clean layout."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    fig.patch.set_facecolor("white")

    # Fixed positions
    pos = {
        "Post":               (0, 0),
        "SourcePost":         (-2.5, -2.5),
        "ReplyPost":          (2.5, -2.5),
        "User":               (-4, 1.5),
        "ConversationThread": (4, 1.5),
        "Event":              (0, 4),
        "VeracityLabel":      (6, -0.5),
    }

    # --- Draw edges ---
    edge_style = {
        "subClassOf": "dashed",
    }
    default_style = "solid"
    default_color = "#555555"
    dashed_color = "#999999"

    drawn_edges = set()
    for src, tgt, data in G.edges(data=True):
        label = data.get("label", "")
        edge_key = (src, tgt, label)
        if edge_key in drawn_edges:
            continue
        drawn_edges.add(edge_key)

        x1, y1 = pos[src]
        x2, y2 = pos[tgt]

        is_self_loop = (src == tgt)
        is_subclass = (label == "subClassOf")
        linestyle = edge_style.get(label, default_style)
        linecolor = dashed_color if is_subclass else default_color

        if is_self_loop:
            # Self-loop above node
            rad = 0.5
            ax.annotate(
                "",
                xy=(x1 + 0.2, y1 + 0.8), xytext=(x1 - 0.2, y1 + 0.8),
                arrowprops=dict(
                    arrowstyle="->", color=linecolor, linewidth=1.8,
                    connectionstyle=f"arc3,rad={rad}",
                ),
                zorder=3,
            )
            ax.text(
                x1, y1 + 1.0, label, fontsize=9, ha="center", va="bottom",
                color="#444444", style="italic", fontweight="bold", zorder=20,
            )
        else:
            rad = 0.0
            # Check for bidirectional edges
            if G.has_edge(tgt, src):
                rad = 0.3
            if src in ("ConversationThread",) and tgt in ("Event", "SourcePost", "VeracityLabel"):
                rad = -0.2

            ax.annotate(
                "",
                xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="->", color=linecolor,
                    linestyle=linestyle, linewidth=1.8,
                    connectionstyle=f"arc3,rad={rad}",
                ),
                zorder=3,
            )

            # Edge label at midpoint
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            offset_x = 0.35 if x1 != x2 else 0.3
            offset_y = 0.2
            if rad != 0:
                offset_y = -0.2

            ax.text(
                mx + offset_x, my + offset_y, label,
                fontsize=9, ha="center", va="center",
                color="#444444", style="italic", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.85),
                zorder=20,
            )

    # --- Draw nodes ---
    nw, nh = 1.5, 0.45
    for node, (x, y) in pos.items():
        color = G.nodes[node]["color"]
        rect = mpatches.FancyBboxPatch(
            (x - nw / 2, y - nh / 2), nw, nh,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor="#333333", linewidth=2.0,
            zorder=10,
        )
        ax.add_patch(rect)
        ax.text(
            x, y, node, fontsize=11, fontweight="bold",
            ha="center", va="center", color="white", zorder=11,
        )

    # --- Summary box ---
    summary_text = (
        "Data Properties tổng: 32\n"
        "Object Properties: 11\n"
        "Classes: 7 (trong đó 2 subclass)"
    )
    ax.text(
        5.5, -3.8, summary_text, fontsize=10,
        ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F0F0F0",
                  edgecolor="#888888", linewidth=1.2),
    )

    # --- Legends ---
    legend_classes = [
        mpatches.Patch(facecolor="#4A90D9", edgecolor="#333333", label="Post"),
        mpatches.Patch(facecolor="#2B6CB0", edgecolor="#333333", label="SourcePost / ReplyPost"),
        mpatches.Patch(facecolor="#38A169", edgecolor="#333333", label="User"),
        mpatches.Patch(facecolor="#DD6B20", edgecolor="#333333", label="ConversationThread"),
        mpatches.Patch(facecolor="#805AD5", edgecolor="#333333", label="Event"),
        mpatches.Patch(facecolor="#FC8181", edgecolor="#333333", label="VeracityLabel"),
    ]
    leg1 = ax.legend(
        handles=legend_classes, loc="lower left",
        fontsize=9, framealpha=0.9, title="Classes",
    )
    ax.add_artist(leg1)

    legend_edges = [
        plt.Line2D([0], [0], color="#555555", linewidth=1.8, linestyle="solid",
                   label="Object Property"),
        plt.Line2D([0], [0], color="#999999", linewidth=1.8, linestyle="dashed",
                   label="rdfs:subClassOf"),
    ]
    ax.legend(
        handles=legend_edges, loc="upper left",
        fontsize=9, framealpha=0.9, title="Relations",
    )

    ax.set_title(
        "Sơ đồ Ontology PHEME v2\n(Classes và Object Properties)",
        fontsize=15, fontweight="bold", pad=20,
    )
    ax.set_xlim(-5.5, 7.5)
    ax.set_ylim(-4, 5.5)
    ax.axis("off")
    plt.tight_layout()
    return fig


def main():
    G = build_ontology_graph()
    fig = draw_ontology_diagram(G)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Ontology full diagram saved: {OUTPUT_PATH}")
    print(f"     Size: {os.path.getsize(OUTPUT_PATH):,} bytes")


if __name__ == "__main__":
    main()