#!/usr/bin/env python3
"""
Plot A: Ontology Class Hierarchy Diagram (PHEME v2)
Output: docs/figures/ontology_class_hierarchy.png
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx


OUTPUT_PATH = os.path.join("docs", "figures", "ontology_class_hierarchy.png")


def build_hierarchy_graph():
    """Build a directed graph representing the ontology class hierarchy."""
    G = nx.DiGraph()

    nodes = {
        "owl:Thing":             {"color": "#E8E8E8", "props": ""},
        "Post":                  {"color": "#4A90D9", "props": "(12 props)"},
        "SourcePost":            {"color": "#2B6CB0", "props": "(inherited)"},
        "ReplyPost":             {"color": "#2B6CB0", "props": "(inherited)"},
        "User":                  {"color": "#38A169", "props": "(7 props)"},
        "ConversationThread":    {"color": "#DD6B20", "props": "(4 props)"},
        "Event":                 {"color": "#805AD5", "props": "(3 props)"},
        "VeracityLabel":         {"color": "#FC8181", "props": "(0 props)"},
        "Rumor":                 {"color": "#E53E3E", "props": ""},
        "NonRumor":              {"color": "#68D391", "props": ""},
    }

    for node, attrs in nodes.items():
        G.add_node(node, color=attrs["color"], props=attrs["props"])

    edges = [
        ("owl:Thing", "Post", "solid"),
        ("owl:Thing", "User", "solid"),
        ("owl:Thing", "ConversationThread", "solid"),
        ("owl:Thing", "Event", "solid"),
        ("owl:Thing", "VeracityLabel", "solid"),
        ("Post", "SourcePost", "dashed"),
        ("Post", "ReplyPost", "dashed"),
        ("VeracityLabel", "Rumor", "dotted"),
        ("VeracityLabel", "NonRumor", "dotted"),
    ]

    for src, tgt, style in edges:
        G.add_edge(src, tgt, style=style)

    return G


def draw_hierarchy(G):
    """Draw the class hierarchy diagram with clean layout."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    fig.patch.set_facecolor("white")

    # --- Manual layout: tree from top to bottom ---
    pos = {}
    pos["owl:Thing"] = (0, 4.5)

    # Level 1
    pos["VeracityLabel"] = (-5, 2.5)
    pos["Post"] = (-2.5, 2.5)
    pos["ConversationThread"] = (0, 2.5)
    pos["User"] = (2.5, 2.5)
    pos["Event"] = (5, 2.5)

    # Level 2: subclasses of Post
    pos["SourcePost"] = (-3.5, 0.5)
    pos["ReplyPost"] = (-1.5, 0.5)

    # Level 3: instances
    pos["Rumor"] = (-5.5, -1.0)
    pos["NonRumor"] = (-4.5, -1.0)

    # --- Draw edges with proper arrows ---
    for src, tgt, data in G.edges(data=True):
        style = data.get("style", "solid")
        x1, y1 = pos[src]
        x2, y2 = pos[tgt]

        if style == "dashed":
            linestyle = "dashed"
            linewidth = 2.0
            color = "#999999"
        elif style == "dotted":
            linestyle = "dotted"
            linewidth = 1.5
            color = "#BBBBBB"
        else:
            linestyle = "solid"
            linewidth = 2.5
            color = "#666666"

        ax.annotate(
            "",
            xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="->",
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                connectionstyle="arc3,rad=0.0",
            ),
            zorder=1,
        )

    # --- Edge labels ---
    edge_labels = {
        ("owl:Thing", "Post"): ("rdfs:subClassOf", -0.8, 0.15),
        ("owl:Thing", "User"): ("rdfs:subClassOf", 0.8, 0.15),
        ("owl:Thing", "ConversationThread"): ("rdfs:subClassOf", 0.8, 0.15),
        ("owl:Thing", "Event"): ("rdfs:subClassOf", 1.0, 0.15),
        ("owl:Thing", "VeracityLabel"): ("rdfs:subClassOf", -1.0, 0.15),
        ("Post", "SourcePost"): ("rdfs:subClassOf", -1.0, 0.15),
        ("Post", "ReplyPost"): ("rdfs:subClassOf", 1.0, 0.15),
        ("VeracityLabel", "Rumor"): ("rdf:type", -1.2, -0.15),
        ("VeracityLabel", "NonRumor"): ("rdf:type", 1.2, -0.15),
    }

    for (src, tgt), (label, off_x, off_y) in edge_labels.items():
        x1, y1 = pos[src]
        x2, y2 = pos[tgt]
        mx, my = (x1 + x2) / 2 + off_x, (y1 + y2) / 2 + off_y
        ax.text(
            mx, my, label, fontsize=8, color="#555555",
            ha="center", va="center", style="italic",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor="none", alpha=0.85),
            zorder=20,
        )

    # --- Draw nodes ---
    node_height = 0.45
    for node, (x, y) in pos.items():
        color = G.nodes[node]["color"]
        props = G.nodes[node]["props"]

        if node in ("Rumor", "NonRumor"):
            # Instance: ellipse, smaller
            w, h = 0.8, 0.35
            ellipse = plt.matplotlib.patches.Ellipse(
                (x, y), w, h,
                facecolor=color, edgecolor="#333333", linewidth=1.5,
                zorder=5,
            )
            ax.add_patch(ellipse)
            label_text = node
        else:
            # Class: rounded rectangle
            w, h = 1.6, node_height
            if node == "owl:Thing":
                w, h = 1.2, node_height
            rect = mpatches.FancyBboxPatch(
                (x - w / 2, y - h / 2), w, h,
                boxstyle="round,pad=0.08",
                facecolor=color, edgecolor="#333333", linewidth=1.5,
                zorder=5,
            )
            ax.add_patch(rect)
            if props:
                label_text = f"{node}\n{props}"
            else:
                label_text = node

        # Text inside node
        text_color = (
            "white"
            if color in ("#4A90D9", "#2B6CB0", "#38A169", "#DD6B20",
                         "#805AD5", "#E53E3E")
            else "#333333"
        )
        ax.text(
            x, y, label_text, fontsize=10, fontweight="bold",
            ha="center", va="center", color=text_color, zorder=6,
        )

    # --- Legend ---
    legend_elements = [
        mpatches.Patch(facecolor="#E8E8E8", edgecolor="#333333",
                       label="owl:Thing (root class)"),
        mpatches.Patch(facecolor="#4A90D9", edgecolor="#333333",
                       label="Lớp (Class)"),
        mpatches.Patch(facecolor="#FC8181", edgecolor="#333333",
                       label="Lớp VeracityLabel"),
        plt.Line2D([0], [0], color="#666666", linewidth=2.5, linestyle="solid",
                   label="rdfs:subClassOf"),
        plt.Line2D([0], [0], color="#999999", linewidth=2.0, linestyle="dashed",
                   label="rdfs:subClassOf (subclass)"),
        plt.Line2D([0], [0], color="#BBBBBB", linewidth=1.5, linestyle="dotted",
                   label="rdf:type (instance)"),
    ]
    ax.legend(
        handles=legend_elements, loc="lower right",
        fontsize=9, framealpha=0.9,
    )

    # --- Title ---
    ax.set_title(
        "Phân cấp lớp Ontology PHEME v2\n(Class Hierarchy)",
        fontsize=15, fontweight="bold", pad=20,
    )

    ax.set_xlim(-7, 7)
    ax.set_ylim(-2, 5.5)
    ax.axis("off")
    plt.tight_layout()
    return fig


def main():
    G = build_hierarchy_graph()
    fig = draw_hierarchy(G)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Class hierarchy diagram saved: {OUTPUT_PATH}")
    print(f"     Size: {os.path.getsize(OUTPUT_PATH):,} bytes")


if __name__ == "__main__":
    main()