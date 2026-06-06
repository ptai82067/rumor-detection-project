#!/usr/bin/env python3
"""
viz_kg_semantic_final.py -- Publication-quality KG Semantic Visualization
====================================================================
Output:
  - visualization/kg_semantic_final.png (300 dpi)
  - visualization/kg_semantic_final.svg
  - visualization/kg_viz_log.txt

Ontology v2 relations used (7):
  belongsToEvent, hasSourcePost, hasVeracity, postedBy,
  repliesTo, participatesInThread, inThread
"""

import os, sys, warnings, datetime
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

GRAPH_FEATURES_PATH = os.path.join("data", "processed", "graph_features_v2.csv")
POST_DATA_PATH = os.path.join("data", "processed", "pheme_features_with_graph.csv")
OUTPUT_PNG = os.path.join("visualization", "kg_semantic_final.png")
OUTPUT_SVG = os.path.join("visualization", "kg_semantic_final.svg")
LOG_PATH = os.path.join("visualization", "kg_viz_log.txt")

def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full = "[%s] %s" % (ts, msg)
    safe = full.encode("ascii", errors="replace").decode("ascii")
    print(safe, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(full + "\n")

def select_thread(gf, post_df):
    labels = post_df.groupby("thread_id")["label"].first().reset_index()
    gf_labeled = gf.merge(labels, on="thread_id", how="inner")
    rumor = gf_labeled[gf_labeled["label"] == 1]
    conditions = [
        ("ALL criteria",
         [("num_nodes","ge",8),("num_nodes","le",25),
          ("source_reply_count","ge",3),("user_rumor_ratio","gt",0.3),("thread_depth","ge",2)]),
        ("source_reply_count>=2",
         [("num_nodes","ge",8),("num_nodes","le",25),
          ("source_reply_count","ge",2),("user_rumor_ratio","gt",0.3),("thread_depth","ge",2)]),
        ("user_rumor_ratio>0.2",
         [("num_nodes","ge",8),("num_nodes","le",25),
          ("source_reply_count","ge",2),("user_rumor_ratio","gt",0.2),("thread_depth","ge",2)]),
        ("thread_depth>=1",
         [("num_nodes","ge",8),("num_nodes","le",25),
          ("source_reply_count","ge",2),("user_rumor_ratio","gt",0.2),("thread_depth","ge",1)]),
    ]
    for label, conds in conditions:
        mask = pd.Series(True, index=rumor.index)
        for col_name, op, val in conds:
            if op == "ge": mask &= rumor[col_name] >= val
            elif op == "le": mask &= rumor[col_name] <= val
            elif op == "gt": mask &= rumor[col_name] > val
        candidates = rumor[mask]
        log("Candidate filter '%s': %d threads" % (label, len(candidates)))
        if len(candidates) > 0:
            tid_raw = candidates.iloc[0]["thread_id"]
            trow = candidates.iloc[0]
            tid_int = int(tid_raw)
            tposts = post_df[post_df["thread_id"] == tid_int].copy()
            log("  Selected thread_id=%d, criteria='%s'" % (tid_int, label))
            return tid_int, trow, tposts, label
    for tid_raw in rumor["thread_id"].values:
        tid_int = int(tid_raw)
        tposts = post_df[post_df["thread_id"] == tid_int]
        if len(tposts) >= 5:
            trow = rumor[rumor["thread_id"] == tid_raw].iloc[0]
            log("  Fallback: thread_id=%d" % tid_int)
            return tid_int, trow, tposts, "fallback"
    raise RuntimeError("No suitable thread found.")

def build_ontology_graph(tid, trow, tposts):
    G = nx.DiGraph()
    src = tposts[tposts["is_source"] == True]
    if len(src) == 0:
        src = tposts[(tposts["depth"] == 0) & (tposts["reply_to"].isna())]
    if len(src) == 0:
        src = tposts[tposts["depth"] == 0].head(1)
    source_row = src.iloc[0]
    src_post_id = int(source_row["post_id"])
    src_user_id = int(source_row["user_id"])
    replies = tposts[tposts["post_id"] != src_post_id].copy()
    replies = replies.sort_values(["depth", "post_id"])
    depth1 = replies[replies["depth"] == 1].head(3)
    depth2 = replies[replies["depth"] == 2].head(1)
    depth3 = replies[replies["depth"] >= 3].head(1)
    selected_replies = pd.concat([depth1, depth2, depth3]).head(4)
    if len(selected_replies) < 3:
        extra = replies[~replies.index.isin(selected_replies.index)].head(4 - len(selected_replies))
        selected_replies = pd.concat([selected_replies, extra]).head(4)
    selected_user_ids = [src_user_id]
    for _, r in selected_replies.iterrows():
        uid = int(r["user_id"])
        if uid not in selected_user_ids:
            selected_user_ids.append(uid)
    selected_user_ids = selected_user_ids[:3]
    event_id = source_row["event_id"]
    event_name = str(event_id) if pd.notna(event_id) else "unknown"
    num_nodes_val = int(trow["num_nodes"])
    thread_depth_val = int(trow["thread_depth"])
    src_pagerank = float(trow["source_pagerank"])
    G.add_node("Event", type="event", display="Event\n" + event_name)
    G.add_node("Thread", type="thread", display="ConversationThread\nsize=%d, depth=%d" % (num_nodes_val, thread_depth_val))
    G.add_node("VeracityLabel", type="veracity", display="VeracityLabel\nRumor (label=1)")
    G.add_node("SourcePost", type="source", display="SourcePost\npagerank=%.3f" % src_pagerank)
    reply_nodes = {}
    for i, (_, rrow) in enumerate(selected_replies.iterrows()):
        node_name = "ReplyPost_%d" % (i + 1)
        depth_val = int(rrow["depth"])
        reply_nodes[node_name] = rrow
        G.add_node(node_name, type="reply", display="ReplyPost\ndepth=%d" % depth_val)
    user_nodes = {}
    for i, uid in enumerate(selected_user_ids):
        node_name = "User_%d" % i
        user_posts = tposts[tposts["user_id"] == uid]
        u_ratio = float(user_posts.iloc[0].get("user_prior_rumor_ratio", 0))
        user_nodes[node_name] = {"id": uid, "ratio": u_ratio}
        G.add_node(node_name, type="user", display="User\nrumor_ratio=%.2f" % u_ratio)
    G.add_edge("Thread", "Event", relation="belongsToEvent")
    G.add_edge("Thread", "SourcePost", relation="hasSourcePost")
    G.add_edge("Thread", "VeracityLabel", relation="hasVeracity")
    G.add_edge("SourcePost", "User_0", relation="postedBy")
    for rn, rrow in reply_nodes.items():
        ruid = int(rrow["user_id"])
        target_user = None
        for un, ud in user_nodes.items():
            if ud["id"] == ruid:
                target_user = un; break
        if target_user is None: target_user = "User_0"
        G.add_edge(rn, target_user, relation="postedBy")
    for rn, rrow in reply_nodes.items():
        reply_to_val = rrow["reply_to"]
        if pd.isna(reply_to_val):
            G.add_edge(rn, "SourcePost", relation="repliesTo")
        else:
            reply_to_int = int(reply_to_val)
            if reply_to_int == src_post_id:
                G.add_edge(rn, "SourcePost", relation="repliesTo")
            else:
                found = False
                for rn2, rrow2 in reply_nodes.items():
                    if rn2 == rn: continue
                    if int(rrow2["post_id"]) == reply_to_int:
                        G.add_edge(rn, rn2, relation="repliesTo"); found = True; break
                if not found:
                    G.add_edge(rn, "SourcePost", relation="repliesTo")
    for un in user_nodes:
        G.add_edge(un, "Thread", relation="participatesInThread")
    G.add_edge("SourcePost", "Thread", relation="inThread")
    for rn in reply_nodes:
        G.add_edge(rn, "Thread", relation="inThread")
    return G, reply_nodes, user_nodes, event_name


# ============================================================
# FINAL LAYOUT — tuned vertical spacing, wider horizontal spread
#   Tier 1 (y=0.95): Event
#   Tier 2 (y=0.62): ConversationThread
#   Tier 3 (y=0.30): SourcePost (-0.40)  VeracityLabel (+0.40)
#   Tier 4 (y=-0.08): ReplyPosts spread -0.48 to +0.48
#   Tier 5 (y=-0.38): Users aligned
# ============================================================
def fixed_layout():
    return {
        "Event":             ( 0.00,  0.95),
        "Thread":            ( 0.00,  0.62),
        "SourcePost":        (-0.40,  0.30),
        "VeracityLabel":     ( 0.40,  0.30),
        "ReplyPost_1":       (-0.48, -0.08),
        "ReplyPost_2":       (-0.16, -0.08),
        "ReplyPost_3":       ( 0.16, -0.08),
        "ReplyPost_4":       ( 0.48, -0.08),
        "User_0":            (-0.48, -0.38),
        "User_1":            (-0.16, -0.38),
        "User_2":            ( 0.30, -0.38),
    }

NODE_COLORS = {
    "event":"#DCD5C4","thread":"#B8CCE0","source":"#B8D4B8",
    "reply":"#D0E0C8","user":"#E0D5C0","veracity":"#E8B8B8"}
NODE_EDGE_COLORS = {
    "event":"#B0A896","thread":"#8899AA","source":"#6A9A6A",
    "reply":"#88AA80","user":"#B8A888","veracity":"#CC8888"}
RELATION_COLORS = {
    "belongsToEvent":"#888888","hasSourcePost":"#777777","hasVeracity":"#BB6666",
    "postedBy":"#779977","repliesTo":"#6699BB","participatesInThread":"#996699","inThread":"#AA8844"}
VISIBLE_LABELS = {"belongsToEvent", "hasSourcePost", "hasVeracity", "repliesTo"}

def draw_graph(G, pos, trow, event_name):
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 8
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # EDGES
    for u, v, d in G.edges(data=True):
        rel = d.get("relation", "unknown")
        if u not in pos or v not in pos: continue
        x1, y1 = pos[u]; x2, y2 = pos[v]
        color = RELATION_COLORS.get(rel, "#999999")
        curve = "arc3,rad=0.12" if rel == "repliesTo" else "arc3,rad=0.0"
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color=color, linewidth=0.8,
                          connectionstyle=curve, shrinkA=12, shrinkB=12), zorder=1)
        if rel not in VISIBLE_LABELS: continue
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = x2 - x1, y2 - y1
        length = max(0.01, (dx*dx + dy*dy)**0.5)
        ox, oy = -dy / length * 0.030, dx / length * 0.030
        ax.text(mx + ox, my + oy, rel, fontsize=6, ha="center", va="center",
                color="#555555", fontstyle="italic",
                bbox=dict(boxstyle="round,pad=0.06", facecolor="white", edgecolor="none", alpha=0.85), zorder=5)

    # NODES — sizes reduced ~25% width, ~20% height
    for node in list(pos.keys()):
        if node not in G.nodes(): continue
        x, y = pos[node]
        ndata = G.nodes[node]
        ntype = ndata["type"]
        display_text = ndata["display"]
        sz = {
            "event": (0.10, 0.055), "thread": (0.13, 0.065),
            "source": (0.10, 0.055), "veracity": (0.10, 0.055),
            "reply": (0.09, 0.048), "user": (0.10, 0.048)
        }.get(ntype, (0.09, 0.048))
        w, h = sz
        rect = mpatches.FancyBboxPatch((x - w, y - h), w * 2, h * 2,
            boxstyle="round,pad=0.03",
            facecolor=NODE_COLORS.get(ntype, "#CCCCCC"),
            edgecolor=NODE_EDGE_COLORS.get(ntype, "#888888"),
            linewidth=0.7, zorder=10)
        ax.add_patch(rect)
        lines = display_text.split("\n")
        line1 = lines[0]
        line2 = lines[1] if len(lines) > 1 else ""
        ax.text(x, y + 0.012, line1, fontsize=6.5 if len(line1) < 20 else 5.5,
                ha="center", va="center", color="#222222", fontweight="bold", zorder=11)
        if line2:
            ax.text(x, y - 0.022, line2, fontsize=5 if len(line2) < 30 else 4.5,
                    ha="center", va="center", color="#555555", zorder=11)

    # FEATURE BOX — far bottom-right, detached
    sp = float(trow["source_pagerank"])
    ur = float(trow["user_rumor_ratio"])
    td = int(trow["thread_depth"])
    ad = float(trow["avg_depth"])
    src = int(trow["source_reply_count"])
    uq = int(trow["unique_users"])
    nn = int(trow["num_nodes"])
    info_lines = [
        "Graph features from this subgraph:",
        "  source_pagerank    = %.4f" % sp,
        "  user_rumor_ratio   = %.3f" % ur,
        "  thread_depth       = %d" % td,
        "  avg_depth          = %.2f" % ad,
        "  source_reply_cnt   = %d" % src,
        "  unique_users       = %d" % uq,
        "  thread_size        = %d" % nn,
        "",
        "These signals require KG",
        "structure, not flat tables.",
    ]
    ax.text(1.02, 0.02, "\n".join(info_lines), fontsize=6.5,
            ha="left", va="bottom", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FCFCFA",
                      edgecolor="#BBBBBB", linewidth=0.5), zorder=20)

    # LEGEND
    node_legend = [
        mpatches.Patch(facecolor=NODE_COLORS["event"], edgecolor=NODE_EDGE_COLORS["event"], label="Event"),
        mpatches.Patch(facecolor=NODE_COLORS["thread"], edgecolor=NODE_EDGE_COLORS["thread"], label="ConversationThread"),
        mpatches.Patch(facecolor=NODE_COLORS["veracity"], edgecolor=NODE_EDGE_COLORS["veracity"], label="VeracityLabel"),
        mpatches.Patch(facecolor=NODE_COLORS["source"], edgecolor=NODE_EDGE_COLORS["source"], label="SourcePost"),
        mpatches.Patch(facecolor=NODE_COLORS["reply"], edgecolor=NODE_EDGE_COLORS["reply"], label="ReplyPost"),
        mpatches.Patch(facecolor=NODE_COLORS["user"], edgecolor=NODE_EDGE_COLORS["user"], label="User"),
    ]
    rel_legend = [
        plt.Line2D([0],[0],color=RELATION_COLORS["belongsToEvent"],linewidth=0.8,label="belongsToEvent"),
        plt.Line2D([0],[0],color=RELATION_COLORS["hasSourcePost"],linewidth=0.8,label="hasSourcePost"),
        plt.Line2D([0],[0],color=RELATION_COLORS["hasVeracity"],linewidth=0.8,label="hasVeracity"),
        plt.Line2D([0],[0],color=RELATION_COLORS["postedBy"],linewidth=0.8,label="postedBy"),
        plt.Line2D([0],[0],color=RELATION_COLORS["repliesTo"],linewidth=0.8,label="repliesTo"),
        plt.Line2D([0],[0],color=RELATION_COLORS["participatesInThread"],linewidth=0.8,label="participatesInThread"),
        plt.Line2D([0],[0],color=RELATION_COLORS["inThread"],linewidth=0.8,label="inThread"),
    ]
    legend1 = ax.legend(handles=node_legend+rel_legend, loc="lower center",
        fontsize=6.5, framealpha=0.9,
        title="Node Types (v2)                     Relation Types (Object Properties)",
        title_fontsize=7.5, ncol=7, bbox_to_anchor=(0.5, -0.12),
        handlelength=1.0, handletextpad=0.3, columnspacing=0.6)
    ax.add_artist(legend1)

    # TITLE
    ax.set_title(
        "Knowledge Graph Semantic Subgraph (PHEME Dataset)\n"
        "Thread %d | Event: %s | Label: RUMOR | %d posts, depth %d, %d direct replies"
        % (int(trow["thread_id"]), event_name, nn, td, src),
        fontsize=11, fontweight="bold", pad=10)

    ax.set_xlim(-0.72, 0.72)
    ax.set_ylim(-0.52, 1.10)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.subplots_adjust(bottom=0.16, top=0.92, left=0.03, right=0.88)
    return fig

def main():
    os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("KG Semantic Final Visualization - Debug Log\n")
        f.write("=" * 60 + "\n")
    log("=" * 60)
    log("Starting KG Semantic Final Visualization (final tuning)")
    log("=" * 60)
    log("Loading graph_features_v2.csv...")
    gf = pd.read_csv(GRAPH_FEATURES_PATH)
    log("  %d threads" % len(gf))
    log("Loading pheme_features_with_graph.csv...")
    post_df = pd.read_csv(POST_DATA_PATH)
    log("  %d posts" % len(post_df))
    log("Selecting thread...")
    tid, trow, tposts, criteria = select_thread(gf, post_df)
    log(""); log("=== SELECTED THREAD ===")
    log("thread_id: %d" % tid)
    log("criteria: %s" % criteria)
    log("num_nodes: %d" % int(trow["num_nodes"]))
    log("thread_depth: %d" % int(trow["thread_depth"]))
    log("source_reply_count: %d" % int(trow["source_reply_count"]))
    log("user_rumor_ratio: %.6f" % float(trow["user_rumor_ratio"]))
    log("source_pagerank: %.8f" % float(trow["source_pagerank"]))
    log("avg_depth: %.6f" % float(trow["avg_depth"]))
    log("unique_users: %d" % int(trow["unique_users"]))
    log(""); log("=== SAMPLE POSTS (first 5) ===")
    sample = tposts.sort_values(["depth", "post_id"]).head(5)
    for _, r in sample.iterrows():
        reply_str = str(int(r["reply_to"])) if pd.notna(r["reply_to"]) else "NaN"
        log("  post_id=%d reply_to=%s depth=%d is_source=%s user_id=%d" % (
            int(r["post_id"]), reply_str, int(r["depth"]),
            str(bool(r["is_source"])), int(r["user_id"])))
    log(""); log("Building ontology graph...")
    G, reply_nodes, user_nodes, event_name = build_ontology_graph(tid, trow, tposts)
    log("  Nodes: %d, Edges: %d" % (G.number_of_nodes(), G.number_of_edges()))
    present_relations = set(d["relation"] for _, _, d in G.edges(data=True))
    expected = {"belongsToEvent","hasSourcePost","hasVeracity",
                "postedBy","repliesTo","participatesInThread","inThread"}
    missing = expected - present_relations
    if missing: log("  WARNING: Missing relations: %s" % missing)
    else: log("  All 7 ontology v2 relations present [OK]")
    log(""); log("Drawing figure...")
    pos = fixed_layout()
    fig = draw_graph(G, pos, trow, event_name)
    log("Exporting PNG...")
    plt.tight_layout(pad=2.0)
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    png_size = os.path.getsize(OUTPUT_PNG)
    log("  PNG: %s (%s bytes)" % (OUTPUT_PNG, "{:,}".format(png_size)))
    log("Exporting SVG...")
    fig.savefig(OUTPUT_SVG, bbox_inches="tight")
    svg_size = os.path.getsize(OUTPUT_SVG)
    log("  SVG: %s (%s bytes)" % (OUTPUT_SVG, "{:,}".format(svg_size)))
    plt.close(fig)
    log(""); log("=" * 60); log("COMPLETE"); log("=" * 60)
    print(""); print("[DONE]", flush=True)

if __name__ == "__main__":
    main()