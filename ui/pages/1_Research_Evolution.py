"""
Page 1: Research Evolution
==========================
Shows the evolution from Ontology V1 -> KG V1 -> Hybrid V1
-> Ontology V2 -> KG V2 -> Hybrid V2 with cards and metrics.
"""
import streamlit as st
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from ui.components.data_loader import (
    load_dataset_statistics, load_final_metrics, load_v1_metadata, load_v2_metadata
)

st.title("🔬 Quá trình phát triển nghiên cứu")
st.markdown("---")

# Project Overview
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("""
    ## Rumor Detection on Social Media sử dụng Ontology và Knowledge Graph
    
    Nghiên cứu này khảo sát việc tích hợp **Knowledge Graph Features**
    với **Semantic Embeddings** để cải thiện phát hiện tin đồn trên mạng xã hội.
    
    **Đóng góp chính**: Kết hợp Graph Features cấu trúc (Centrality, User Reputation,
    Thread Topology) với MiniLM Embeddings để phân loại Hybrid.
    """)

with col2:
    stats = load_dataset_statistics()
    if stats:
        st.metric("Kích thước dữ liệu", f"{stats.get('Number of Tweets', 'N/A')}")
        st.metric("Số cuộc hội thoại", f"{stats.get('Number of Conversation Threads', 'N/A')}")
        st.metric("Người dùng", f"{stats.get('Number of Users', 'N/A')}")

st.markdown("---")

# Evolution Timeline
st.subheader("📊 Lộ trình phát triển nghiên cứu")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("### Giai đoạn 1: Baseline\n\n"
            "**TF-IDF**\n\n"
            "- Logistic Regression\n"
            "- 5,000 TF-IDF Features\n"
            "- Recall: ~61%")

with col2:
    st.warning("### Giai đoạn 2: Tích hợp Graph\n\n"
               "**KG Feature Extraction**\n\n"
               "- PHEME Ontology V1/V2\n"
               "- Xây dựng Reply Graph\n"
               "- 14 Graph Topology Features\n"
               "- Cải thiện Recall: +30.4%")

with col3:
    st.success("### Giai đoạn 3: Hybrid hoàn chỉnh\n\n"
               "**MiniLM + Graph + Propagation**\n\n"
               "- Sentence Transformer (384 chiều)\n"
               "- Feature Fusion (402 chiều)\n"
               "- Accuracy 98.11%\n"
               "- Rumor Recall 96.46%")

st.markdown("---")

# Pipeline Architecture
st.subheader("🏗️ Kiến trúc Pipeline hoàn chỉnh")

st.markdown("""
```
Dữ liệu PHEME thô (5 Sự kiện, 5,802 Threads, 103,212 Tweets)
    │
    ├── Preprocessing Pipeline
    │   ├── Phân tích Thread (Source + Reactions)
    │   ├── Chuẩn hóa & làm sạch
    │   └── Trích xuất Propagation Features
    │
    ├── Knowledge Graph Construction
    │   ├── Ontology V1 (Classes: Post, User, Event, Thread)
    │   ├── Ontology V2 (Subclasses: SourcePost, ReplyPost)
    │   ├── Sinh RDF Triples (76,066 nodes, 65,565 edges)
    │   └── Phát hiện & loại bỏ Cycle
    │
    ├── Graph Feature Extraction
    │   ├── Node Centrality (PageRank, Degree, Betweenness)
    │   ├── User Reputation (Rumor Ratio, Post Count)
    │   ├── Thread Structure (Subtree Size, Sibling Count)
    │   └── Source Authority (Credibility, Network Size)
    │
    ├── Semantic Embedding (MiniLM)
    │   └── 384 chiều Sentence Embeddings
    │
    └── Hybrid Classifier
        ├── Feature Fusion (402 hoặc 5,398 chiều)
        ├── Logistic Regression
        └── Kết quả: Rumor / Non-Rumor
```
""")

st.markdown("---")

# Performance Summary
st.subheader("📈 Sự tiến hóa về hiệu năng")

metrics_df = load_final_metrics()
if metrics_df is not None:
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Baseline Recall", f"{metrics_df['Recall'].iloc[0]:.1%}")
    with col2:
        best_recall = metrics_df['Recall'].max()
        best_model = metrics_df.loc[metrics_df['Recall'].idxmax(), 'Model']
        st.metric("Recall tốt nhất", f"{best_recall:.1%}", delta=f"+{best_recall - metrics_df['Recall'].iloc[0]:.1%}")
    with col3:
        st.metric("Accuracy tốt nhất", f"{metrics_df['Accuracy'].max():.1%}")
    with col4:
        fn_reduction = metrics_df['FN'].iloc[0] - metrics_df['FN'].iloc[-1]
        st.metric("Giảm FN", f"{fn_reduction:,}", delta=f"-{fn_reduction:,}")
else:
    st.info("Chạy Pipeline huấn luyện để tạo final_metrics_table.csv")

st.markdown("---")
st.caption("Rumor Detection Thesis — Trường Đại học Nha Trang (NTU)")
