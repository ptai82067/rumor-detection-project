"""
Page 2: Rumor Detection
========================
Interactive demo: select model version, thread, and analyze.
Sidebar: model version selector, event/thread selector.
Main area: conversation thread, prediction, confidence, features.
"""
import streamlit as st
import pandas as pd
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from ui.components.data_loader import (
    load_pheme_features, load_graph_features_v2, get_sample_threads,
    get_thread_data, get_v2_features_for_thread
)
from ui.components.model_manager import predict_v2, predict_v1, load_v1_model, load_v2_model

st.title("🕵️ Rumor Detection")
st.markdown("---")

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("Cài đặt dự đoán")

# Model version selector
model_version = st.sidebar.radio(
    "Phiên bản mô hình",
    ["V2 (Thread-Level) — Khuyến nghị", "V1 (Post-Level)"],
    index=0,
    help="V2 sử dụng MiniLM + Propagation + Graph (402 chiều). V1 sử dụng TF-IDF + MiniLM + Graph (5,398 chiều)."
)

is_v2 = "V2" in model_version

# Load samples
samples = get_sample_threads()

if len(samples) == 0:
    st.error("Không tìm thấy mẫu cuộc hội thoại. Chạy Pipeline huấn luyện trước.")
    st.stop()

# Create thread selector
thread_options = {}
for s in samples:
    label_emoji = "🔴" if s['label'] == 'Rumor' else "🟢"
    display = f"{label_emoji} Thread {s['thread_id']} — {s['label']} ({s['num_posts']} bài đăng)"
    thread_options[display] = s

selected_display = st.sidebar.selectbox(
    "Chọn cuộc hội thoại mẫu",
    options=list(thread_options.keys()),
    index=0
)

selected_thread = thread_options[selected_display]
thread_id = selected_thread['thread_id']

# Analyze button
analyze = st.sidebar.button("🔍 Phân tích cuộc hội thoại", type="primary", use_container_width=True)

# ============================================================
# MAIN AREA
# ============================================================
# Thread info header
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Thread ID", f"{thread_id}")
with col2:
    st.metric("Nhãn thật", selected_thread['label'],
              delta="Rumor" if selected_thread['label'] == 'Rumor' else "Non-Rumor",
              delta_color="inverse" if selected_thread['label'] == 'Rumor' else "normal")
with col3:
    st.metric("Bài đăng", selected_thread['num_posts'])
with col4:
    st.metric("Mô hình", "V2 (402 chiều)" if is_v2 else "V1 (5,398 chiều)")

st.markdown(f"**Văn bản nguồn**: {selected_thread['source_text']}")

if analyze:
    thread_posts = get_thread_data(thread_id)

    if thread_posts is None or len(thread_posts) == 0:
        st.error(f"Không tìm thấy dữ liệu cho Thread {thread_id}")
        st.stop()

    with st.spinner("Đang thực hiện dự đoán..."):
        if is_v2:
            features = get_v2_features_for_thread(thread_id)
            if features is None:
                st.error("Không tìm thấy đặc trưng cho cuộc hội thoại này.")
                st.stop()

            graph_features, prop_features = features
            source_text = str(thread_posts[thread_posts['is_source']]['text'].iloc[0]) if any(thread_posts['is_source']) else str(thread_posts['text'].iloc[0])

            pred, confidence, inference_time, n_features = predict_v2(
                thread_id, graph_features, prop_features, source_text
            )
        else:
            # V1: use the source post
            source_post = thread_posts[thread_posts['is_source']].iloc[0] if any(thread_posts['is_source']) else thread_posts.iloc[0]
            post_id = int(source_post['post_id'])
            text = str(source_post['text'])

            pred, confidence, inference_time, n_features = predict_v1(post_id, text)

    if pred is None:
        st.warning("⚠️ Không tìm thấy file mô hình. Chạy `scripts/train_and_save_v1.py` và/hoặc `scripts/train_and_save_v2.py` trước.")
        st.stop()

    # ============================================================
    # RESULTS
    # ============================================================
    st.markdown("---")
    st.subheader("📊 Kết quả dự đoán")

    # Result cards
    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:
        prediction_label = "🔴 Rumor" if pred == 1 else "🟢 Non-Rumor"
        st.markdown(f"### Kết quả dự đoán\n# {prediction_label}")

    with result_col2:
        st.markdown(f"### Độ tin cậy\n# {confidence:.2%}")
        st.progress(float(confidence))

    with result_col3:
        st.markdown(f"### Chi tiết\n"
                    f"**Đặc trưng**: {n_features:,} chiều\n"
                    f"**Thời gian suy luận**: {inference_time:.1f} ms\n"
                    f"**Mô hình**: {'V2 (Thread-Level)' if is_v2 else 'V1 (Post-Level)'}")

    # ============================================================
    # CONVERSATION THREAD
    # ============================================================
    st.markdown("---")
    st.subheader("💬 Cuộc hội thoại")

    thread_posts_sorted = thread_posts.sort_values(['depth', 'time'] if 'time' in thread_posts.columns else ['depth', 'post_id'])

    for _, post in thread_posts_sorted.iterrows():
        depth = int(post['depth'])
        is_source = bool(post['is_source']) if 'is_source' in post else (depth == 0)
        indent = " " * depth

        if is_source:
            st.markdown(f"**{indent}📌 Bài đăng gốc** (depth={depth})")
        else:
            parent_info = ""
            if 'reply_to' in post and pd.notna(post['reply_to']):
                parent_info = f" ↳ trả lời {int(float(post['reply_to']))}"
            st.markdown(f"**{indent}💬 Trả lời**{parent_info} (depth={depth})")

        st.markdown(f"{indent}> {str(post['text'][:200])}")

        if depth >= 3:
            st.markdown(f"{indent}*... (các trả lời sâu hơn được ẩn để dễ đọc)*")
            break

    # ============================================================
    # FEATURE EXPLORER (V2)
    # ============================================================
    if is_v2:
        st.markdown("---")
        st.subheader("🔧 Khám phá đặc trưng")

        tab1, tab2 = st.tabs(["Graph Features (14)", "Propagation Features (4)"])

        with tab1:
            if features:
                gf = features[0]
                gf_df = pd.DataFrame(list(gf.items()), columns=['Đặc trưng', 'Giá trị'])
                st.dataframe(gf_df, use_container_width=True, hide_index=True)

        with tab2:
            if features:
                pf = features[1]
                pf_df = pd.DataFrame(list(pf.items()), columns=['Đặc trưng', 'Giá trị'])
                st.dataframe(pf_df, use_container_width=True, hide_index=True)

    # ============================================================
    # FEATURE EXPLORER (V1)
    # ============================================================
    else:
        st.markdown("---")
        st.subheader("🔧 Khám phá đặc trưng")
        st.info("V1 sử dụng 5,000 TF-IDF + 384 MiniLM + 14 Graph Features = 5,398 tổng cộng. "
                "TF-IDF Features là vector thưa (sparse) với số chiều cao.")

        graph_cols = [
            'node_in_degree', 'node_out_degree', 'pagerank_score',
            'betweenness_centrality', 'closeness_centrality',
            'user_prior_rumor_ratio', 'user_post_count', 'user_thread_count',
            'user_avg_depth', 'subtree_reply_count', 'sibling_count',
            'position_in_thread', 'source_user_credibility', 'source_network_size'
        ]
        try:
            from utils.graph_features import get_graph_feature_columns
            graph_cols = get_graph_feature_columns()
        except ImportError:
            pass

        try:
            df_graph = pd.read_csv(os.path.join(PROJECT_ROOT, "data/processed/pheme_features_with_graph.csv"))
            g_row = df_graph[df_graph['post_id'] == post_id]
            if len(g_row) > 0:
                g_df = pd.DataFrame({
                    'Đặc trưng': graph_cols,
                    'Giá trị': [float(g_row[c].iloc[0]) for c in graph_cols]
                })
                st.dataframe(g_df, use_container_width=True, hide_index=True)
        except:
            st.info("Graph Features có sẵn sau khi chạy Pipeline huấn luyện.")

else:
    st.info("👈 Chọn một cuộc hội thoại từ thanh bên và nhấn **Phân tích cuộc hội thoại** để chạy trình diễn.")
    st.markdown("""
    ### Cách hoạt động

    1. **V2 (Khuyến nghị)**: 402 đặc trưng = 384 MiniLM Embeddings + 4 Propagation Features + 14 Graph Topology Features
    2. **V1**: 5,398 đặc trưng = 5,000 TF-IDF + 384 MiniLM + 14 Graph Features

    Mô hình phân loại các cuộc hội thoại là **Rumor** hoặc **Non-Rumor** kèm điểm tin cậy.
    """)