"""
Page 4: Feature Analysis
=========================
Shows V1 and V2 feature composition with dimension comparison charts,
feature group descriptions, and detailed breakdown tables.
"""
import streamlit as st
import pandas as pd
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from ui.components.metrics_charts import create_feature_dimension_chart

st.title("📊 Phân tích đặc trưng")
st.markdown("---")

# Feature Dimension Overview
st.subheader("So sánh Feature Dimensions")

col1, col2 = st.columns(2)

with col1:
    st.info("### V1 — Post-Level (5,398 chiều)")
    st.markdown("""
    | Thành phần | Số chiều | Nguồn |
    |-----------|-----------|--------|
    | TF-IDF | **5,000** | `TfidfVectorizer(max_features=5000, ngram_range=(1,2))` |
    | MiniLM | **384** | `sentence-transformers/all-MiniLM-L6-v2` |
    | Graph | **14** | Post-level: Centrality, Reputation, Thread Structure, Authority |
    | **Tổng** | **5,398** | `[TF-IDF + MiniLM + Graph]` |
    """)

with col2:
    st.success("### V2 — Thread-Level (402 chiều)")
    st.markdown("""
    | Thành phần | Số chiều | Nguồn |
    |-----------|-----------|--------|
    | MiniLM | **384** | `sentence-transformers/all-MiniLM-L6-v2` (văn bản nguồn) |
    | Propagation | **4** | `thread_size, max_depth, avg_depth, reply_rate` |
    | Graph | **14** | Thread-level: Topology, PageRank, Centrality, User Ratio |
    | **Tổng** | **402** | `[MiniLM + Propagation + Graph]` |
    """)

# Feature dimension chart
st.plotly_chart(create_feature_dimension_chart(), use_container_width=True)

st.markdown("---")

# V2 Feature Groups
st.subheader("Nhóm đặc trưng V2 — Phân tích chi tiết")

v2_tab1, v2_tab2, v2_tab3 = st.tabs(["📐 Graph Topology", "📈 Propagation", "🧠 MiniLM"])

with v2_tab1:
    st.markdown("### 14 Thread-Level Graph Features")
    st.markdown("""
    Các đặc trưng này được trích xuất từ Knowledge Graph Reply Trees và thể hiện
    các tính chất cấu trúc của các cuộc hội thoại.
    """)

    graph_features_data = {
        'Đặc trưng': [
            'thread_depth', 'num_nodes', 'num_edges',
            'avg_branching_factor', 'max_branching_factor',
            'source_reply_count', 'leaf_ratio', 'avg_depth',
            'source_pagerank', 'avg_pagerank',
            'source_centrality', 'avg_centrality',
            'user_rumor_ratio', 'unique_users'
        ],
        'Nhóm': [
            'Topology', 'Topology', 'Topology',
            'Topology', 'Topology',
            'Topology', 'Topology', 'Topology',
            'PageRank', 'PageRank',
            'Centrality', 'Centrality',
            'User', 'User'
        ],
        'Mô tả': [
            'Độ sâu tối đa của Thread',
            'Tổng số bài đăng trong Thread',
            'Tổng số mối quan hệ trả lời',
            'Số trả lời trung bình mỗi node',
            'Số trả lời tối đa cho một bài đăng',
            'Số trả lời trực tiếp cho Source Post',
            'Tỷ lệ bài đăng không có trả lời (leaf)',
            'Độ sâu trung bình các bài đăng',
            'PageRank của Source Post',
            'PageRank trung bình trong Thread',
            'Degree Centrality của Source Post',
            'Degree Centrality trung bình trong Thread',
            'Tỷ lệ đăng tin đồn lịch sử của người dùng',
            'Số người dùng duy nhất trong Thread'
        ]
    }
    gf_df = pd.DataFrame(graph_features_data)
    st.dataframe(gf_df, use_container_width=True, hide_index=True)

with v2_tab2:
    st.markdown("### 4 Propagation Features")
    st.markdown("""
    Propagation Features thể hiện cách thông tin lan truyền qua cấu trúc Thread,
    bao gồm kích thước, độ sâu và tốc độ phản hồi.
    """)

    prop_features_data = {
        'Đặc trưng': ['thread_size', 'max_depth', 'avg_depth', 'reply_rate'],
        'Mô tả': [
            'Tổng số bài đăng trong Thread',
            'Độ sâu tối đa của cây trả lời',
            'Độ sâu trung bình của tất cả bài đăng',
            'Tổng bài đăng / (thread_size + 1) — mật độ trả lời chuẩn hóa'
        ]
    }
    pf_df = pd.DataFrame(prop_features_data)
    st.dataframe(pf_df, use_container_width=True, hide_index=True)

with v2_tab3:
    st.markdown("### MiniLM Sentence Embeddings")
    st.markdown("""
    **Model**: `all-MiniLM-L6-v2` từ SentenceTransformers

    | Thuộc tính | Giá trị |
    |----------|-------|
    | Số chiều | 384 |
    | Nguồn | V2: văn bản nguồn (bài đăng đầu tiên); V1: toàn bộ văn bản |
    | Dung lượng model | ~80 MB |
    | Loại | Pooled Sentence Embeddings (Mean Pooling) |
    | Dữ liệu huấn luyện | 1B+ cặp câu (học tương phản) |

    MiniLM là phiên bản distilled của BERT được tối ưu hóa cho tương đồng ngữ nghĩa cấp câu.
    Nó tạo ra Embedding có kích thước cố định thể hiện ý nghĩa ngữ nghĩa của văn bản.
    """)

st.markdown("---")

# V1 Feature Groups
st.subheader("Nhóm đặc trưng V1 — Phân tích chi tiết")

v1_tab1, v1_tab2, v1_tab3 = st.tabs(["📝 TF-IDF", "🧠 MiniLM", "🕸️ Graph"])

with v1_tab1:
    st.markdown("### TF-IDF Features (5,000 chiều)")
    st.markdown("""
    | Tham số | Giá trị |
    |-----------|-------|
    | max_features | 5,000 |
    | ngram_range | (1, 2) |
    | stop_words | 'english' |
    | min_df | 2 |
    | max_df | 0.95 |

    TF-IDF ghi nhận các mẫu từ vựng: unigrams và bigrams được gán trọng số
    dựa trên tần số nghịch đảo tài liệu trong toàn bộ kho ngữ liệu.
    """)

with v1_tab2:
    st.markdown("### MiniLM Sentence Embeddings (384 chiều)")
    st.markdown("""
    Cùng kiến trúc với V2, nhưng được áp dụng trên **toàn bộ văn bản bài đăng**
    thay vì chỉ văn bản nguồn.
    """)

with v1_tab3:
    st.markdown("### 14 Post-Level Graph Features")
    st.markdown("""
    Khác với tổng hợp cấp Thread của V2, Graph Features của V1 được tính cho từng bài đăng:

    | Nhóm đặc trưng | Các Feature |
    |--------------|----------|
    | **Node Centrality** | `node_in_degree`, `node_out_degree`, `pagerank_score`, `betweenness_centrality`, `closeness_centrality` |
    | **User Reputation** | `user_prior_rumor_ratio`, `user_post_count`, `user_thread_count`, `user_avg_depth` |
    | **Thread Structure** | `subtree_reply_count`, `sibling_count`, `position_in_thread` |
    | **Source Authority** | `source_user_credibility`, `source_network_size` |

    **Nguồn**: `utils/graph_features.py` — được tính qua `extract_all_graph_features()`
    từ Reply Graph (NetworkX DiGraph) xây dựng từ Knowledge Graph.
    """)

st.markdown("---")

# Key Insight
st.info("""
**Kết luận chính**: V2 đạt hiệu năng cao hơn (98.11%) với chỉ 402 đặc trưng
nhờ hoạt động ở cấp Thread, trong khi V1 (96.23%) sử dụng 5,398 đặc trưng ở cấp Post.
Việc tổng hợp cấp Thread cung cấp tín hiệu mạnh hơn bằng cách hợp nhất
ngữ cảnh cuộc hội thoại vào một vector đặc trưng duy nhất.
""")