"""
Page 3: Ontology & KG Explorer
================================
Tabs for Ontology V1, Ontology V2, KG V1, KG V2 visualizations.
Displays existing figures and pre-generated diagrams.
"""
import streamlit as st
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from ui.components.data_loader import load_ontology_text
from ui.components.kg_visualizer import display_static_kg_viz, display_ontology_diagram

st.title("🕸️ Khám phá Ontology & Knowledge Graph")
st.markdown("---")

# Dataset Statistics Dashboard
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Tổng Triple (KG v1)", "~76,066 nodes, 65,565 edges")
with col2:
    st.metric("Class (V1)", "5", help="Event, Post, User, ConversationThread, VeracityLabel")
with col3:
    st.metric("Class (V2)", "7", help="+ SourcePost, ReplyPost là Subclasses")
with col4:
    st.metric("Object Properties", "7+", help="postedBy, aboutEvent, repliesTo, inThread, hasVeracity, ...")

tabs = st.tabs(["Ontology V1", "Ontology V2", "KG V1 Visualization", "KG V2 Visualization", "KG Statistics"])

# ============================================================
# TAB 1: Ontology V1
# ============================================================
with tabs[0]:
    st.subheader("PHEME Ontology V1 — Ontology tối thiểu")
    st.markdown("""
    Ontology V1 định nghĩa 5 Class cốt lõi để mô hình hóa sự lan truyền tin đồn trên mạng xã hội:

    | Class | Mô tả |
    |-------|-------|
    | **Event** | Sự kiện thực tế tạo ra thảo luận trên mạng xã hội |
    | **Post** | Bài đăng trên mạng xã hội (tweet) trong PHEME dataset |
    | **User** | Người dùng tạo ra bài đăng |
    | **ConversationThread** | Cuộc hội thoại gồm bài đăng gốc và các trả lời |
    | **VeracityLabel** | Nhãn xác thực: tin đồn (1) hoặc không phải tin đồn (0) |

    **Object Properties**: `postedBy`, `aboutEvent`, `repliesTo`, `inThread`, `hasVeracity`
    """)

    col1, col2 = st.columns(2)
    with col1:
        display_ontology_diagram(version=1)

    with col2:
        text = load_ontology_text(version=1)
        if text:
            with st.expander("Xem Ontology V1 Source (TTL)"):
                st.code(text[:2000], language="turtle")

# ============================================================
# TAB 2: Ontology V2
# ============================================================
with tabs[1]:
    st.subheader("PHEME Ontology V2 — Ontology mở rộng")
    st.markdown("""
    V2 mở rộng V1 với các Subclass và thuộc tính bổ sung cho mô hình hóa ngữ nghĩa phong phú hơn:

    **Subclass mới**:

    | Class | Parent | Mô tả |
    |-------|--------|-------|
    | **SourcePost** | Post | Bài đăng gốc bắt đầu một cuộc hội thoại |
    | **ReplyPost** | Post | Bài đăng trả lời một bài đăng khác trong cuộc hội thoại |

    **Data Properties mới**: `text`, `createdAt`, `depth`, `childrenCount`,
    `timeSinceSource`, `threadSize`, `maxDepth`, `replySpeed`
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.info("**Cải tiến chính của V2**")
        st.markdown("""
        - Thêm phân biệt SourcePost/ReplyPost để phân tích cấu trúc Thread
        - Duyệt cây trả lời: `post → repliesTo → parent`
        - Propagation Metrics: depth, children count, timing
        - Tổng hợp Thread-level: size, depth, reply speed
        - Sửa lỗi URI Fragment giúp trích xuất 65,565 Reply Edges
        """)

    with col2:
        text = load_ontology_text(version=2)
        if text:
            with st.expander("Xem Ontology V2 Source (TTL)"):
                st.code(text[:2000], language="turtle")

# ============================================================
# TAB 3: KG V1 Visualization
# ============================================================
with tabs[2]:
    st.subheader("Knowledge Graph V1 — Reply Graph cấp Post")
    st.markdown("""
    KG V1 được xây dựng từ PHEME dataset sử dụng Ontology V1.
    Nó mô hình hóa các bài đăng, người dùng, sự kiện và cuộc hội thoại cùng mối quan hệ trả lời.
    """)

    col1, col2 = st.columns([3, 1])
    with col1:
        displayed = display_static_kg_viz()
        if not displayed:
            st.info("Không tìm thấy hình ảnh KG. Chạy script trực quan hóa KG trước.")

    with col2:
        st.markdown("**Thống kê KG V1**")
        st.markdown("""
        - **Nodes**: 76,066 (posts + users + events + threads)
        - **Edges**: 65,565 mối quan hệ trả lời
        - **Cấu trúc**: Directed Acyclic Graph (cây mỗi Thread)
        - **Cycle Detection**: DFS-based với loại bỏ cạnh
        """)

        st.markdown("**Data Properties**")
        st.markdown("""
        - `text`: Nội dung bài đăng
        - `createdAt`: Thời gian
        - `depth`: Vị trí trong Thread
        - `childrenCount`: Số trả lời trực tiếp
        - `timeSinceSource`: Khoảng cách thời gian
        """)

# ============================================================
# TAB 4: KG V2 Visualization
# ============================================================
with tabs[3]:
    st.subheader("Knowledge Graph V2 — Thread-Level Aggregated Graph")
    st.markdown("""
    KG V2 mở rộng V1 với các đặc trưng Graph cấp Thread (14 Graph Topology Features)
    để tăng cường phát hiện tin đồn. Đây là **kiến trúc KG cuối cùng** được sử dụng
    trong Hybrid Model.
    """)

    col1, col2 = st.columns([3, 1])
    with col1:
        displayed = display_static_kg_viz()
        if not displayed:
            st.info("Sử dụng cùng hình ảnh trực quan với KG V1 (hiển thị ở trên)")

    with col2:
        st.markdown("**14 Thread-Level Graph Features**")
        st.markdown("""
        **Topology**:
        - thread_depth, num_nodes, num_edges
        - avg_branching_factor, max_branching_factor
        - source_reply_count, leaf_ratio
        - avg_depth

        **PageRank**:
        - source_pagerank, avg_pagerank

        **Centrality**:
        - source_centrality, avg_centrality

        **User**:
        - user_rumor_ratio, unique_users
        """)

# ============================================================
# TAB 5: KG Statistics
# ============================================================
with tabs[4]:
    st.subheader("Thống kê chi tiết Knowledge Graph")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Quy trình xây dựng")
        st.markdown("""
        **1. Định nghĩa Ontology** (file TTL)
        - Class và Property được định nghĩa trong OWL/RDF
        - Version 1: 5 lớp, 5 Object Properties
        - Version 2: 7 lớp, mở rộng Data Properties

        **2. Sinh RDF Triple** (`build_kg.py`)
        - Lớp `KnowledgeGraphBuilder` xử lý Feature CSV
        - Tạo URI cho posts, users, events, threads
        - Xác thực: không trùng lặp, cấu trúc đúng, tính chất phi chu trình

        **3. Cycle Detection & Removal**
        - Phát hiện Cycle dựa trên DFS trong Reply Graph
        - Phát hiện Self-loop
        - Xác thực tham chiếu chéo Thread

        **4. Graph Feature Extraction** (`graph_features.py`)
        - Reply Graph → NetworkX DiGraph
        - Tính toán PageRank, Centrality, Betweenness
        - Phân tích User Reputation và Thread Structure
        """)

    with col2:
        st.markdown("### Kết quả xác thực")
        st.markdown("""
        **Sau khi sửa lỗi URI (05A.1)**:

        | Kiểm tra | Kết quả |
        |-------|--------|
        | Reply Graph Nodes | 76,066 |
        | Reply Graph Edges | 65,565 |
        | Graph Columns (14) | Tất cả đều có |
        | Non-zero Topology | Đã xác nhận |
        | Cycles detected | 0 (hoàn toàn phi chu trình) |
        | Enriched Dataset | 102,440 x 31 |

        **Nguyên nhân gốc (05A)**:
        - URI parsing kiểm tra `'/post/'` nhưng URI thực tế dùng `'#post/'`
        - Đã sửa trong `utils/graph_features.py` dòng 34
        - Trước khi sửa: 0 edges. Sau khi sửa: 65,565 edges.
        """)

st.markdown("---")
st.caption("Knowledge Graph được xây dựng từ PHEME dataset sử dụng RDFLib và NetworkX")