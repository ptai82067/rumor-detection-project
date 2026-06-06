#!/usr/bin/env python3
"""
PHEME Rumor Detection — Thesis Demo UI
========================================
Graduation thesis defense demonstration for:

    Rumor Detection on Social Media using Ontology and Knowledge Graph

Launch with:
    streamlit run ui/app.py
"""
import streamlit as st
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

st.set_page_config(
    page_title="Rumor Detection — Đồ án Tốt nghiệp",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.title("🎓 Đồ án Tốt nghiệp")
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Rumor Detection on Social Media**
*sử dụng Ontology và Knowledge Graph*
""")
st.sidebar.markdown("---")

# Check model status
v1_meta_path = os.path.join(PROJECT_ROOT, "models", "v1", "metadata.json")
v2_meta_path = os.path.join(PROJECT_ROOT, "models", "v2", "metadata.json")
v1_ready = os.path.exists(v1_meta_path)
v2_ready = os.path.exists(v2_meta_path)

if v1_ready or v2_ready:
    st.sidebar.success("✅ Mô hình đã sẵn sàng" if (v1_ready and v2_ready) else "⚠️ Mô hình chưa đầy đủ")
else:
    st.sidebar.warning("⚠️ Chạy script huấn luyện trước")

st.sidebar.markdown("---")

# Navigation
st.sidebar.markdown("### 📑 Danh mục")

st.sidebar.markdown("""
1. 🔬 Quá trình phát triển nghiên cứu
2. 🕵️ Rumor Detection
3. 🕸️ Khám phá Ontology & KG
4. 📊 Phân tích đặc trưng
5. 📈 Kết quả thực nghiệm
""")

st.sidebar.markdown("---")
st.sidebar.markdown("**HCMUT — 2026**")

# ============================================================
# MAIN PAGE — Welcome / Landing
# ============================================================

st.title("🎓 Rumor Detection on Social Media")
st.markdown("## sử dụng Ontology và Knowledge Graph")
st.markdown("---")

st.markdown("""
### Chào mừng đến với Đồ án Tốt nghiệp

Ứng dụng tương tác này trình diễn toàn bộ quy trình nghiên cứu về
**phát hiện tin đồn trên mạng xã hội** sử dụng Knowledge Graph
và Embedding ngữ nghĩa.

### Các nội dung có thể khám phá:

| Trang | Mô tả |
|-------|-------|
| **🔬 Quá trình phát triển nghiên cứu** | Lộ trình nghiên cứu từ Baseline đến Hybrid Model |
| **🕵️ Rumor Detection** | Trình diễn phát hiện tin đồn tương tác — chọn cuộc hội thoại và dự đoán |
| **🕸️ Khám phá Ontology & KG** | Khám phá PHEME Ontology V1/V2 và trực quan hóa Knowledge Graph |
| **📊 Phân tích đặc trưng** | Phân tích chi tiết tất cả Feature Dimensions và nhóm đặc trưng |
| **📈 Kết quả thực nghiệm** | Chỉ số cuối cùng, Ablation Study, Confusion Matrices, biểu đồ |
""")

st.markdown("---")

# Quick stats dashboard
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Mô hình", "2 (V1 + V2)", help="V1: Post-Level (5,398 chiều), V2: Thread-Level (402 chiều)")
with col2:
    st.metric("Accuracy tốt nhất", "98.11%", help="Full Hybrid model (V2)")
with col3:
    st.metric("Recall tốt nhất", "96.46%", help="Full Hybrid model (V2)")
with col4:
    st.metric("Giảm FN", "-1,169", help="So với TF-IDF Baseline")

st.markdown("---")

with st.expander("ℹ️ Hướng dẫn sử dụng"):
    st.markdown("""
    1. **Điều hướng** bằng thanh bên hoặc tab trang ở phía trên
    2. **Bắt đầu với Trang 1** để hiểu quá trình phát triển nghiên cứu
    3. **Thử Trang 2** cho trình diễn phát hiện tin đồn tương tác
    4. **Khám phá Trang 3-5** để phân tích chi tiết về Ontology, đặc trưng và kết quả
    
    **Lưu ý**: Trình diễn yêu cầu các file mô hình đã được huấn luyện.
    Chạy các lệnh sau nếu chưa được tạo:
    
    ```bash
    python scripts/train_and_save_v1.py
    python scripts/train_and_save_v2.py
    ```
    
    Các tập dữ liệu, hình ảnh trực quan và bảng kết quả đã được tải sẵn
    và không cần xử lý thêm.
    """)