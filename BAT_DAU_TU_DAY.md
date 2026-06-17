# 🎓 HƯỚNG DẪN BẮT ĐẦU — ĐỒ ÁN TỐT NGHIỆP

## Phát hiện tin đồn trên mạng xã hội sử dụng Ontology và Knowledge Graph

---

## 📋 MỤC LỤC

1. [Giới thiệu đề tài](#1-giới-thiệu-đề-tài)
2. [Cấu trúc thư mục](#2-cấu-trúc-thư-mục)
3. [Yêu cầu hệ thống](#3-yêu-cầu-hệ-thống)
4. [Cài đặt Python](#4-cài-đặt-python)
5. [Tạo môi trường ảo](#5-tạo-môi-trường-ảo)
6. [Cài đặt dependencies](#6-cài-đặt-dependencies)
7. [Kiểm tra cài đặt thành công](#7-kiểm-tra-cài-đặt-thành-công)
8. [Chạy giao diện demo (Streamlit)](#8-chạy-giao-diện-demo-streamlit)
9. [Huấn luyện lại mô hình](#9-huấn-luyện-lại-mô-hình)
10. [Xem kết quả](#10-xem-kết-quả)
11. [Troubleshooting — Các lỗi thường gặp](#11-troubleshooting--các-lỗi-thường-gặp)
12. [FAQ](#12-faq)

---

## 1. GIỚI THIỆU ĐỀ TÀI

**Tên đề tài**: *Phát hiện tin đồn trên mạng xã hội sử dụng Ontology và Knowledge Graph*

Đồ án xây dựng hệ thống phát hiện tin đồn (rumor detection) trên dữ liệu mạng xã hội Twitter (PHEME dataset). Hệ thống kết hợp nhiều kỹ thuật:

| Kỹ thuật | Mô tả |
|----------|-------|
| **TF-IDF** | Trích xuất đặc trưng văn bản truyền thống (bag-of-words + n-grams) |
| **MiniLM (Sentence-BERT)** | Embedding ngữ nghĩa câu (384 chiều) |
| **Propagation Features** | Đặc trưng lan truyền của hội thoại (thread_size, depth, reply_rate) |
| **Graph Features (Knowledge Graph)** | Đặc trưng cấu trúc đồ thị từ Knowledge Graph (PageRank, centrality, v.v.) |
| **Ontology** | Ontology PHEME V1 và V2 định nghĩa các lớp, thuộc tính và quan hệ |

**Kết quả mô hình tốt nhất (V2 Thread-Level)**:

| Metric | Giá trị |
|--------|---------|
| **Accuracy** | **98.11%** |
| Precision | 97.94% |
| Recall | 96.46% |
| F1-Score | 97.19% |

---

## 2. CẤU TRÚC THƯ MỤC

```
rumor-detection-project/
├── BAT_DAU_TU_DAY.md            ← (BẠN ĐANG Ở ĐÂY) Hướng dẫn toàn diện
├── README.md                    Tổng quan dự án (ngắn gọn)
├── README_UI.md                 Hướng dẫn UI chi tiết (bằng tiếng Anh)
├── requirements.txt             Danh sách Python packages cần cài
│
├── config/                      Cấu hình (đang trống)
├── data/
│   ├── raw/                     Dữ liệu gốc PHEME (cần tải từ nguồn)
│   └── processed/               Dữ liệu đã xử lý (có sẵn trong repo)
│       ├── pheme_features.csv           102,440 posts
│       ├── pheme_features_with_graph.csv Posts + graph features
│       ├── graph_features_v2.csv        Thread-level graph features
│       ├── pheme_clean.csv              Dữ liệu sạch sau preprocessing
│       ├── pheme_kg.ttl                 Knowledge Graph V1 (RDF/Turtle)
│       └── pheme_kg_v2.ttl              Knowledge Graph V2
│
├── preprocessing/               Pipeline tiền xử lý dữ liệu
│   ├── loader.py                Nạp dữ liệu PHEME gốc
│   ├── parser.py                Phân tích cú pháp thread
│   ├── normalizer.py            Chuẩn hóa văn bản
│   ├── propagation_features.py  Tính đặc trưng lan truyền
│   └── run_pipeline.py          Chạy toàn bộ pipeline
│
├── ontology/
│   ├── pheme_ontology_v1.ttl    Ontology V1 (157 dòng)
│   └── pheme_ontology_v2.ttl    Ontology V2 (mở rộng)
│
├── knowledge_graph/
│   ├── build_kg.py              Xây dựng Knowledge Graph V1
│   └── build_kg_v2.py           Xây dựng Knowledge Graph V2
│
├── models/
│   ├── v1/                      (CẦN CHẠY SCRIPT ĐỂ TẠO — xem mục 9)
│   │   ├── model.joblib              LogisticRegression (5,398 dim)
│   │   ├── tfidf_vectorizer.joblib   TfidfVectorizer (5,000 vocab)
│   │   ├── scaler_graph.joblib       StandardScaler (14 dim)
│   │   ├── scaler_propagation.joblib StandardScaler (4 dim)
│   │   └── metadata.json             Metrics và cấu hình
│   └── v2/                      (ĐÃ CÓ SẴN — có thể dùng ngay)
│       ├── model.joblib              LogisticRegression (402 dim)
│       ├── scaler.joblib             StandardScaler (402 dim)
│       └── metadata.json             Metrics và cấu hình
│
├── scripts/
│   ├── train_and_save_v1.py     Huấn luyện và lưu V1
│   ├── train_and_save_v2.py     Huấn luyện và lưu V2 + Ablation
│   └── validate_saved_models.py Kiểm tra model đã huấn luyện
│
├── src/
│   └── extract_graph_features_v2.py  Trích xuất graph features phiên bản 2
│
├── utils/
│   ├── bert_feature_fusion.py   Trích xuất MiniLM embeddings + fusion
│   ├── graph_features.py        Trích xuất graph features từ KG (NetworkX)
│   └── visualization/           Công cụ trực quan hóa
│
├── ui/                          Giao diện Streamlit
│   ├── app.py                   Entry point (chạy: streamlit run ui/app.py)
│   ├── components/
│   │   ├── data_loader.py       Nạp dữ liệu có caching
│   │   ├── model_manager.py     Tải model + thực hiện dự đoán
│   │   ├── kg_visualizer.py     Hiển thị Knowledge Graph
│   │   └── metrics_charts.py    Biểu đồ Plotly
│   └── pages/
│       ├── 1_Research_Evolution.py      Quá trình phát triển
│       ├── 2_Rumor_Detection.py         Demo phát hiện tin đồn
│       ├── 3_Ontology_KG_Explorer.py    Khám phá Ontology & KG
│       ├── 4_Feature_Analysis.py        Phân tích đặc trưng
│       └── 5_Experimental_Results.py    Kết quả thực nghiệm
│
├── notebooks/                   Jupyter notebooks (tham khảo)
│   ├── 03_rumor_detection_baseline...   Baseline (03)
│   ├── 04_graph_feature_integration...  Graph features (04)
│   ├── 05_bert_graph_fusion.ipynb       Hybrid model (05)
│   └── ablation_study.ipynb            Ablation study
│
├── results/
│   ├── pheme_dataset_statistics.csv     Thống kê dataset
│   └── ablation/                        Kết quả ablation study
│       ├── ablation_table.csv           Bảng kết quả (CSV)
│       ├── ablation_table.md            Bảng kết quả (Markdown)
│       └── figures/                     Biểu đồ (PNG)
│           ├── fig1_ablation_bar_metrics.png
│           ├── fig2_recall_fn_trend.png
│           ├── fig3_confusion_matrices.png
│           └── fig4_feature_contribution.png
│
├── docs/figures/                       Hình ảnh ontology và KG
├── visualization/                       Hình ảnh KG semantic
├── reports/                             Báo cáo đánh giá
├── logs/                                Log files
├── tests/                               Unit tests (đang trống)
├── run_ablation.py                      Script ablation study độc lập
├── first_baseline.py                    Script baseline đầu tiên
├── main.py                              Entry point cũ (không dùng)
└── final_metrics_table.csv              Bảng metrics cuối cùng
```

---

## 3. YÊU CẦU HỆ THỐNG

| Hệ điều hành | Khả năng tương thích | Ghi chú |
|-------------|----------------------|---------|
| **Windows 10/11** | ✅ Ưu tiên | Đã kiểm tra |
| **Linux (Ubuntu 20.04+)** | ✅ Tốt | pip + venv chuẩn |
| **macOS** | ✅ Tốt | Tương tự Linux |

**Cấu hình tối thiểu khuyến nghị**:
- RAM: ≥ 8 GB (khuyến nghị 16 GB cho inference)
- Ổ cứng: ≥ 10 GB trống
- Internet: cần để tải model `all-MiniLM-L6-v2` và PyTorch
- CPU: hỗ trợ AVX2 (hầu hết CPU từ 2015 trở lên)

---

## 4. CÀI ĐẶT PYTHON

**Phiên bản yêu cầu: Python 3.10**

> ⚠️ **Quan trọng**: Không dùng Python 3.12 hoặc 3.13 vì một số package (như `sentence-transformers`, `rdflib`, `owlready2`) có thể chưa tương thích. Python 3.11 cũng hoạt động, nhưng 3.10 được khuyến nghị để đảm bảo ổn định nhất.

### Windows

**Cách 1 — Download từ python.org** (khuyến nghị):
1. Truy cập https://www.python.org/downloads/release/python-31011/
2. Tải `Windows installer (64-bit)` — file: `python-3.10.11-amd64.exe`
3. Khi cài đặt: ✅ **NHỚ tick "Add Python to PATH"**
4. Click Install Now

**Cách 2 — Dùng winget** (nếu có Windows Package Manager):
```
winget install Python.Python.3.10
```

Sau khi cài, kiểm tra:
```cmd
python --version
```
Kết quả: `Python 3.10.x`

### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev
```

### macOS
```bash
brew install python@3.10
```

---

## 5. TẠO MÔI TRƯỜNG ẢO

> ⚠️ **Bắt buộc**: Luôn tạo môi trường ảo riêng cho đồ án để tránh xung đột package.

```cmd
:: Windows (CMD hoặc PowerShell)
cd d:\rumor-detection-project
python -m venv venv
venv\Scripts\activate
```

Trên **Linux/macOS**:
```bash
cd /path/to/rumor-detection-project
python3.10 -m venv venv
source venv/bin/activate
```

Khi môi trường ảo được kích hoạt, bạn sẽ thấy dấu hiệu:
- Windows: `(venv) D:\rumor-detection-project>`
- Linux/macOS: `(venv) user@machine:~/rumor-detection-project$`

---

## 6. CÀI ĐẶT DEPENDENCIES

### Bước 1: Nâng cấp pip
```cmd
python -m pip install --upgrade pip
```

### Bước 2: Cài PyTorch (trước tiên — quan trọng)

PyTorch là package lớn nhất (~2 GB) và cần được cài trước.

```cmd
:: Windows/Linux (CPU) — khuyến nghị nếu không có GPU
pip install torch>=1.9.0

:: Windows (NVIDIA GPU) — dùng CUDA
pip install torch>=1.9.0 --index-url https://download.pytorch.org/whl/cu118

:: Linux (NVIDIA GPU)
pip install torch>=1.9.0 --index-url https://download.pytorch.org/whl/cu118
```

> ⏱ Thời gian cài PyTorch: 3–10 phút tùy đường truyền.

### Bước 3: Cài tất cả dependencies
```cmd
pip install -r requirements.txt
```

> ⏱ Thời gian: 5–15 phút (tùy thuộc tốc độ mạng và cấu hình máy).

### Bước 4: Cài model spaCy (bổ sung)
```cmd
python -m spacy download en_core_web_sm
```

### Bước 5: Kiểm tra các package quan trọng

```cmd
python -c "import numpy; import pandas; import sklearn; import torch; import transformers; import sentence_transformers; import networkx; import rdflib; import streamlit; print('✅ All critical packages loaded successfully')"
```

---

## 7. KIỂM TRA CÀI ĐẶT THÀNH CÔNG

### Kiểm tra môi trường
```cmd
python -c "
import sys;
import torch;
import sklearn;
import sentence_transformers;
import streamlit;
import networkx;
import rdflib;
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'scikit-learn: {sklearn.__version__}')
print(f'Streamlit: {streamlit.__version__}')
print(f'NetworkX: {networkx.__version__}')
print(f'RDFLib: {rdflib.__version__}')
print('✅ Environment ready!')
"
```

### Kiểm tra dữ liệu
```cmd
python -c "
import os, pandas as pd;
root = '.';
files = [
    'data/processed/pheme_features.csv',
    'data/processed/pheme_features_with_graph.csv',
    'data/processed/graph_features_v2.csv',
    'data/processed/pheme_kg.ttl',
    'ontology/pheme_ontology_v1.ttl',
    'ontology/pheme_ontology_v2.ttl',
    'results/ablation/ablation_table.csv',
    'final_metrics_table.csv'
];
for f in files:
    path = os.path.join(root, f);
    exists = os.path.exists(path);
    size = os.path.getsize(path)//1024 if exists else 0;
    print(f'{\"✅\" if exists else \"❌\"} {f} ({size} KB)');
"
```

### Kiểm tra model đã huấn luyện
```cmd
python -c "
import os;
v1_ok = os.path.exists('models/v1/model.joblib');
v2_ok = os.path.exists('models/v2/model.joblib');
print(f'V1 model: {\"✅ Sẵn sàng\" if v1_ok else \"❌ CHƯA CÓ (chạy scripts/train_and_save_v1.py)\"}');
print(f'V2 model: {\"✅ Sẵn sàng\" if v2_ok else \"❌ CHƯA CÓ (chạy scripts/train_and_save_v2.py)\"}');
"
```

---

## 8. CHẠY GIAO DIỆN DEMO (STREAMLIT)

### 8.1. Yêu cầu trước khi chạy

- ✅ Môi trường ảo `venv` đã kích hoạt
- ✅ Đã cài `pip install -r requirements.txt`
- ✅ Model V2 phải có sẵn (xem mục 9 nếu chưa có)

### 8.2. Lệnh chạy

```cmd
streamlit run ui/app.py
```

### 8.3. URL truy cập

Sau khi chạy, Terminal sẽ hiển thị:

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Mở trình duyệt và truy cập: **http://localhost:8501**

> 💡 Nếu cổng 8501 bị chiếm, Streamlit sẽ tự động dùng cổng 8502,...

### 8.4. Các trang trong UI

| Trang | Mô tả | Cách sử dụng |
|-------|-------|--------------|
| **🏠 Trang chủ** | Tổng quan dự án, thông số nhanh | Mở app là thấy ngay |
| **🔬 Quá trình phát triển** (Page 1) | Timeline 3 pha: Baseline → Graph → Hybrid | Click vào sidebar |
| **🕵️ Rumor Detection** (Page 2) | Demo phát hiện tin đồn tương tác | Chọn thread → Click "Phân tích" |
| **🕸️ Ontology & KG** (Page 3) | Khám phá Ontology V1/V2, KG visualization | Chọn tab để xem |
| **📊 Feature Analysis** (Page 4) | Phân tích chi tiết các nhóm đặc trưng | Xem bảng và biểu đồ |
| **📈 Kết quả thực nghiệm** (Page 5) | Metrics cuối, ablation, confusion matrices | Xem tất cả kết quả |

### 8.5. Cách thực hiện dự đoán (Page 2)

1. Ở thanh bên trái, chọn **"V2 (Thread-Level)"** (khuyến nghị)
2. Chọn một cuộc hội thoại từ dropdown
3. Nhấn nút **"🔍 Phân tích cuộc hội thoại"**
4. Kết quả hiển thị:
   - **Dự đoán**: 🔴 Rumor hoặc 🟢 Non-Rumor
   - **Độ tin cậy**: progress bar %
   - **Số chiều đặc trưng**: 402 (V2) hoặc 5,398 (V1)
   - **Thời gian suy luận**: ~ms
   - **Nội dung hội thoại**: hiển thị các bài đăng
   - **Feature explorer**: Graph Features (14) và Propagation Features (4)

---

## 9. HUẤN LUYỆN LẠI MÔ HÌNH

> ⚠️ Các lệnh này sẽ chạy trong 15–60 phút tùy cấu hình máy.

### 9.1. Huấn luyện V1 (Post-Level — 5,398 chiều)

```cmd
python scripts/train_and_save_v1.py
```

Script này sẽ:
- Tải dữ liệu từ `data/processed/pheme_features_with_graph.csv`
- Trích xuất TF-IDF (5,000), MiniLM (384), Graph (14)
- Huấn luyện 4 hệ thống (A: TF-IDF, B: MiniLM, C: MiniLM+Graph, D: Hybrid)
- Lưu artifact vào `models/v1/` (model.joblib, tfidf_vectorizer.joblib, scalers, metadata.json)

**Kết quả mong đợi (V1)**:
```
Accuracy:  ~96.23%
```

### 9.2. Huấn luyện V2 (Thread-Level — 402 chiều, Best Model)

```cmd
python scripts/train_and_save_v2.py
```

Script này sẽ:
- Tải dữ liệu từ `data/processed/pheme_features.csv` + `graph_features_v2.csv`
- Xây dựng thread-level dataframe
- Trích xuất MiniLM (384), Propagation (4), Graph (14)
- Huấn luyện Full Hybrid + 4 config khác (ablation study)
- Lưu artifact vào `models/v2/` (model.joblib, scaler.joblib, metadata.json)
- Lưu ablation table vào `results/ablation/ablation_table.csv`

**Kết quả mong đợi (V2)**:
```
Accuracy:  98.11%
Recall:    96.46%
```

### 9.3. Kiểm tra model sau khi huấn luyện

```cmd
python scripts/validate_saved_models.py
```

Script này:
- Kiểm tra tất cả artifact files
- Xác thực số chiều feature
- Chạy sample prediction
- Tạo báo cáo: `reports/model_validation_report.md`

### 9.4. Chạy ablation study độc lập (tạo figures)

```cmd
python run_ablation.py
```

Script này tạo:
- `results/ablation/figures/fig1_ablation_bar_metrics.png`
- `results/ablation/figures/fig2_recall_fn_trend.png`
- `results/ablation/figures/fig3_confusion_matrices.png`
- `results/ablation/figures/fig4_feature_contribution.png`

---

## 10. XEM KẾT QUẢ

### 10.1. Figures — Ablation Study
Mở thư mục: `results/ablation/figures/`

| File | Mô tả |
|------|-------|
| `fig1_ablation_bar_metrics.png` | So sánh Acc/Prec/Recall/F1 giữa 5 config |
| `fig2_recall_fn_trend.png` | Xu hướng Recall + False Negatives |
| `fig3_confusion_matrices.png` | Ma trận nhầm lẫn cho 5 config |
| `fig4_feature_contribution.png` | Đóng góp của từng nhóm feature |

### 10.2. Tables — Ablation Study
- `results/ablation/ablation_table.csv` — Bảng CSV đầy đủ
- `results/ablation/ablation_table.md` — Bảng Markdown

### 10.3. Ontology
- `ontology/pheme_ontology_v1.ttl` — Ontology V1 (157 dòng, Turtle format)
- `ontology/pheme_ontology_v2.ttl` — Ontology V2 (mở rộng)

### 10.4. Knowledge Graph
- `data/processed/pheme_kg.ttl` — KG V1 (từ `build_kg.py`)
- `data/processed/pheme_kg_v2.ttl` — KG V2
- `visualization/kg_semantic_final.png` — Ảnh KG visualization

### 10.5. Reports
- `reports/FINAL_TEST_REPORT.md` — Báo cáo kiểm thử cuối cùng
- `reports/RUNTIME_VALIDATION.md` — Kiểm tra runtime
- `reports/ui_deployment_readiness.md` — Mức độ sẵn sàng triển khai

### 10.6. Figures — Ontology & KG
Mở thư mục: `docs/figures/`

| File | Mô tả |
|------|-------|
| `ontology_class_hierarchy.png` | Phân cấp lớp Ontology |
| `ontology_full_diagram.png` | Sơ đồ Ontology đầy đủ |
| `kg_sample_subgraph.png` | Mẫu Knowledge Graph |
| `kg_statistics_chart.png` | Thống kê KG |

---

## 11. TROUBLESHOOTING — CÁC LỖI THƯỜNG GẶP

### ⚠️ 11.1. Python không được nhận diện

**Lỗi**: `'python' is not recognized as an internal or external command`

**Nguyên nhân**: Python chưa được thêm vào PATH.

**Khắc phục**:
1. Gỡ cài đặt Python hiện tại
2. Cài lại Python 3.10, **nhớ tick "Add Python to PATH"**
3. Hoặc dùng đường dẫn đầy đủ:
   ```cmd
   C:\Users\LOC\AppData\Local\Programs\Python\Python310\python.exe -m venv venv
   ```

### ⚠️ 11.2. venv không kích hoạt được

**Lỗi**: `venv\Scripts\activate : File cannot be loaded because running scripts is disabled`

**Nguyên nhân**: Execution Policy của PowerShell bị hạn chế.

**Khắc phục**: Chạy trong **CMD** thay vì PowerShell, hoặc:
```powershell
powershell -ExecutionPolicy Bypass -File venv\Scripts\Activate.ps1
```

### ⚠️ 11.3. pip install lỗi — Microsoft Visual C++ Redistributable

**Lỗi**: `error: Microsoft Visual C++ 14.0 or greater is required`

**Nguyên nhân**: Thiếu C++ build tools (thường xảy ra với `pygraphviz` hoặc `spacy`).

**Khắc phục**:
1. Tải và cài: https://aka.ms/vs/17/release/vc_redist.x64.exe
2. Hoặc từ Visual Studio Installer → cài "Desktop development with C++"

### ⚠️ 11.4. torch lỗi — không tương thích CPU

**Lỗi**: `Your CPU does not support AVX2 instruction set`

**Nguyên nhân**: CPU quá cũ (trước 2015).

**Khắc phục**: Cài phiên bản PyTorch không yêu cầu AVX:
```cmd
pip install torch>=1.9.0 --index-url https://download.pytorch.org/whl/cpu
```

### ⚠️ 11.5. sentence-transformers lỗi — không tải được model

**Lỗi**: `OSError: Can't load model 'all-MiniLM-L6-v2'`

**Nguyên nhân**: Không có internet hoặc huggingface hub bị chặn.

**Khắc phục**:
1. Đảm bảo có internet
2. Nếu bị chặn (ở Việt Nam có thể gặp):
   ```cmd
   pip install -U huggingface_hub
   export HF_ENDPOINT=https://hf-mirror.com    # Linux/macOS
   set HF_ENDPOINT=https://hf-mirror.com        # Windows
   ```
3. Hoặc tải model thủ công từ HuggingFace, đặt vào `models/all-MiniLM-L6-v2/`

### ⚠️ 11.6. Không tìm thấy dataset

**Lỗi**: `FileNotFoundError: data/processed/pheme_features.csv`

**Nguyên nhân**: Thiếu file dữ liệu.

**Khắc phục**:
- Các file processed đã có sẵn trong repository (đã kiểm tra)
- Nếu thiếu `pheme_features_with_graph.csv`, chạy:
  ```cmd
  python -c "from utils.graph_features import extract_all_graph_features; import pandas as pd; df = pd.read_csv('data/processed/pheme_features.csv'); df_enriched = extract_all_graph_features(df); df_enriched.to_csv('data/processed/pheme_features_with_graph.csv', index=False)"
  ```

### ⚠️ 11.7. Streamlit không chạy

**Lỗi**: `streamlit : command not found` hoặc `streamlit run ui/app.py` báo lỗi.

**Nguyên nhân**: Streamlit chưa được cài hoặc chưa kích hoạt venv.

**Khắc phục**:
```cmd
venv\Scripts\activate     # Kích hoạt môi trường ảo trước
streamlit run ui/app.py
```

### ⚠️ 11.8. File model.joblib không tồn tại

**Lỗi**: Khi mở UI thấy: "⚠️ Chạy script huấn luyện trước"

**Nguyên nhân**: Model chưa được huấn luyện.

**Khắc phục**:
```cmd
python scripts/train_and_save_v1.py
python scripts/train_and_save_v2.py
```

### ⚠️ 11.9. Lỗi memory — OOM khi extract MiniLM embeddings

**Lỗi**: `MemoryError` hoặc Python bị crash

**Nguyên nhân**: RAM không đủ cho batch_size mặc định (32 hoặc 64).

**Khắc phục**: Giảm batch_size trong script train:
Sửa dòng `batch_size=32` → `batch_size=8` hoặc `batch_size=4` trong file `scripts/train_and_save_v1.py`.

### ⚠️ 11.10. owlready2 lỗi trên Windows

**Lỗi**: `ImportError: DLL load failed while importing owlready2`

**Nguyên nhân**: Thiếu C++ Redistributable.

**Khắc phục**: Cài đặt Visual C++ Redistributable (xem mục 11.3). Nếu vẫn lỗi, `owlready2` không bắt buộc — ontology vẫn có dạng TTL và có thể đọc bằng `rdflib`.

### ⚠️ 11.11. Port 8501 đã được sử dụng

**Lỗi**: `Port 8501 is already in use`

**Nguyên nhân**: Có một instance Streamlit khác đang chạy.

**Khắc phục**:
```cmd
streamlit run ui/app.py --server.port 8502
```
Hoặc tắt instance cũ trong Task Manager.

### ⚠️ 11.12. Module 'utils' not found

**Lỗi**: `ModuleNotFoundError: No module named 'utils'`

**Nguyên nhân**: Chạy script từ sai thư mục.

**Khắc phục**: Luôn chạy lệnh từ thư mục gốc `d:\rumor-detection-project`, nơi chứa thư mục `utils/`.

---

## 12. FAQ

### ❓ Hỏi: Tôi cần tải PHEME dataset gốc không?

**Đáp**: **Không.** Tất cả dữ liệu đã được xử lý sẵn trong thư mục `data/processed/`. Bạn chỉ cần tải dataset gốc nếu muốn chạy lại preprocessing pipeline từ đầu.

### ❓ Hỏi: Tôi không có GPU, có chạy được không?

**Đáp**: **Có.** Toàn bộ đồ án chạy trên CPU. Không yêu cầu GPU. Thời gian huấn luyện lâu hơn nhưng vẫn khả thi (~30–60 phút).

### ❓ Hỏi: Làm sao để biết model đã hoạt động?

**Đáp**: Mở UI, vào trang **🕵️ Rumor Detection**, chọn một thread và nhấn "Phân tích". Nếu thấy kết quả dự đoán + confidence, model hoạt động tốt.

### ❓ Hỏi: UI có hỗ trợ tiếng Việt không?

**Đáp**: **Có.** Toàn bộ giao diện Streamlit đã được viết bằng tiếng Việt.

### ❓ Hỏi: Làm sao xem metric chi tiết của từng model?

**Đáp**: Mở file `models/v1/metadata.json` và `models/v2/metadata.json` bằng bất kỳ text editor nào. Hoặc mở **📈 Kết quả thực nghiệm** (Page 5) trong UI.

### ❓ Hỏi: Tôi muốn thêm dữ liệu mới để kiểm tra?

**Đáp**: Hiện tại UI chỉ hỗ trợ dự đoán trên các thread có sẵn trong dataset PHEME. Để thêm dữ liệu mới, cần mở rộng phần data loading.

### ❓ Hỏi: V1 và V2 khác nhau thế nào?

**Đáp**: 
| Tiêu chí | V1 (Post-Level) | V2 (Thread-Level) |
|----------|-----------------|-------------------|
| Đơn vị dự đoán | Từng bài đăng | Cả luồng hội thoại |
| Số chiều | 5,398 | 402 |
| TF-IDF | ✅ 5,000 features | ❌ Không dùng |
| MiniLM | ✅ 384 | ✅ 384 |
| Propagation | ❌ Không dùng | ✅ 4 features |
| Graph | ✅ 14 features | ✅ 14 features |
| Accuracy | ~96.23% | **98.11%** |

### ❓ Hỏi: Làm thế nào để generate lại các figures?

**Đáp**: Chạy các lệnh sau:
```cmd
:: Ablation figures
python run_ablation.py

:: Ontology diagrams
python generate_ontology_diagram.py

:: Mermaid diagrams
python generate_mermaid_diagram.py

:: KG visualization
python notebooks/viz_kg_semantic_final.py
```

---

## 🎯 TÓM TẮT — CÁC LỆNH NHANH

| Mục đích | Lệnh |
|----------|------|
| Tạo môi trường | `python -m venv venv` |
| Kích hoạt (Windows) | `venv\Scripts\activate` |
| Kích hoạt (Linux/Mac) | `source venv/bin/activate` |
| Cài dependencies | `pip install -r requirements.txt` |
| Huấn luyện V1 | `python scripts/train_and_save_v1.py` |
| Huấn luyện V2 | `python scripts/train_and_save_v2.py` |
| Kiểm tra model | `python scripts/validate_saved_models.py` |
| Chạy UI | `streamlit run ui/app.py` |
| Ablation + figures | `python run_ablation.py` |
| Build KG V1 | `python knowledge_graph/build_kg.py` |

---

## 📝 LUỒNG CHẠY ĐẦY ĐỦ (CHO NGƯỜI MỚI BẮT ĐẦU)

```
Bước 1: Cài Python 3.10
    ↓
Bước 2: Mở terminal tại d:\rumor-detection-project
    ↓
Bước 3: python -m venv venv
    ↓
Bước 4: venv\Scripts\activate
    ↓
Bước 5: pip install -r requirements.txt
    ↓
Bước 6: python scripts/train_and_save_v2.py   (15-45 phút)
    ↓
Bước 7: python scripts/train_and_save_v1.py   (30-60 phút)
    ↓
Bước 8: streamlit run ui/app.py
    ↓
Bước 9: Mở http://localhost:8501 trong trình duyệt
    ↓
✅ HOÀN THÀNH — Có thể demo và khám phá!
```

---

*Tài liệu được tạo ngày: 09/06/2026*  
*Đồ án tốt nghiệp — NTU 2026*