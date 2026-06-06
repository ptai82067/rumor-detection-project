# Vietnamese Localization Report

**Date**: 2026-06-07  
**Objective**: Localize all 6 Streamlit UI files from English to Vietnamese  
**Files Modified**: 6

---

## Files Modified

| # | File | Action |
|---|------|--------|
| 1 | `ui/app.py` | Full localization |
| 2 | `ui/pages/1_Research_Evolution.py` | Full localization |
| 3 | `ui/pages/2_Rumor_Detection.py` | Full localization |
| 4 | `ui/pages/3_Ontology_KG_Explorer.py` | Full localization |
| 5 | `ui/pages/4_Feature_Analysis.py` | Full localization |
| 6 | `ui/pages/5_Experimental_Results.py` | Full localization |

## NOT Modified

| File | Reason |
|------|--------|
| `ui/components/data_loader.py` | No UI text — only data loading functions |
| `ui/components/model_manager.py` | No UI text — only model logic |
| `ui/components/metrics_charts.py` | No UI text — only Plotly chart building |
| `ui/components/kg_visualizer.py` | No UI text — only image display logic |
| `scripts/*` | Not UI files |

## Translation Statistics

### Total strings translated: ~120

| Category | Count | Examples |
|----------|-------|---------|
| Navigation labels | 10 | Trang chủ, Danh mục, Cài đặt dự đoán |
| Headers / Titles | 15 | Quá trình phát triển nghiên cứu, Phân tích đặc trưng |
| Button labels | 3 | Phân tích cuộc hội thoại |
| Metric labels | 10 | Mô hình, Độ tin cậy, Kết quả dự đoán |
| Help text / Tooltips | 8 | V1: Post-Level (5,398 chiều) |
| Markdown paragraphs | 25+ | Project descriptions, pipeline explanations |
| Table headers | 5 | Đặc trưng, Giá trị, Nhóm, Mô tả |
| Info / Warning / Error messages | 15 | Không tìm thấy mẫu cuộc hội thoại |
| Captions / Footer | 2 | Rumor Detection Thesis — Trường Đại học Bách khoa TP.HCM |

## Terminology Preserved in English

These technical terms were intentionally left in English per localization rules:

### Core Technical Terms
- **Rumor Detection** — Title term, kept in English
- **Knowledge Graph / KG** — Kept as-is
- **Ontology** — Kept as-is
- **MiniLM** — Model name
- **TF-IDF** — Algorithm name
- **Logistic Regression** — Algorithm name
- **Embedding** — Technical term
- **Graph Features** — Feature type
- **Propagation Features** — Feature type
- **Feature Engineering** — Process name
- **Thread** — Data structure (kept in English alongside Vietnamese "cuộc hội thoại")
- **Post** — Data structure (kept in English alongside Vietnamese "bài đăng")
- **Source Post** — Data structure
- **PageRank** — Algorithm name
- **Centrality** — Graph metric

### Evaluation Metrics
- **Accuracy** — Kept as-is (also referred to as "Độ chính xác" in context)
- **Precision** — Kept as-is
- **Recall** — Kept as-is
- **F1-Score / F1** — Kept as-is
- **ROC-AUC** — Kept as-is
- **MCC** — Kept as-is
- **False Negatives / FN** — Kept as-is

### Experimental Terms
- **Ablation Study** — Kept as-is
- **Confusion Matrix** — Kept as-is
- **Baseline** — Kept as-is
- **Hybrid** — Kept as-is

### Technical Identifiers
- **Streamlit** — Framework name
- **Plotly** — Library name
- **PHEME** — Dataset name
- **HCMUT** — University abbreviation

## Specific Transliteration Choices

| English | Vietnamese | Context |
|---------|-----------|---------|
| Home | Trang chủ | Navigation |
| Research Evolution | Quá trình phát triển nghiên cứu | Page title |
| Rumor Detection | Rumor Detection | Page title (kept English) |
| Ontology & KG Explorer | Khám phá Ontology & KG | Page title |
| Feature Analysis | Phân tích đặc trưng | Page title |
| Experimental Results | Kết quả thực nghiệm | Page title |
| Prediction | Kết quả dự đoán | Result label |
| Confidence | Độ tin cậy | Result label |
| Model Version | Phiên bản mô hình | Settings |
| Thread Selection | Chọn cuộc hội thoại | Dropdown label |
| Feature Explorer | Khám phá đặc trưng | Section title |
| Dataset Statistics | Thống kê dữ liệu | Section title |
| Analyze Thread | Phân tích cuộc hội thoại | Button label |
| Rumor | Tin đồn | Result label |
| Non-Rumor | Không phải tin đồn | Result label |
| How to Use | Hướng dẫn sử dụng | Expander title |
| Model Information | Thông tin mô hình | Section title |

## Pages Updated

| Page | Sections Localized |
|------|-------------------|
| **app.py** (Home) | Title, subtitle, welcome text, page descriptions table, metric cards (Models, Best Accuracy, Best Recall, FN Reduction), How to Use expander |
| **1_Research_Evolution.py** | Page title, project overview, 3-phase timeline cards, pipeline architecture diagram, performance evolution (baseline recall, best recall, accuracy, FN reduction), footer |
| **2_Rumor_Detection.py** | Sidebar (model selector, thread selector, analyze button), thread info header (Thread ID, ground truth, posts, model), results section (prediction, confidence, details), conversation thread display, feature explorer tabs, how it works |
| **3_Ontology_KG_Explorer.py** | Page title, statistics dashboard, all 5 tabs (Ontology V1, Ontology V2, KG V1, KG V2, KG Statistics), tables, expanders, Vietnamese dataset descriptions |
| **4_Feature_Analysis.py** | Page title, V1/V2 dimension tables, feature dimension chart, V2 feature groups (Graph Topology, Propagation, MiniLM), V1 feature groups (TF-IDF, MiniLM, Graph), key insight |
| **5_Experimental_Results.py** | Page title, final metrics section, ablation study, confusion matrices, pre-generated figures, key findings summary |

## Business Logic Preserved

- No `set_page_config()` was modified
- No data loading functions were touched
- No model prediction code was altered
- No chart rendering logic was changed
- All function signatures remain identical
- All import statements are untouched