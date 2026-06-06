# CONTEXT HIỆN TẠI — Rumor Detection using Ontology & Knowledge Graph

> File này mô tả toàn bộ trạng thái hiện tại của dự án.
> Cline mới: Đọc file này TRƯỚC khi làm bất cứ điều gì.
> Sau khi đọc xong, CHỜ người dùng giao nhiệm vụ cụ thể.

════════════════════════════════════════════════════════
## 1. THÔNG TIN ĐỀ TÀI
════════════════════════════════════════════════════════

| Mục | Giá trị |
|-----|---------|
| Sinh viên | Phạm Phước Tài — MSSV 64132083 |
| Trường | Đại học Nha Trang — Ngành Hệ thống thông tin |
| GVHD | TS. Phạm Thị Thu Thúy |
| Đề tài | Phát hiện tin đồn trên mạng xã hội sử dụng Ontology và Knowledge Graph |
| Bài toán | Thread-level binary classification: Rumor (1) vs Non-Rumor (0) |
| Dataset | PHEME — 5 events, 5,802 threads, 102,440 tweets |
| GitHub | https://github.com/ptai82067/rumor-detection-project |

════════════════════════════════════════════════════════
## 2. PIPELINE KỸ THUẬT CHÍNH THỨC
════════════════════════════════════════════════════════

Đầu vào: PHEME conversation threads (JSON)

3 nhánh feature song song:
  [A] Semantic  : MiniLM (all-MiniLM-L6-v2) → 384 chiều
  [B] Propagation: thread_size, max_depth, avg_depth, reply_rate → 4 chiều
  [C] Graph/KG  : 14 handcrafted features trích từ KG v2 → 14 chiều

→ Concatenate: 384 + 4 + 14 = 402 chiều
→ Classifier: LogisticRegression(C=1.0, max_iter=1000, random_state=42)
→ Split: Stratified train/test theo thread_id, test_size=0.2, random_state=42

### Kết quả Full Hybrid (KG v2, đã xác nhận chạy lại):
  Accuracy  = 98.11%
  Precision = 0.9794
  Recall    = 0.9646
  F1        = 0.9719
  FN        = 14

### Đóng góp của KG v2 so với MiniLM alone:
  - Recall tăng: 0.7468 → 0.9646 (+21.78%)
  - FN giảm: 100 → 14 (giảm 86 cases)

════════════════════════════════════════════════════════
## 3. ONTOLOGY v2 — ĐÃ FINALIZE
════════════════════════════════════════════════════════

File: `ontology/pheme_ontology_v2.ttl` (372 dòng, Turtle)

Classes (7):
  Post, SourcePost (subclass), ReplyPost (subclass),
  ConversationThread, User, Event, VeracityLabel

Object Properties (11):
  postedBy, aboutEvent, repliesTo, inThread, hasVeracity,
  hasSourcePost, hasReply, belongsToEvent, hasStance,
  participatesInThread, hasParentPost

Data Properties (32):
  Post(12): postId, tweetText, textLength, isSourcePost,
            isReply, replyCount, subtreeReplyCount,
            siblingCount, positionInThread, depth,
            createdAt, timeSinceSource
  User(7) : userId, userPostCount, userFollowerCount,
            userFollowingCount, userVerified,
            userPriorRumorRatio, userThreadCount
  Thread(4): threadId, rumorLabel, threadSize, maxDepth
  Event(3) : eventId, eventName, eventType
  Graph(6) : centralityScore, pagerankScore,
             childrenCount, subtreeSize,
             responseTime, networkCentrality

════════════════════════════════════════════════════════
## 4. KNOWLEDGE GRAPH v2 — ĐÃ FINALIZE
════════════════════════════════════════════════════════

File: `data/processed/pheme_kg_v2.ttl` (~120MB, Turtle)
Prefix: http://example.org/pheme# (namespace EX)

Thống kê:
  - Tổng triples: 2,732,764
  - SourcePost nodes: 5,802
  - ReplyPost nodes: 96,638
  - User nodes: 49,345
  - Event nodes: 5
  - Thread nodes: 5,802
  - repliesTo edges: 58,070
  - participatesInThread: 67,509

Đặc điểm quan trọng:
  - User nodes có 7 data properties (userPriorRumorRatio, userThreadCount,...)
  - Thread nodes có rumorLabel, belongsToEvent, hasSourcePost
  - Phân biệt rõ SourcePost vs ReplyPost
  - Không có cycle trong reply tree (đã kiểm tra)
  - 100% threads join được với feature pipeline

════════════════════════════════════════════════════════
## 5. CẤU TRÚC THƯ MỤC (chỉ file quan trọng)
════════════════════════════════════════════════════════

```
rumor-detection/
├── ontology/
│   ├── pheme_ontology_v1.ttl          ← Ontology v1 (cũ)
│   └── pheme_ontology_v2.ttl          ← Ontology v2 FINAL (DÙNG CÁI NÀY)
├── knowledge_graph/
│   ├── build_kg.py                    ← KG v1 builder
│   └── build_kg_v2.py                 ← KG v2 builder (DÙNG CÁI NÀY)
├── data/
│   ├── raw/                           ← PHEME JSON gốc (gitignored)
│   └── processed/
│       ├── pheme_clean.csv            ← Data cleaned (~21MB)
│       ├── pheme_features.csv         ← Feature dataset 17 cột (~28MB)
│       ├── pheme_features_with_graph.csv ← 31 cột (~36MB)
│       ├── pheme_kg_v2.ttl            ← KG v2 (~120MB, gitignored)
│       ├── graph_features_v2.csv      ← 14 graph features từ KG v2 (558KB)
│       └── minilm_embeddings_thread.npy ← MiniLM embeddings (8.9MB)
├── src/
│   └── extract_graph_features_v2.py   ← Extract 14 features từ KG v2
├── preprocessing/
│   ├── loader.py, parser.py, normalizer.py, features.py
│   ├── propagation_features.py        ← 4 propagation features
│   └── run_pipeline.py                ← Pipeline orchestrator
├── utils/
│   ├── graph_features.py              ← Graph features (version cũ, post-level)
│   ├── bert_feature_fusion.py         ← BERT fusion utilities
│   └── visualization/
│       ├── run_all_figures.py         ← Chạy 4 scripts sinh figure
│       ├── plot_class_hierarchy.py    ← Figure A: Class hierarchy
│       ├── plot_ontology_diagram.py   ← Figure B: Ontology diagram
│       ├── plot_kg_sample.py          ← Figure C: KG sample subgraph
│       ├── plot_kg_statistics.py      ← Figure D: KG statistics
│       └── __init__.py
├── notebooks/
│   ├── ablation_study.ipynb           ← ABLATION STUDY (chạy được)
│   ├── 03_rumor_detection_baseline_final_fixed.ipynb ← Baseline
│   ├── 04_graph_feature_integration.ipynb ← Graph features cũ
│   ├── 05_final_hybrid_results_thesis.ipynb ← Kết quả hybrid cũ
│   ├── 09_final_thesis_results_synthesis.ipynb ← Tổng hợp kết quả
│   ├── pheme_dataset_statistics.ipynb ← Thống kê dataset
│   └── figures/                       ← Figures từ notebooks
├── results/
│   ├── ablation/
│   │   ├── ablation_table.csv         ← Bảng kết quả ablation CSV
│   │   ├── ablation_table.md          ← Bảng kết quả ablation MD
│   │   └── figures/
│   │       ├── fig1_ablation_bar_metrics.png
│   │       ├── fig2_recall_fn_trend.png
│   │       ├── fig3_confusion_matrices.png
│   │       └── fig4_feature_contribution.png
│   ├── pheme_dataset_statistics.csv
│   └── pheme_dataset_statistics.md
├── docs/
│   ├── VSCODE_EXTENSIONS_RECOMMENDED.md
│   └── figures/                       ← 4 PNG từ utils/visualization/
├── project_brain_bundle/              ← Context documents
│   ├── 00_PROJECT_OVERVIEW.md         ← Tổng quan dự án
│   ├── 01_CURRENT_PIPELINE_STATE.md   ← Pipeline state (cũ)
│   ├── 04_BASELINE_RESULTS.md         ← Baseline results
│   ├── 08_GRAPH_FEATURE_RESULTS.md    ← Graph feature results
│   ├── 09_BERT_GRAPH_EXTENSION_PLAN.md ← Extension plan
│   ├── 10_SESSION_HANDOFF_BERT_STAGE.md ← Handoff (cũ)
│   └── 11_CURRENT_CONTEXT.md          ← FILE NÀY (context mới nhất)
├── run_ablation.py                    ← Runner chạy ablation study
├── .gitignore                         ← Large files ignored (KG .ttl, embeddings .npy)
├── requirements.txt
└── README.md
```

════════════════════════════════════════════════════════
## 6. TIẾN ĐỘ HIỆN TẠI (5 nhóm)
════════════════════════════════════════════════════════

✅ Nhóm 1 — Ontology v2: ĐÃ XONG
  - File: ontology/pheme_ontology_v2.ttl
  - 7 classes, 11 object properties, 32 data properties

✅ Nhóm 2 — KG v2: ĐÃ XONG
  - File: data/processed/pheme_kg_v2.ttl (~120MB)
  - 2,732,764 triples
  - Script: knowledge_graph/build_kg_v2.py

✅ Nhóm 3 — Visualization (4 PNG): ĐÃ XONG
  - Output: docs/figures/ (4 files)
  - Scripts: utils/visualization/ (6 files)
  - Chạy bằng: .venv\Scripts\python utils/visualization/run_all_figures.py

✅ Nhóm 4 — Dataset Statistics: ĐÃ XONG
  - Notebook: notebooks/pheme_dataset_statistics.ipynb
  - Results: results/pheme_dataset_statistics.csv + .md
  - Script hỗ trợ: notebooks/compute_pheme_statistics.py, quick_pheme_stats.py

✅ Nhóm 5 — Ablation Study: ĐÃ XONG
  - Notebook: notebooks/ablation_study.ipynb
  - Script: src/extract_graph_features_v2.py + run_ablation.py
  - Features: data/processed/graph_features_v2.csv
  - Figures: results/ablation/figures/ (4 PNG)
  - Tables: results/ablation/ablation_table.csv + .md
  - Embeddings: data/processed/minilm_embeddings_thread.npy

❌ Nhóm 6 — Báo cáo Word: CHƯA BẮT ĐẦU
  - Đây là nhiệm vụ tiếp theo

════════════════════════════════════════════════════════
## 7. KẾT QUẢ ABLATION STUDY (Nhóm 5)
════════════════════════════════════════════════════════

| Config | Features | Acc | Prec | Recall | F1 | FN |
|--------|----------|-----|------|--------|----|----|
| TF-IDF | 10K | 0.8269 | 0.8368 | 0.6101 | 0.7057 | 154 |
| MiniLM | 384 | 0.8338 | 0.7603 | 0.7468 | 0.7535 | 100 |
| +Propagation | 384+4 | 0.8355 | 0.7642 | 0.7468 | 0.7554 | 100 |
| +Graph(KG v2) | 384+14 | 0.9811 | 0.9794 | 0.9646 | 0.9719 | 14 |
| Full Hybrid ★ | 384+4+14 | 0.9811 | 0.9794 | 0.9646 | 0.9719 | 14 |

Nhận xét:
- Graph features từ KG v2 đóng góp cực kỳ lớn: Recall +0.2178, FN -86
- Propagation features đóng góp rất ít trên nền MiniLM
- Full Hybrid gần như tương đương MiniLM+Graph (propagation không thêm giá trị)

════════════════════════════════════════════════════════
## 8. FILES QUAN TRỌNG CHO Cline MỚI
════════════════════════════════════════════════════════

### Scripts có thể chạy trực tiếp:
  .venv\Scripts\python src/extract_graph_features_v2.py
  .venv\Scripts\python run_ablation.py
  .venv\Scripts\python utils/visualization/run_all_figures.py

### Notebooks có thể mở và chạy:
  notebooks/ablation_study.ipynb
  notebooks/pheme_dataset_statistics.ipynb

### Data files chính:
  data/processed/pheme_features.csv (post-level, 102,440 rows)
  data/processed/graph_features_v2.csv (thread-level, 5,802 rows)
  data/processed/minilm_embeddings_thread.npy (5,802, 384)

════════════════════════════════════════════════════════
## 9. QUYẾT ĐỊNH ĐÃ KHÓA — KHÔNG THAY ĐỔI
════════════════════════════════════════════════════════

- Classifier: Logistic Regression (không dùng deep learning)
- Feature dim: 384 + 4 + 14 = 402 chiều
- Đánh giá: Thread-level (không phải tweet-level)
- KG format: Turtle (.ttl), prefix http://example.org/pheme#
- Ontology: v2 final, không mở rộng thêm
- Split logic: random_state=42, stratified theo thread_id
- Graph feat: PHẢI trích từ KG v2 (dùng graph_features_v2.csv)

════════════════════════════════════════════════════════
## 10. YÊU CẦU CỦA GVHD CẦN ĐÁP ỨNG
════════════════════════════════════════════════════════

1. So sánh 5 bước tăng dần (TF-IDF → Full Hybrid) ★ ĐÃ LÀM
2. Nhấn mạnh Recall và False Negatives ★ ĐÃ LÀM
3. Có bảng Ablation Study chỉ rõ đóng góp từng nhóm features ★ ĐÃ LÀM
4. Trả lời: "Nhờ ontology/KG thì hệ thống phát hiện rumor tốt hơn ở
   những trường hợp nào?" ★ ĐÃ LÀM (trong notebook cell cuối)
5. CHƯA LÀM: Viết báo cáo Word tổng hợp (Nhóm 6)

════════════════════════════════════════════════════════
## 11. CÁC LƯU Ý KỸ THUẬT
════════════════════════════════════════════════════════

### Môi trường:
- Python: 3.14.5 (venv: .venv\)
- OS: Windows 11
- Shell mặc định: cmd.exe (C:\Windows\system32\cmd.exe)
- Pip đã cài: sentence-transformers, rdflib, networkx, scikit-learn, pandas, numpy, matplotlib

### Large files bị gitignore (không push lên GitHub):
- data/processed/pheme_kg_v2.ttl (~120MB)
- data/processed/pheme_kg.ttl (~62MB)
- data/processed/pheme_clean.csv (~21MB)
- data/processed/pheme_features.csv (~28MB)
- data/processed/pheme_features_with_graph.csv (~36MB)
- data/processed/minilm_embeddings*.npy (~9MB)

### Unicode issue trên Windows:
- Print Vietnamese có dấu trong terminal cmd.exe bị lỗi
- Dùng tiếng Việt không dấu trong print() statements
- OK trong matplotlib titles/labels và file .ipynb

### Graph features v2 được trích từ pheme_features_with_graph.csv (không phải từ KG .ttl trực tiếp):
- Lý do: KG .ttl 120MB quá lớn để load toàn bộ
- pheme_features_with_graph.csv đã chứa đủ thông tin (reply_to, depth, user_prior_rumor_ratio,...)
- Các features được tính bằng NetworkX trên per-thread reply trees

════════════════════════════════════════════════════════
## 12. NHIỆM VỤ TIẾP THEO (Nhóm 6)
════════════════════════════════════════════════════════

Viết báo cáo Word (file .docx) tổng hợp toàn bộ kết quả.
Cần bao gồm:
- Chương 1: Giới thiệu
- Chương 2: Cơ sở lý thuyết (Ontology, KG, PHEME)
- Chương 3: Phương pháp đề xuất (pipeline 3 nhánh)
- Chương 4: Thực nghiệm & Kết quả (4 figures ablation, statistics table)
- Chương 5: Kết luận

════════════════════════════════════════════════════════
## HẾT CONTEXT
════════════════════════════════════════════════════════