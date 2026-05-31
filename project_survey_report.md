# PROJECT SURVEY REPORT

## Dự án: Rumor Detection với Ontology & Knowledge Graph

---

## 1. Cấu trúc dự án

```
d:/rumor-detection-project/
│
├── .gitignore
├── .vscode/
│   └── extensions.json                          # VS Code recommended extensions
│
├── main.py                                      # Entry point (in ra hướng dẫn)
├── README.md                                    # Tổng quan dự án
├── requirements.txt                             # Python dependencies (38 packages)
│
├── config/
│   └── config.py                                # Config (hiện tại chỉ có comment)
│
├── preprocessing/                               # ★ Pipeline tiền xử lý dữ liệu
│   ├── __init__.py
│   ├── run_pipeline.py                          # Orchestrator chạy toàn bộ pipeline
│   ├── loader.py                                # Đọc dữ liệu PHEME (duyệt events/threads)
│   ├── parser.py                                # Parse JSON tweet (source + reactions)
│   ├── normalizer.py                            # Chuẩn hóa schema đồng nhất
│   ├── features.py                              # ★ TODO - chưa implement
│   └── propagation_features.py                  # ★ Trích xuất propagation features
│
├── knowledge_graph/                             # ★ Xây dựng Knowledge Graph
│   └── build_kg.py                              # ★ File chính: RDF triples từ features
│
├── ontology/                                    # ★ Ontology OWL
│   └── pheme_ontology_v1.ttl                    # ★ Ontology v1 (Turtle format)
│
├── utils/                                       # Utility functions
│   ├── __init__.py
│   ├── bert_feature_fusion.py                   # BERT embeddings + fusion features
│   └── graph_features.py                        # ★ Trích xuất graph features từ KG
│
├── models/
│   └── __init__.py                              # (trống)
│
├── data/
│   ├── interim/                                 # Dữ liệu trung gian
│   ├── processed/                               # ★ Dữ liệu đã xử lý
│   │   ├── pheme_clean.csv                      #   - Dữ liệu sạch (raw normalized)
│   │   ├── pheme_features.csv                   #   - Features + propagation
│   │   ├── pheme_features_with_graph.csv        #   - Features + graph features
│   │   └── pheme_kg.ttl                         #   - Knowledge Graph RDF triples
│   └── raw/                                     # Raw PHEME dataset (cấu trúc events/threads)
│
├── notebooks/                                   # ★ Jupyter Notebooks thực nghiệm
│   ├── 01_pheme_analysis.ipynb                  # Phân tích PHEME dataset
│   ├── 02_data_summary_statistics.ipynb         # Thống kê dữ liệu
│   ├── 03_rumor_detection_baseline*.ipynb       # Baseline models (nhiều phiên bản)
│   ├── 04_graph_feature_integration*.ipynb      # Tích hợp graph features
│   ├── 05_bert_graph_fusion.ipynb               # BERT + Graph fusion
│   ├── 05_final_hybrid_results_thesis*.ipynb    # Kết quả hybrid cuối cùng
│   ├── 05A_graph_topology_failure_audit.ipynb   # Audit lỗi topology
│   ├── 05B_bert_graph_fusion_fixed.py           # Fixed version (Python script)
│   ├── 09_final_thesis_results_synthesis*.ipynb # Tổng hợp kết quả luận văn
│   ├── run_bert_graph_fusion.py                 # Runner cho BERT fusion
│   ├── final_metrics_table.csv                  # Metrics cuối cùng
│   └── figures/                                 # Biểu đồ kết quả
│
├── project_brain_bundle/                        # Tài liệu nội bộ (session handoff)
│   ├── 00_PROJECT_OVERVIEW.md
│   ├── 01_CURRENT_PIPELINE_STATE.md
│   ├── 02_DATA_BEFORE_AFTER_FIX.md
│   ├── 03_KNOWN_BUGS_AND_FIXES.md
│   ├── 04_BASELINE_RESULTS.md
│   ├── 05_GRAPH_INTEGRATION_PLAN.md
│   ├── 06_NEXT_ACTION_PROMPTS.md
│   ├── 07_DO_NOT_BREAK_RULES.md
│   ├── 08_GRAPH_FEATURE_RESULTS.md
│   ├── 09_BERT_GRAPH_EXTENSION_PLAN.md
│   ├── 10_SESSION_HANDOFF_BERT_STAGE.md
│   └── regression_pack/
│
├── tests/                                       # (trống - chưa có test)
│
├── docs/
│   └── VSCODE_EXTENSIONS_RECOMMENDED.md         # Tài liệu extension đề xuất
│
├── ontology_diagram/                            # Output diagram (Graphviz)
├── ontology_mermaid_class_diagram.md            # Ontology class diagram (Mermaid)
├── ontology_mermaid_er_diagram.md               # Ontology ER diagram (Mermaid)
│
├── first_baseline.py                            # Script baseline đầu tiên
├── debug_graph_issue.py                         # Debug KG issues
├── validate_graph_fix.py                        # Validate KG fixes
├── generate_mermaid_diagram.py                  # Generator Mermaid diagrams
├── generate_ontology_diagram.py                 # Generator ontology diagrams
│
├── final_metrics_table.csv                      # Metrics tổng hợp
└── logs/                                        # Log files
    └── __init__.py
```

---

## 2. Trạng thái Ontology hiện tại

### 2.1. File ontology

| Thuộc tính  | Giá trị |
|-------------|---------|
| **Đường dẫn** | `d:\rumor-detection-project\ontology\pheme_ontology_v1.ttl` |
| **Định dạng** | Turtle (.ttl) |
| **Kích thước** | 157 dòng |
| **Version** | v1 |

### 2.2. Namespace/Prefix

| Prefix | URI |
|--------|-----|
| rdf | http://www.w3.org/1999/02/22-rdf-syntax-ns# |
| rdfs | http://www.w3.org/2000/01/rdf-schema# |
| owl | http://www.w3.org/2002/07/owl# |
| xsd | http://www.w3.org/2001/XMLSchema# |
| **ex** | **http://example.org/pheme#** |
| schema | http://schema.org/ |

### 2.3. Classes hiện có (5 classes)

| # | Class | Label | Comment |
|---|-------|-------|---------|
| 1 | `ex:Event` | Event | A real-world event that generates social media discussion. |
| 2 | `ex:Post` | Post | A social media post (tweet) in the PHEME dataset. |
| 3 | `ex:User` | User | A social media user who creates posts. |
| 4 | `ex:ConversationThread` | ConversationThread | A thread of conversation consisting of a root post and its replies. |
| 5 | `ex:VeracityLabel` | VeracityLabel | Label indicating whether a conversation thread is a rumor (1) or non-rumor (0). |

### 2.4. Object Properties hiện có (5 properties)

| # | Property | Label | Domain → Range | Comment |
|---|----------|-------|----------------|---------|
| 1 | `ex:postedBy` | posted by | `ex:Post` → `ex:User` | Relates a post to the user who created it. |
| 2 | `ex:aboutEvent` | about event | `ex:Post` → `ex:Event` | Relates a post to the event it discusses. |
| 3 | `ex:repliesTo` | replies to | `ex:Post` → `ex:Post` | Relates a reply post to the post it replies to. Forms a tree structure. |
| 4 | `ex:inThread` | in thread | `ex:Post` → `ex:ConversationThread` | Relates a post to the conversation thread it belongs to. |
| 5 | `ex:hasVeracity` | has veracity | `ex:ConversationThread` → `ex:VeracityLabel` | Relates a conversation thread to its veracity label (rumor/non-rumor). |

### 2.5. Data Properties hiện có (8 properties)

| # | Property | Label | Domain → Range | Comment |
|---|----------|-------|----------------|---------|
| 1 | `ex:text` | text | `ex:Post` → `xsd:string` | The text content of the post. |
| 2 | `ex:createdAt` | created at | `ex:Post` → `xsd:dateTime` | The timestamp when the post was created. |
| 3 | `ex:depth` | depth | `ex:Post` → `xsd:integer` | The depth of the post in the conversation tree (0 for root posts). |
| 4 | `ex:childrenCount` | children count | `ex:Post` → `xsd:integer` | Number of direct replies to this post. |
| 5 | `ex:timeSinceSource` | time since source | `ex:Post` → `xsd:float` | Time elapsed since the thread's source post was created (in seconds). |
| 6 | `ex:threadSize` | thread size | `ex:ConversationThread` → `xsd:integer` | Total number of posts in the conversation thread. |
| 7 | `ex:maxDepth` | max depth | `ex:ConversationThread` → `xsd:integer` | Maximum depth reached in the conversation tree. |
| 8 | `ex:replySpeed` | reply speed | `ex:ConversationThread` → `xsd:float` | Average reply rate in the thread (replies per hour). |

### 2.6. Individuals (2 instances)

| # | Individual | Type | Label |
|---|------------|------|-------|
| 1 | `ex:Rumor` | `ex:VeracityLabel` | Rumor |
| 2 | `ex:NonRumor` | `ex:VeracityLabel` | Non-Rumor |

### 2.7. Constraints/Axioms

- `ex:ConversationThread` có **exactly 1** `ex:hasVeracity` (owl:cardinality = 1)
- `ex:VeracityLabel` chỉ có thể là `ex:Rumor` hoặc `ex:NonRumor` (owl:oneOf)

### 2.8. Nhận xét - Những gì còn thiếu so với yêu cầu

| Mục | Hiện trạng | Cần bổ sung |
|-----|------------|-------------|
| **User properties** | ❌ Không có Data Property nào cho User | user_name, screen_name, followers_count, friends_count, verified, location |
| **Event properties** | ❌ Không có Data Property nào cho Event | event_name, event_date, event_location, event_type, event_description |
| **Sentiment** | ❌ Không có | hasSentiment, sentiment_score, sentiment_label |
| **Linguistic features** | ❌ Không có | hasLinguisticFeature, readability_score, subjectivity_score |
| **Propagation patterns** | ❌ Không có | hasPropagationPattern, propagation_speed, burstiness |
| **Credibility** | ❌ Không có | hasCredibilityScore, credibility_level |
| **Urgency** | ❌ Không có | hasUrgency, urgency_score |
| **Temporal relations** | ❌ Không có | hasTemporalRelation, reply_delay, time_to_first_reply |
| **Post subtypes** | ❌ Không phân biệt source/reply | SourcePost, ReplyPost (subClassOf Post) |
| **External links** | ❌ Không có | sameAs (Wikidata, DBpedia), seeAlso |

---

## 3. Trạng thái Knowledge Graph hiện tại

### 3.1. File/code xây dựng KG

| Thuộc tính | Giá trị |
|-------------|---------|
| **Đường dẫn code** | `d:\rumor-detection-project\knowledge_graph\build_kg.py` |
| **Kích thước** | 587 dòng |
| **Thư viện chính** | `rdflib` (Graph, Namespace, URIRef, Literal, RDF, RDFS, OWL, XSD) |
| **Thư viện phụ** | `pandas`, `logging` |
| **Class chính** | `KnowledgeGraphBuilder` |
| **Input** | `data/processed/pheme_features.csv` |
| **Output** | `data/processed/pheme_kg.ttl` |

### 3.2. Pipeline xây dựng KG

```
pheme_features.csv  ──→  KnowledgeGraphBuilder  ──→  pheme_kg.ttl
     (DataFrame)              .build_complete_kg()      (RDF Turtle)
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
               load_data()  validate_data()  build_knowledge_graph()
                                             validate_graph()
                                             save_graph()
```

### 3.3. Các loại triple đang được tạo

#### 3.3.1. Post triples

| Subject | Predicate | Object | Ghi chú |
|---------|-----------|--------|---------|
| `ex:post/{post_id}` | rdf:type | ex:Post | Instance declaration |
| `ex:post/{post_id}` | ex:postedBy | `ex:user/{user_id}` | User relationship |
| `ex:post/{post_id}` | ex:aboutEvent | `ex:event/{event_id}` | Event relationship |
| `ex:post/{post_id}` | ex:inThread | `ex:thread/{thread_id}` | Thread membership |
| `ex:post/{post_id}` | ex:text | `"{text}"^^xsd:string` | Text content |
| `ex:post/{post_id}` | ex:createdAt | `"{iso}"^^xsd:dateTime` | Creation time |
| `ex:post/{post_id}` | ex:depth | `{n}^^xsd:integer` | Depth in tree |
| `ex:post/{post_id}` | ex:childrenCount | `{n}^^xsd:integer` | Number of direct replies |
| `ex:post/{post_id}` | ex:timeSinceSource | `{n}^^xsd:float` | Time from source (seconds) |
| `ex:post/{post_id}` | ex:repliesTo | `ex:post/{parent_id}` | Parent post **(có validation)** |

#### 3.3.2. User triples

| Subject | Predicate | Object | Ghi chú |
|---------|-----------|--------|---------|
| `ex:user/{user_id}` | rdf:type | ex:User | Instance declaration |

> ⚠️ **User node KHÔNG có bất kỳ Data Property nào** (không có tên, followers, v.v.)

#### 3.3.3. Event triples

| Subject | Predicate | Object | Ghi chú |
|---------|-----------|--------|---------|
| `ex:event/{event_id}` | rdf:type | ex:Event | Instance declaration |

> ⚠️ **Event node KHÔNG có bất kỳ Data Property nào** (không có tên, ngày tháng, v.v.)

#### 3.3.4. ConversationThread triples

| Subject | Predicate | Object | Ghi chú |
|---------|-----------|--------|---------|
| `ex:thread/{thread_id}` | rdf:type | ex:ConversationThread | Instance declaration |
| `ex:thread/{thread_id}` | ex:threadSize | `{n}^^xsd:integer` | Total posts in thread |
| `ex:thread/{thread_id}` | ex:maxDepth | `{n}^^xsd:integer` | Max depth of thread |
| `ex:thread/{thread_id}` | ex:replySpeed | `{n}^^xsd:float` | Replies per hour |
| `ex:thread/{thread_id}` | ex:hasVeracity | `ex:Rumor` hoặc `ex:NonRumor` | Ground truth label |

### 3.4. URI Format

| Entity | URI Pattern | Ví dụ |
|--------|-------------|-------|
| Post | `http://example.org/pheme#post/{post_id}` | `ex:post/552783238415265792` |
| User | `http://example.org/pheme#user/{user_id}` | `ex:user/384779793` |
| Event | `http://example.org/pheme#event/{event_id}` | `ex:event/charliehebdo` |
| Thread | `http://example.org/pheme#thread/{thread_id}` | `ex:thread/552783238415265792` |

### 3.5. Join key giữa KG và DataFrame

| Key | Kiểu dữ liệu | Ghi chú |
|-----|-------------|---------|
| **post_id** | int (từ CSV) ↔ int (trong URI) | Join bằng cách trích xuất số từ URI |
| **user_id** | int (từ CSV) ↔ int (trong URI) | |
| **event_id** | string (từ CSV) ↔ string (trong URI) | |
| **thread_id** | int (từ CSV) ↔ int (trong URI) | |

### 3.6. Validation trong KG builder

| Validation | Mô tả | Action |
|------------|-------|--------|
| ✨ Duplicate detection | Phát hiện và bỏ qua posts/users/events/threads đã xử lý | Skip |
| ✨ Depth consistency | depth=0 → không được có repliesTo | Bỏ qua edge |
| ✨ Cross-thread check | repliesTo phải cùng thread_id | Bỏ qua edge |
| ✨ Depth ordering | parent_depth < child_depth | Bỏ qua edge |
| ✨ Cycle detection | DFS detect cycles trong repliesTo | Remove cycle edges |
| ✨ Self-loop check | post không thể repliesTo chính nó | Remove self-loop |
| ✨ Missing parent | repliesTo trỏ đến post_id không tồn tại | Cảnh báo, bỏ qua edge |

### 3.7. Nhận xét - Những gì còn thiếu so với yêu cầu

| Mục | Hiện trạng | Cần bổ sung |
|-----|------------|-------------|
| **User properties** | ❌ Chỉ có rdf:type | Thêm user_name, followers_count, verified, location |
| **Event properties** | ❌ Chỉ có rdf:type | Thêm event_name, event_date, event_location |
| **Sentiment triples** | ❌ Không có | Thêm ex:hasSentiment với sentiment score |
| **Linguistic triples** | ❌ Không có | Thêm linguistic features (readability, subjectivity) |
| **Propagation triples** | ❌ Không có | Thêm propagation_speed, burstiness |
| **SPARQL queries** | ❌ Không có | File riêng để query KG |
| **Neo4j support** | ❌ Chỉ dùng rdflib | Có thể thêm Neo4j cho graph traversal |
| **External links** | ❌ Không có | sameAs, owl:equivalentClass với Wikidata |
| **Graph statistics** | ⚠️ Có basic | Thêm density, clustering coefficient, path length |

---

## 4. Graph Features hiện tại

### 4.1. Danh sách 14 features đầy đủ

#### Nhóm 1: Node Centrality (5 features) - từ NetworkX reply graph

| # | Feature | Cách tính | Nguồn |
|---|---------|-----------|-------|
| 1 | `node_in_degree` | `G.in_degree(post_id)` - số replies trực tiếp đến post | NetworkX từ KG |
| 2 | `node_out_degree` | `G.out_degree(post_id)` - 0 (root) hoặc 1 (reply) | NetworkX từ KG |
| 3 | `pagerank_score` | `nx.pagerank(G, alpha=0.85, max_iter=100)` | NetworkX từ KG |
| 4 | `betweenness_centrality` | `nx.betweenness_centrality(G, k=1000)` (sample nếu > 10000 nodes) | NetworkX từ KG |
| 5 | `closeness_centrality` | `nx.closeness_centrality(G.to_undirected())` | NetworkX từ KG |

#### Nhóm 2: User Reputation (4 features) - từ DataFrame

| # | Feature | Cách tính | Nguồn |
|---|---------|-----------|-------|
| 6 | `user_prior_rumor_ratio` | Label sum / post count (theo user) | DataFrame |
| 7 | `user_post_count` | Count posts theo user_id | DataFrame |
| 8 | `user_thread_count` | nunique thread_id theo user | DataFrame |
| 9 | `user_avg_depth` | Mean depth theo user | DataFrame |

#### Nhóm 3: Thread Structure (3 features) - từ NetworkX + DataFrame

| # | Feature | Cách tính | Nguồn |
|---|---------|-----------|-------|
| 10 | `subtree_reply_count` | Recursive DFS: tổng số replies trong subtree của post | NetworkX từ KG |
| 11 | `sibling_count` | Số posts cùng parent (cùng repliesTo) | NetworkX từ KG |
| 12 | `position_in_thread` | cumcount() theo thời gian trong thread | DataFrame |

#### Nhóm 4: Source Authority (2 features) - từ DataFrame

| # | Feature | Cách tính | Nguồn |
|---|---------|-----------|-------|
| 13 | `source_user_credibility` | `1 - user_prior_rumor_ratio` (của source user) | DataFrame |
| 14 | `source_network_size` | `user_post_count` (của source user) | DataFrame |

### 4.2. Pipeline tích hợp features vào model

```
pheme_kg.ttl                        pheme_features.csv
     │                                    │
     ▼                                    │
build_reply_graph() ──► nx.DiGraph        │
     │                                    │
     ├── compute_node_centrality() ───►   │
     ├── compute_thread_structure() ──►   │
     │                                    ▼
     │                    extract_all_graph_features()
     │                                    │
     ▼                                    ▼
     │              pheme_features_with_graph.csv
     │                                    │
     │                                    ▼
     │              bert_feature_fusion.py
     │                                    │
     ▼                                    ▼
     │              [BERT embeddings (384)] + [graph features (14)] + [propagation features (4-6)]
     │                                    │
     ▼                                    ▼
     │              Train/Test split → Classifier → Metrics
```

### 4.3. Feature dimensions

| Feature set | Dimensions |
|-------------|-----------|
| BERT embeddings (MiniLM) | 384 |
| Propagation features | 4 (is_reply, thread_size, children_count, depth) |
| Graph features | 14 (từ graph_features.py) |
| **Full Hybrid** | **384 + 4 + 14 = 402** |

---

## 5. Dữ liệu

### 5.1. File CSV chính

| File | Đường dẫn |
|------|-----------|
| **pheme_clean.csv** | `d:\rumor-detection-project\data\processed\pheme_clean.csv` |
| **pheme_features.csv** | `d:\rumor-detection-project\data\processed\pheme_features.csv` |
| **pheme_features_with_graph.csv** | `d:\rumor-detection-project\data\processed\pheme_features_with_graph.csv` |
| **pheme_kg.ttl** | `d:\rumor-detection-project\data\processed\pheme_kg.ttl` |

### 5.2. Cấu trúc dữ liệu (từ code analysis)

**pheme_clean.csv** (output của run_pipeline.py):
- Columns: post_id, user_id, text, time, event_id, reply_to, thread_id, label

**pheme_features.csv** (output của PropagationFeatureExtractor):
- Từ pheme_clean.csv + node/thread level features:
  - Node-level: depth, children_count, time_since_source, is_source
  - Thread-level: thread_size, thread_duration_hours, first_reply_time_seconds, reply_speed_per_hour, max_depth

**pheme_features_with_graph.csv** (output của extract_all_graph_features):
- Từ pheme_features.csv + 14 graph features

### 5.3. Thống kê từ code và metrics

| Metric | Giá trị |
|--------|---------|
| Số posts trong dataset | ~6,500+ (từ training + test trong baseline) |
| Số events | 5 (charliehebdo, sydneysiege, ferguson, ottawashooting, germanwings-crash) |
| Số threads | ~+ (tóm tắt từ label distribution) |
| Label ratio | Rumor ~45% / Non-rumor ~55% (ước lượng từ baseline) |
| False Negatives (Full Hybrid) | 246 |
| Best Accuracy | 96.23% (Full Hybrid) |

---

## 6. Danh sách Notebooks

| # | Tên Notebook | Mục tiêu | Input | Output chính |
|---|-------------|----------|-------|-------------|
| 1 | `01_pheme_analysis.ipynb` | Phân tích PHEME dataset | data/raw/pheme | Thống kê events, tweets |
| 2 | `02_data_summary_statistics.ipynb` | Thống kê dữ liệu chi tiết | pheme_features.csv | HTML report |
| 3 | `03_rumor_detection_baseline.ipynb` | Baseline với TF-IDF + Propagation | pheme_features.csv | Metrics đầu tiên |
| 4 | `03_rumor_detection_baseline_fixed.ipynb` | Baseline fixed (sửa lỗi) | pheme_features.csv | Metrics improved |
| 5 | `03_rumor_detection_baseline_fixed_v2.ipynb` | Baseline v2 | pheme_features.csv | Metrics v2 |
| 6 | `03_rumor_detection_baseline_final_fixed.ipynb` | Baseline cuối cùng | pheme_features.csv | Metrics final |
| 7 | `04_graph_feature_integration.ipynb` | Tích hợp graph features | pheme_features_with_graph.csv | Metrics with graph |
| 8 | `04_graph_feature_integration_executed.ipynb` | Executed version | pheme_features_with_graph.csv | Kết quả chạy |
| 9 | `05_bert_graph_fusion.ipynb` | BERT + Graph fusion | pheme_features_with_graph.csv | Metrics fusion |
| 10 | `05_final_hybrid_results_thesis.ipynb` | Kết quả hybrid cuối cùng | pheme_features_with_graph.csv | Metrics thesis |
| 11 | `05_final_hybrid_results_thesis_executed.ipynb` | Executed thesis results | pheme_features_with_graph.csv | **Kết quả cuối cùng** |
| 12 | `05A_graph_topology_failure_audit.ipynb` | Audit lỗi topology graph | pheme_kg.ttl | Báo cáo lỗi KG |
| 13 | `09_final_thesis_results_synthesis.ipynb` | Tổng hợp kết quả | Các metrics CSVs | Báo cáo tổng hợp |
| 14 | `09_final_thesis_results_synthesis_executed.ipynb` | Executed synthesis | Các metrics CSVs | Final report |

### Kết quả metrics cuối cùng (từ `final_metrics_table.csv`)

| Model | Accuracy | Precision | Recall | F1 | ROC_AUC | MCC | FN |
|-------|----------|-----------|--------|-----|---------|-----|-----|
| TF-IDF + Propagation | 0.8065 | 0.6545 | 0.7735 | **0.7090** | 0.8847 | 0.5700 | 1415 |
| TF-IDF + Graph | 0.8092 | 0.6584 | 0.7775 | **0.7130** | 0.8858 | 0.5760 | 1390 |
| MiniLM only | 0.6865 | 0.4894 | 0.6543 | **0.5600** | 0.7413 | 0.3325 | 2159 |
| MiniLM + Graph | 0.9591 | 0.9118 | 0.9585 | **0.9346** | 0.9929 | 0.9055 | 259 |
| **Full Hybrid** | **0.9623** | **0.9194** | **0.9606** | **0.9396** | **0.9941** | **0.9127** | **246** |

---

## 7. Tổng hợp những việc cần làm (GAP ANALYSIS)

### 7.1. Ontology cần bổ sung

| Priority | Mục cần thêm | Lý do | File cần sửa |
|----------|-------------|-------|-------------|
| 🔴 Cao | Data properties cho `User` (user_name, followers_count, friends_count, verified, location) | User node hiện không có thông tin gì | `ontology/pheme_ontology_v1.ttl` |
| 🔴 Cao | Data properties cho `Event` (event_name, event_date, event_location, event_type) | Event node hiện không có thông tin gì | `ontology/pheme_ontology_v1.ttl` |
| 🟡 Trung bình | `ex:hasSentiment` Object Property + `ex:Sentiment` Class | Để gắn sentiment scores vào posts | `ontology/pheme_ontology_v1.ttl` |
| 🟡 Trung bình | `ex:Sentiment` Data Properties (sentiment_score, sentiment_label) | Sentiment giúp phát hiện tin đồn | `ontology/pheme_ontology_v1.ttl` |
| 🟡 Trung bình | `ex:hasUrgency` Data Property cho Post | Urgency là indicator của rumor | `ontology/pheme_ontology_v1.ttl` |
| 🟡 Trung bình | `ex:hasCredibilityScore` Data Property cho User | Credibility của user ảnh hưởng đến rumor detection | `ontology/pheme_ontology_v1.ttl` |
| 🟢 Thấp | `ex:SourcePost`, `ex:ReplyPost` subclass của `ex:Post` | Phân biệt source và reply posts | `ontology/pheme_ontology_v1.ttl` |
| 🟢 Thấp | Propagarion Patterns (`ex:hasPropagationPattern`, `ex:BurstPattern`, `ex:SteadyPattern`) | Pattern lan truyền khác nhau giữa rumor/non-rumor | `ontology/pheme_ontology_v1.ttl` |

### 7.2. KG cần bổ sung

| Priority | Mục cần thêm | Lý do | File cần sửa |
|----------|-------------|-------|-------------|
| 🔴 Cao | User properties triples (user_name, followers_count, verified) | Cần lấy từ PHEME dataset JSON | `knowledge_graph/build_kg.py` |
| 🔴 Cao | Event properties triples (event_name, event_date) | Cần hardcode hoặc lấy từ dataset | `knowledge_graph/build_kg.py` |
| 🟡 Trung bình | Sentiment triples cho mỗi Post | Tính từ text bằng TextBlob/VADER | `knowledge_graph/build_kg.py` |
| 🟡 Trung bình | Linguistic feature triples (readability, subjectivity) | Tính từ text content | `knowledge_graph/build_kg.py` |
| 🟡 Trung bình | Propagation triples (reply_speed cục bộ, burstiness) | Tính từ thread structure | `knowledge_graph/build_kg.py` |
| 🟢 Thấp | Thêm `ex:RetweetCount`, `ex:FavoriteCount` | Nếu có trong dữ liệu | `knowledge_graph/build_kg.py` |
| 🟢 Thấp | External links (owl:sameAs to Wikidata) | Liên kết đến tri thức ngoài | `knowledge_graph/build_kg.py` |

### 7.3. Code cần chỉnh sửa

| Priority | File | Việc cần làm | Chi tiết |
|----------|------|-------------|----------|
| 🔴 Cao | `knowledge_graph/build_kg.py` | Bổ sung User properties | Thêm ex:userName, ex:followersCount, ex:verified cho mỗi User node. Cần sửa `process_user()` để nhận thêm data từ DataFrame |
| 🔴 Cao | `knowledge_graph/build_kg.py` | Bổ sung Event properties | Thêm ex:eventName, ex:eventDate cho mỗi Event node. Cần sửa `process_event()` |
| 🟡 TB | `preprocessing/features.py` | **Implement feature extraction** | File hiện tại là TODO stub. Cần implement các feature engineering functions |
| 🟡 TB | `knowledge_graph/build_kg.py` | Thêm Sentiment triples | Tính sentiment từ text bằng TextBlob hoặc VADER, thêm ex:hasSentiment với sentiment score |
| 🟡 TB | `utils/graph_features.py` | Thêm clustering coefficient | Thêm feature từ NetworkX: `nx.clustering()`, `nx.average_clustering()` |
| 🟡 TB | `utils/graph_features.py` | Thêm graph density features | `nx.density()`, `nx.degree_assortativity_coefficient()` |
| 🟢 Thấp | `utils/bert_feature_fusion.py` | Thêm ablation config | Cho phép chạy ablation study dễ dàng hơn |
| 🟢 Thấp | `main.py` | Cập nhật CLI commands | Thêm commands cho KG build, feature extraction |
| 🟢 Thấp | Thêm file | `sparql_queries.rq` hoặc tương tự | Các SPARQL queries mẫu để query KG |
| 🟢 Thấp | Thêm file | `tests/` test files | Unit tests cho KG builder, feature extractor |

### 7.4. Tổng quan tiến độ dự án

| Module | Trạng thái | % Hoàn thành |
|--------|-----------|-------------|
| Preprocessing pipeline | ✅ Hoàn thành | 100% |
| Ontology v1 | ✅ Hoàn thành (cơ bản) | 60% (cần mở rộng) |
| Knowledge Graph builder | ✅ Hoàn thành (cơ bản) | 70% (cần thêm properties) |
| Graph feature extraction | ✅ Hoàn thành | 90% |
| BERT fusion | ✅ Hoàn thành | 90% |
| Baseline models | ✅ Hoàn thành | 100% |
| Hybrid models | ✅ Hoàn thành | 100% |
| Results metrics | ✅ Hoàn thành | 100% |
| Feature engineering | ❌ Chưa implement (`features.py`) | 0% |
| Tests | ❌ Chưa có | 0% |
| SPARQL queries | ❌ Chưa có | 0% |
| Neo4j integration | ❌ Chưa có | 0% |

---

## 8. Phụ lục: Thư viện & Dependencies

| Package | Version | Mục đích |
|---------|---------|----------|
| numpy | ≥1.21.0 | Numerical computing |
| pandas | ≥1.3.0 | Data manipulation |
| scikit-learn | ≥1.0.0 | ML models & metrics |
| torch | ≥1.9.0 | Deep learning backend |
| transformers | ≥4.0.0 | Hugging Face models |
| sentence-transformers | ≥0.4.0 | BERT embeddings (MiniLM) |
| nltk | ≥3.6 | Text processing |
| spacy | ≥3.0.0 | NLP pipeline |
| textblob | ≥0.15.0 | Sentiment analysis |
| networkx | ≥2.6 | **Graph analysis** |
| pygraphviz | ≥0.9 | **Graph visualization** |
| rdflib | ≥6.0.0 | **RDF/KG handling** |
| owlready2 | ≥0.24 | **OWL ontology** |
| matplotlib | ≥3.3.0 | Visualization |
| seaborn | ≥0.11.0 | Statistical plots |
| plotly | ≥5.0.0 | Interactive plots |
| jupyter | ≥1.0.0 | Notebooks |
| black | ≥21.0.0 | Code formatting |
| flake8 | ≥4.0.0 | Linting |

---

*Báo cáo được tạo ngày: 30/05/2026*
*Tổng hợp từ toàn bộ source code và dữ liệu dự án `rumor-detection-project`*