# Current Pipeline State

## Completed Pipeline Stages

### Stage 1: Data Preprocessing ✅

- **Input**: Raw PHEME dataset
- **Output**: `data/processed/pheme_clean.csv` (102,440 rows)
- **Processing**: Deduplication, text cleaning, label normalization
- **Status**: Frozen - do not modify without regression testing

### Stage 2: Feature Engineering ✅

- **Output**: `data/processed/pheme_features.csv` (17 columns)
- **Features**:
  - `post_id`, `user_id`, `thread_id`, `event_id` (IDs)
  - `text`, `time`, `reply_to` (raw data)
  - `label` (0=non-rumor, 1=rumor)
  - `depth`, `children_count`, `time_since_source` (propagation)
  - `is_source`, `thread_size`, `thread_duration_hours` (thread stats)
  - `first_reply_time_seconds`, `reply_speed_per_hour`, `max_depth` (thread dynamics)

### Stage 3: Knowledge Graph Construction ✅

- **Input**: `data/processed/pheme_features.csv`
- **Output**: `data/processed/pheme_kg.ttl` (1,065,885 triples)
- **Pipeline**: `knowledge_graph/build_kg.py`
- **Validation**: Dataset validation → Graph building → Cycle detection → RDF validation
- **Status**: Stable - all bugs fixed

### Stage 4: Baseline Model ✅

- **Model**: Logistic Regression (balanced class weights)
- **Features**: TF-IDF (5000 features) + 4 propagation features
- **Split**: 80/20 stratified train/test (RANDOM_STATE=42)
- **Notebook**: `notebooks/03_rumor_detection_baseline_final_fixed.ipynb`
- **Status**: Frozen baseline for comparison

## Exact Stable Statistics

| Metric                | Value          |
| --------------------- | -------------- |
| Total posts           | 102,440        |
| Unique threads        | 5,802          |
| Unique users          | 49,345         |
| Unique events         | 5              |
| Non-rumor posts       | 71,210 (69.5%) |
| Rumor posts           | 31,230 (30.5%) |
| Root posts (depth=0)  | 36,875         |
| Reply posts (depth>0) | 65,565         |
| Max thread depth      | 16             |
| KG triples            | 1,065,885      |
| repliesTo edges       | 65,565         |

## Frozen Preprocessing Logic

```python
# Text cleaning (do not modify)
def clean_text(text):
    if pd.isna(text): return ""
    text = text.lower()
    text = ' '.join(text.split())
    return text

# Label handling (do not modify)
y = df['label']  # Already numeric (0, 1)

# Train/test split (do not modify)
X_train_idx, X_test_idx, y_train, y_test = train_test_split(
    np.arange(len(df)), y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF (do not modify)
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000, stop_words='english',
    ngram_range=(1, 2), min_df=2, max_df=0.95
)
```

## Current Feature Stack

| Feature Type   | Count     | Description                       |
| -------------- | --------- | --------------------------------- |
| TF-IDF         | 5,000     | Text n-grams (unigrams + bigrams) |
| is_reply       | 1         | Binary: is this post a reply?     |
| thread_size    | 1         | Number of posts in thread         |
| children_count | 1         | Number of direct replies          |
| depth          | 1         | Distance from root post           |
| **Total**      | **5,004** | Combined feature vector           |

## Current Classifier

```python
LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight='balanced'  # Handles 70/30 class imbalance
)
```

## Critical Files Used in Training

1. `data/processed/pheme_clean.csv` - Clean dataset
2. `data/processed/pheme_features.csv` - Feature dataset
3. `notebooks/03_rumor_detection_baseline_final_fixed.ipynb` - Training notebook
4. `knowledge_graph/build_kg.py` - KG construction (for future graph features)
5. `ontology/pheme_ontology_v1.ttl` - Ontology (for future semantic features)
