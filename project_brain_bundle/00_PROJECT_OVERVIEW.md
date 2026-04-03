# Project Overview - PHEME Rumor Detection

## Project Task

Rumor detection on social media using the PHEME dataset combined with Knowledge Graph techniques to improve detection accuracy and interpretability.

## Dataset

- **Source**: PHEME dataset (5 real-world events)
- **Size**: 102,440 posts across 5,802 conversation threads
- **Classes**: Binary (Non-Rumor: 71,210, Rumor: 31,230)
- **Events**: ferguson, charliehebdo, germanwings-crash, ottawashooting, sydneysiege
- **Users**: 49,345 unique users

## Current Completed Phase

✅ **Phase 1 Complete**: Baseline model with text + propagation features

- Logistic Regression with TF-IDF (5000 features) + 4 propagation features
- All critical bugs fixed (data leakage, index mapping, label handling)
- Proper train/test split with stratification (80/20)

## Stable Dataset State

- **Clean CSV**: `data/processed/pheme_clean.csv` (102,440 rows)
- **Features CSV**: `data/processed/pheme_features.csv` (102,440 rows, 17 columns)
- **No duplicates**: All post_ids are unique
- **No NaN in critical fields**: post_id, user_id, event_id, thread_id, label

## Stable KG Status

✅ **Knowledge Graph Built**: `data/processed/pheme_kg.ttl`

- **Total triples**: 1,065,885
- **Posts**: 102,440 | **Users**: 49,345 | **Events**: 5 | **Threads**: 5,802
- **Reply relationships**: 65,565
- **Ontology**: `ontology/pheme_ontology_v1.ttl` (5 classes, 5 object properties, 8 datatype properties)
- **Veracity labels**: ex:NonRumor and ex:Rumor as individuals (not classes)

## Next Milestone

🎯 **Graph Feature Integration**: Add KG-derived features to baseline model

- Node centrality features (in-degree, out-degree)
- Thread structure features (depth, subtree size)
- User reputation features (prior rumor ratio)
- Source authority scores

## Primary Evaluation Metric

**Recall for Rumor class** - Minimizing false negatives (missing actual rumors) is critical for misinformation detection.

## Key Files

| File                                                      | Purpose                  |
| --------------------------------------------------------- | ------------------------ |
| `knowledge_graph/build_kg.py`                             | KG construction pipeline |
| `notebooks/03_rumor_detection_baseline_final_fixed.ipynb` | Baseline model           |
| `ontology/pheme_ontology_v1.ttl`                          | Ontology definition      |
| `data/processed/pheme_features.csv`                       | Feature dataset          |
| `logs/kg_build_after_fix.log`                             | KG build logs            |
