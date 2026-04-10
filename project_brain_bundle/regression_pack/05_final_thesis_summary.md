# Milestone 05 FINAL: Thesis-Ready Hybrid Results Summary

## Executive Summary

**Status**: ✅ Complete — Final validated results for BERT + Graph Feature Fusion.

**Key Finding**: The restored graph topology signal produces a **+30.4% recall improvement** when added to MiniLM embeddings, recovering **1,900 false negatives**. The full hybrid model achieves **96.2% accuracy** and **96.1% rumor recall**.

## Final Results

| System | Accuracy | Precision | Recall | F1-Score | ROC-AUC | MCC |
|--------|----------|-----------|--------|----------|---------|-----|
| A: TF-IDF + Prop | 0.8065 | 0.6545 | 0.7735 | 0.7090 | 0.8847 | 0.5700 |
| B: MiniLM Only | 0.6865 | 0.4894 | 0.6543 | 0.5600 | 0.7413 | 0.3325 |
| C: MiniLM + Graph | 0.9591 | 0.9118 | 0.9585 | 0.9346 | 0.9929 | 0.9055 |
| **D: Hybrid** | **0.9623** | **0.9194** | **0.9606** | **0.9396** | **0.9941** | **0.9127** |

## Ablation Analysis

### Graph Contribution to MiniLM (C vs B)

| Metric | Gain |
|--------|------|
| Accuracy | +0.2726 |
| Precision | +0.4224 |
| Recall | +0.3042 |
| F1-Score | +0.3746 |
| ROC-AUC | +0.2516 |
| MCC | +0.5729 |

**False Negative Reduction**: 1,900 rumors recovered

## Deliverables

- `notebooks/05_final_hybrid_results_thesis.ipynb` — Complete thesis-ready notebook
- `notebooks/figures/` — Publication-quality figures (5 figures)
- `project_brain_bundle/regression_pack/05B_results.json` — Raw results data

## Reproducibility

- **RANDOM_STATE**: 42
- **Dataset**: data/processed/pheme_features_with_graph.csv (102,440 × 31)
- **KG triples**: 1,065,885
- **All results on same frozen test set**

---
*Generated: 2026-04-08*
*Milestone: 05 FINAL*
*Status: ✅ Thesis-Ready*
