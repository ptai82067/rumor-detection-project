# Next Action Prompts

## Task 1: Feature Integration

**Prompt for Future Cline Session:**

```
TASK: Implement graph feature extraction from the PHEME knowledge graph.

CONTEXT:
- Knowledge graph exists at data/processed/pheme_kg.ttl (1,065,885 triples)
- Baseline model uses TF-IDF + 4 propagation features
- Goal: Add KG-derived features to improve recall

IMPLEMENT:
1. Create extract_graph_features() function that:
   - Loads KG from pheme_kg.ttl
   - Builds NetworkX DiGraph from repliesTo edges
   - Computes: in_degree, out_degree, pagerank for each post
   - Returns DataFrame with post_id and graph features

2. Integrate into baseline pipeline:
   - Add graph features to feature matrix
   - Keep all existing preprocessing unchanged
   - Maintain same train/test split (RANDOM_STATE=42)

3. Run ablation study:
   - Baseline only (existing)
   - Baseline + graph features
   - Compare metrics

DELIVERABLES:
- New file: utils/graph_features.py
- Updated notebook: notebooks/04_graph_feature_integration.ipynb
- Metrics comparison table

CONSTRAINTS:
- Do NOT modify data/processed/pheme_features.csv
- Do NOT change train/test split logic
- Do NOT modify baseline model configuration
- Always compare against frozen baseline
```

## Task 2: Experiment Comparison

**Prompt for Future Cline Session:**

```
TASK: Run comprehensive ablation study comparing graph feature variants.

CONTEXT:
- Baseline model: LogisticRegression with TF-IDF + 4 propagation features
- Graph features implemented in utils/graph_features.py
- Primary metric: Recall for Rumor class

EXPERIMENTS TO RUN:
1. Baseline (no graph features)
2. Baseline + node centrality (in_degree, out_degree, pagerank)
3. Baseline + user reputation (user_prior_rumor_ratio, user_post_count)
4. Baseline + all graph features
5. Graph features only (no TF-IDF)

FOR EACH EXPERIMENT:
- Record: Accuracy, Precision, Recall, F1, ROC-AUC
- Generate confusion matrix
- Analyze false negative patterns

DELIVERABLES:
- Results table comparing all experiments
- Bar chart of metrics across experiments
- Analysis of which features contribute most to recall improvement

CONSTRAINTS:
- Use same train/test split for all experiments
- Use same model (LogisticRegression) for fair comparison
- Document any changes to feature preprocessing
```

## Task 3: Recall Analysis

**Prompt for Future Cline Session:**

```
TASK: Deep analysis of recall improvement and false negative reduction.

CONTEXT:
- Graph features have been integrated
- Baseline recall: ~0.65
- Target recall: >0.75

ANALYZE:
1. False negative reduction:
   - Compare FN count before/after graph features
   - Identify which types of rumors are now correctly classified
   - Analyze remaining false negatives

2. Feature importance:
   - Which graph features contribute most to recall?
   - Are there features that hurt recall?
   - Interaction between text and graph features

3. Case studies:
   - Select 5 rumors that were FN before but TP after
   - Analyze what graph features enabled correct classification
   - Select 5 remaining FNs and analyze why they're still missed

DELIVERABLES:
- Detailed recall analysis report
- Case study examples with feature values
- Recommendations for further improvement

CONSTRAINTS:
- Focus on Rumor class recall (not overall accuracy)
- Use statistical tests to confirm significance
- Avoid overfitting analysis to test set
```

## Task 4: Report Writing Support

**Prompt for Future Cline Session:**

```
TASK: Generate technical report for graph feature integration results.

CONTEXT:
- All experiments completed
- Results show X% improvement in recall
- Need to document methodology and findings

REPORT SECTIONS:
1. Abstract (150 words)
2. Introduction (motivation, problem statement)
3. Related Work (rumor detection, KG integration)
4. Methodology (dataset, baseline, graph features)
5. Experiments (setup, results, analysis)
6. Discussion (implications, limitations)
7. Conclusion (summary, future work)

REQUIREMENTS:
- Include all metrics tables
- Include confusion matrices
- Include feature importance analysis
- Cite relevant papers (PHEME dataset, rumor detection)
- Follow academic writing style

DELIVERABLES:
- Full report in Markdown format
- Separate figures directory with all charts
- BibTeX file with references

CONSTRAINTS:
- Be honest about limitations
- Don't overclaim results
- Compare fairly with baseline
- Acknowledge any negative results
```

## Execution Order

1. **Task 1** (Feature Integration) - Must complete first
2. **Task 2** (Experiment Comparison) - Depends on Task 1
3. **Task 3** (Recall Analysis) - Depends on Task 2
4. **Task 4** (Report Writing) - Depends on Tasks 2-3

## Key Reminders for Future Sessions

- Always start from frozen baseline
- Never modify data/processed/ files
- Preserve RANDOM_STATE=42 for reproducibility
- Document all changes in commit messages
- Run regression tests after each change
