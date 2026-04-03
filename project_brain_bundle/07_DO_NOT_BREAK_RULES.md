# Do Not Break Rules

## Critical Constraints for Future AI Sessions

### 🚫 NEVER Modify These Files

```
data/processed/pheme_clean.csv      # Frozen clean dataset
data/processed/pheme_features.csv   # Frozen feature dataset
data/processed/pheme_kg.ttl         # Frozen knowledge graph
```

**Rationale**: These files are the ground truth for all experiments. Modifying them invalidates all previous results and comparisons.

### 🚫 NEVER Change These Parameters

```python
RANDOM_STATE = 42           # Ensures reproducible splits
test_size = 0.2             # Maintains 80/20 split
stratify = y                # Preserves class distribution
class_weight = 'balanced'   # Handles class imbalance
```

**Rationale**: Changing these breaks comparability with baseline results.

### 🚫 NEVER Modify Label Mapping

```python
# Labels are ALREADY numeric (0, 1)
# DO NOT use: df['label'].map({'non-rumor': 0, 'rumor': 1})
# Just use: y = df['label']
```

**Rationale**: The critical Bug #1 was caused by incorrect label mapping. Never reintroduce this.

### 🚫 NEVER Fit TF-IDF on Full Dataset

```python
# WRONG: tfidf.fit(df['text'])  # Data leakage!
# CORRECT:
tfidf.fit(df.iloc[X_train_idx]['text'])  # Fit on train only
X_train_tfidf = tfidf.transform(df.iloc[X_train_idx]['text'])
X_test_tfidf = tfidf.transform(df.iloc[X_test_idx]['text'])
```

**Rationale**: Fitting on full dataset causes data leakage and inflated metrics.

### 🚫 NEVER Skip Baseline Comparison

Before claiming improvement:

1. Run baseline with exact same parameters
2. Verify baseline metrics match expected values
3. Only then compare new model against baseline

**Rationale**: Without fair comparison, improvements are meaningless.

## Required Validation Before Any Change

### Pre-Change Checklist

- [ ] Backed up current state
- [ ] Documented exact changes to be made
- [ ] Identified which files will be modified
- [ ] Confirmed no changes to frozen files

### Post-Change Checklist

- [ ] Baseline metrics unchanged
- [ ] KG triple count still 1,065,885
- [ ] No new NaN values in critical columns
- [ ] Train/test split still stratified
- [ ] Both classes present in train and test
- [ ] All tests pass

### Regression Test Commands

```bash
# Verify KG integrity
python -c "from rdflib import Graph; g = Graph(); g.parse('data/processed/pheme_kg.ttl'); print('Triples:', len(g))"
# Expected: 1065885

# Verify dataset integrity
python -c "import pandas as pd; df = pd.read_csv('data/processed/pheme_features.csv'); print('Shape:', df.shape); print('NaN in label:', df['label'].isna().sum())"
# Expected: (102440, 17), 0

# Verify KG build still works
python -m knowledge_graph.build_kg 2>&1 | tail -5
# Expected: "Knowledge graph construction completed successfully!"
```

## Design Constraints

### No Redesign Before Graph Feature Experiment

1. Complete graph feature integration first
2. Run full ablation study
3. Analyze results thoroughly
4. Only then consider model changes

**Rationale**: Premature optimization without data is harmful.

### Every Feature Change Must Be Regression-Tested

For any new feature:

1. Add feature to pipeline
2. Run baseline (without new feature)
3. Run new model (with new feature)
4. Compare metrics
5. Document improvement (or lack thereof)

**Rationale**: Not all features help; measure before claiming improvement.

### No Architecture Changes Without Justification

Before changing model architecture:

1. Show baseline is insufficient
2. Provide theoretical justification
3. Run controlled experiment
4. Document trade-offs

**Rationale**: Complexity should be earned, not added arbitrarily.

## Documentation Requirements

### Commit Message Format

```
[Task] Brief description

- What was changed
- Why it was changed
- Impact on metrics
- Regression test results
```

### Experiment Log

For each experiment, record:

- Date and time
- Feature set used
- Model configuration
- All metrics (accuracy, precision, recall, F1, AUC)
- Observations and insights

## Emergency Recovery

If something breaks:

1. **Don't panic** - All changes are tracked
2. **Check git status** - See what was modified
3. **Revert changes** - `git checkout <file>` or `git reset`
4. **Re-run validation** - Use regression test commands
5. **Document what happened** - Add to bug report

## Contact for Clarification

If unsure whether an action violates these rules:

1. Check this document first
2. Review 03_KNOWN_BUGS_AND_FIXES.md for context
3. When in doubt, ask before proceeding
