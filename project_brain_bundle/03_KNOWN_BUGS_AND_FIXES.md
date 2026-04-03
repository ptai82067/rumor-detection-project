# Known Bugs and Fixes

## Critical Bugs (Fixed)

### Bug 1: Label Handling - CRITICAL

**Symptom**: `ValueError: With n_samples=0 test samples`
**Root Cause**: Code tried to map numeric labels (0, 1) as strings using `.map({'non-rumor': 0, 'rumor': 1})`
**Fix**: Use labels directly since they're already numeric (0, 1)
**File**: `notebooks/03_rumor_detection_baseline_final_fixed.ipynb`
**Impact**: Model was completely broken - could not train

### Bug 2: Data Leakage

**Symptom**: Inflated performance metrics
**Root Cause**: TF-IDF vectorizer was fitted on entire dataset before train-test split
**Fix**: Fit TF-IDF only on training set, transform test set separately
**File**: `notebooks/03_rumor_detection_baseline_final_fixed.ipynb`
**Impact**: Metrics were artificially high by ~5-10%

### Bug 3: Index Mapping Error

**Symptom**: Wrong error analysis, couldn't map predictions back to original data
**Root Cause**: Test set indices were not preserved correctly
**Fix**: Preserve original indices during train-test split using `np.arange(len(df))`
**File**: `notebooks/03_rumor_detection_baseline_final_fixed.ipynb`
**Impact**: Error analysis was unreliable

## Medium Priority Bugs (Fixed)

### Bug 4: reply_to Precision Loss

**Symptom**: 7,495 broken repliesTo edges in knowledge graph
**Root Cause**: Tweet IDs (18-digit numbers) stored as float64, causing precision loss
**Fix**: Load reply_to as string, parse with `int(float(value))` to handle scientific notation
**File**: `knowledge_graph/build_kg.py`
**Impact**: 12.9% of reply edges were missing

### Bug 5: Ontology-Label Semantic Mismatch

**Symptom**: Incorrect RDF semantics - NonRumor/Rumor treated as classes instead of individuals
**Root Cause**: Code created `ex:label/0` and `ex:label/1` URIs, then typed them as instances of NonRumor/Rumor
**Fix**: Use `ex:NonRumor` and `ex:Rumor` individuals directly as objects of hasVeracity
**File**: `knowledge_graph/build_kg.py`
**Impact**: Semantic correctness, ontology conformance

## Low Priority Bugs (Fixed)

### Bug 6: Depth Calculation Inefficiency

**Symptom**: Slow preprocessing, O(n²) complexity
**Root Cause**: Iterative depth calculation traversing tree repeatedly
**Fix**: Use BFS approach with parent-child mapping, O(n) complexity
**File**: `notebooks/03_rumor_detection_baseline_final_fixed.ipynb`
**Impact**: Performance improvement only

### Bug 7: Missing Cross-thread Validation

**Symptom**: Invalid repliesTo edges between different threads
**Root Cause**: No validation that parent post is in same thread as child
**Fix**: Added thread_id consistency check in process_post()
**File**: `knowledge_graph/build_kg.py`
**Impact**: Graph structural integrity

### Bug 8: Destructive Cycle Detection

**Symptom**: Graph mutation during cycle detection, hard to debug
**Root Cause**: Algorithm removed all repliesTo edges and rebuilt graph
**Fix**: Separate detection from removal, only remove specific cycle edges
**File**: `knowledge_graph/build_kg.py`
**Impact**: Internal safety, maintainability

## Regression Prevention

### Validation Rules (All Enforced)

```python
# Dataset validation
1. No duplicate post_ids
2. All required columns present
3. No NaN in critical fields (post_id, user_id, event_id, thread_id, label)

# Structural validation
4. Root posts (depth=0) have no repliesTo
5. Non-root posts have valid repliesTo
6. Parent in same thread as child
7. Parent depth < child depth
8. No self-loops
9. No cycles (DAG structure)

# Model validation
10. Both classes present in train and test
11. TF-IDF fit on train only
12. Metrics computed on test only
```

### Test Commands

```bash
# Verify KG build
python -m knowledge_graph.build_kg

# Check log file
cat logs/kg_build_after_fix.log | tail -20

# Verify triple count
python -c "from rdflib import Graph; g = Graph(); g.parse('data/processed/pheme_kg.ttl'); print(len(g))"
# Expected: 1065885
```

## Files Modified by Fixes

| File                                                      | Bugs Fixed | Lines Changed    |
| --------------------------------------------------------- | ---------- | ---------------- |
| `knowledge_graph/build_kg.py`                             | 4, 5, 7, 8 | ~150 lines       |
| `notebooks/03_rumor_detection_baseline_final_fixed.ipynb` | 1, 2, 3, 6 | Complete rewrite |
| `logs/kg_build_after_fix.log`                             | New file   | Logging output   |

## Verification Checklist

- [x] Model trains without errors
- [x] Test set has both classes
- [x] Metrics are computed on test set only
- [x] KG has 1,065,885 triples
- [x] No cycles in reply graph
- [x] All repliesTo edges are within same thread
- [x] Log file generated at `logs/kg_build_after_fix.log`
