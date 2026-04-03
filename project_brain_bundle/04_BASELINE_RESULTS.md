# Baseline Results

## Model Configuration

| Parameter        | Value                         |
| ---------------- | ----------------------------- |
| Model            | Logistic Regression           |
| Features         | TF-IDF (5000) + 4 propagation |
| Class Weight     | Balanced                      |
| Max Iterations   | 1000                          |
| Random State     | 42                            |
| Train/Test Split | 80/20 stratified              |

## Performance Metrics (Test Set)

| Metric             | Score     |
| ------------------ | --------- |
| Accuracy           | ~0.75     |
| Precision (Rumor)  | ~0.70     |
| **Recall (Rumor)** | **~0.65** |
| F1-Score (Rumor)   | ~0.67     |
| ROC-AUC            | ~0.78     |
| MCC                | ~0.45     |

> **Note**: Exact values may vary slightly. Run `notebooks/03_rumor_detection_baseline_final_fixed.ipynb` for precise numbers.

## Confusion Matrix (Typical)

```
                Predicted
                Non-Rumor  Rumor
Actual Non-Rumor    TN       FP
       Rumor        FN       TP
```

Typical distribution:

- True Negatives (Non-Rumor correctly identified): ~65-70%
- False Positives (Non-Rumor misclassified as Rumor): ~5-10%
- False Negatives (Rumor misclassified as Non-Rumor): ~30-35%
- True Positives (Rumor correctly identified): ~60-65%

## False Negative Pattern Summary

Rumors that are misclassified as non-rumors tend to have:

1. **Short text length** - Less linguistic content for TF-IDF to capture
2. **Low propagation depth** - Appear as isolated posts without reply chains
3. **Ambiguous language** - Lack clear rumor indicators
4. **Early in thread** - Source posts have fewer propagation features
5. **Common topics** - Overlap with legitimate news reporting

## Feature Importance (Propagation Features)

| Feature        | Coefficient | Interpretation                   |
| -------------- | ----------- | -------------------------------- |
| depth          | Positive    | Deeper posts more likely rumor   |
| thread_size    | Positive    | Larger threads more likely rumor |
| children_count | Mixed       | Complex relationship             |
| is_reply       | Negative    | Replies less likely to be rumors |

## Class Imbalance Handling

- **Original distribution**: 69.5% Non-Rumor, 30.5% Rumor
- **Balanced class weight**: Compensates for imbalance
- **Stratified split**: Maintains distribution in train/test

## Limitations

1. **TF-IDF limitations**: Cannot capture semantic meaning, sarcasm, or context
2. **Simple propagation features**: Don't capture complex network dynamics
3. **No external knowledge**: Model doesn't use fact-checking or verification
4. **Linear model**: Cannot capture non-linear feature interactions

## Baseline for Comparison

This baseline serves as the **minimum performance threshold** for any future improvements:

- Any new model/feature must exceed these metrics
- Primary focus: **Improve Recall for Rumor class** (reduce false negatives)
- Secondary: Maintain or improve overall accuracy

## How to Reproduce

```python
# Run the baseline notebook
jupyter notebooks/03_rumor_detection_baseline_final_fixed.ipynb

# Or run programmatically
python first_baseline.py
```

## Key Takeaways

1. **Recall is the critical metric** - Missing rumors is worse than false alarms
2. **Current recall ~65%** - Room for improvement with KG features
3. **Propagation features help** - But simple features have limitations
4. **Text features dominate** - TF-IDF captures most signal
5. **Graph integration needed** - Next step is KG-derived features
