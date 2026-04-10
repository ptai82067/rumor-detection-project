# BERT + Graph Feature Fusion Plan

## Goal

Improve rumor recall beyond 0.7775 by replacing TF-IDF lexical signals
with BERT [CLS] semantic embeddings while preserving graph handcrafted signals.

## Proposed Architecture

BERT text encoder ([CLS]) + normalized graph features + propagation features
→ concatenation → classifier

## Why This Matters

TF-IDF cannot capture semantic ambiguity in short social posts.
BERT may recover rumor cases missed due to lexical sparsity.

## Current best graph features

- position_in_thread
- user_post_count
- source_network_size
- user_prior_rumor_ratio

## Experiment Matrix

1. BERT only
2. BERT + propagation
3. BERT + graph
4. BERT + propagation + graph

## Success Criteria

- Recall > 0.7775
- FN < 1390
- no data leakage
- same split as frozen baseline
