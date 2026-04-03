# Graph Integration Plan

## Objective

Integrate Knowledge Graph-derived features into the baseline model to improve rumor detection recall.

## Current State

- ✅ Baseline model: Logistic Regression with TF-IDF + 4 propagation features
- ✅ Knowledge Graph: 1,065,885 triples, 65,565 repliesTo edges
- ✅ Ontology: 5 classes, 5 object properties, 8 datatype properties

## Candidate Graph Features

### 1. Node Centrality Features

| Feature                  | Description                  | Extraction Method                         |
| ------------------------ | ---------------------------- | ----------------------------------------- |
| `node_in_degree`         | Number of replies received   | Count repliesTo edges pointing to post    |
| `node_out_degree`        | Number of replies sent       | Always 0 or 1 (reply_to is single-valued) |
| `betweenness_centrality` | Bridge posts in conversation | NetworkX on repliesTo graph               |
| `closeness_centrality`   | Proximity to all other posts | NetworkX on thread subgraph               |
| `pagerank_score`         | Authority in reply network   | NetworkX PageRank                         |

### 2. Thread Structure Features

| Feature               | Description                   | Extraction Method          |
| --------------------- | ----------------------------- | -------------------------- |
| `thread_depth`        | Already exists                | From CSV                   |
| `subtree_reply_count` | Total replies in subtree      | BFS from post in repliesTo |
| `sibling_count`       | Posts replying to same parent | Group by reply_to value    |
| `position_in_thread`  | Order of post in thread       | Sort by time within thread |

### 3. User Reputation Features

| Feature                  | Description                   | Extraction Method                     |
| ------------------------ | ----------------------------- | ------------------------------------- |
| `user_prior_rumor_ratio` | Historical rumor rate         | Group by user_id, compute rumor ratio |
| `user_post_count`        | Total posts by user           | Group by user_id, count               |
| `user_thread_count`      | Threads user participates in  | Group by user_id, unique threads      |
| `user_avg_depth`         | Average depth of user's posts | Group by user_id, mean depth          |

### 4. Source Authority Features

| Feature                   | Description                      | Extraction Method                     |
| ------------------------- | -------------------------------- | ------------------------------------- |
| `source_user_credibility` | Credibility of thread originator | Prior rumor ratio of source user      |
| `source_spread_rate`      | How fast thread grows            | reply_speed_per_hour (already exists) |
| `source_network_size`     | Reach of source user             | Connections in repliesTo graph        |

## Integration Point in Pipeline

```python
# Add after existing propagation features, before model training
def extract_graph_features(df, kg_path="data/processed/pheme_kg.ttl"):
    """Extract features from knowledge graph and add to dataframe."""
    from rdflib import Graph, Namespace
    import networkx as nx

    # Load KG
    g = Graph()
    g.parse(kg_path, format="turtle")
    EX = Namespace("http://example.org/pheme#")

    # Build reply graph
    G = nx.DiGraph()
    for s, p, o in g.triples((None, EX.repliesTo, None)):
        source_id = extract_id(s)
        target_id = extract_id(o)
        G.add_edge(target_id, source_id)  # parent -> child

    # Extract features for each post
    features = []
    for post_id in df['post_id']:
        post_features = {}
        if post_id in G:
            post_features['in_degree'] = G.in_degree(post_id)
            post_features['out_degree'] = G.out_degree(post_id)
            post_features['pagerank'] = nx.pagerank(G).get(post_id, 0)
        else:
            post_features['in_degree'] = 0
            post_features['out_degree'] = 0
            post_features['pagerank'] = 0
        features.append(post_features)

    return pd.DataFrame(features)
```

## Ablation Strategy

### Experiment 1: Add Node Centrality Only

- Baseline + in_degree + out_degree + pagerank
- Hypothesis: Centrality features improve recall by 3-5%

### Experiment 2: Add User Reputation Only

- Baseline + user_prior_rumor_ratio + user_post_count
- Hypothesis: User features improve recall by 5-8%

### Experiment 3: Full Graph Features

- Baseline + all graph features
- Hypothesis: Combined features improve recall by 8-12%

### Experiment 4: Graph Features Only (No TF-IDF)

- Only graph features, no text
- Hypothesis: Graph alone achieves ~60% of full model performance

## Evaluation Protocol

1. **Same train/test split** (RANDOM_STATE=42, stratified 80/20)
2. **Same baseline model** (Logistic Regression, balanced weights)
3. **Same TF-IDF parameters** (max_features=5000, ngram_range=(1,2))
4. **Compare metrics**: Accuracy, Precision, Recall, F1, ROC-AUC
5. **Focus**: Recall for Rumor class (primary), F1 (secondary)

## Recall Improvement Hypothesis

### Why Graph Features Should Help

1. **Rumors have distinct propagation patterns**:
   - Rumors tend to spread faster and deeper
   - Rumor threads have more branching
   - Rumor sources have different credibility patterns

2. **Current features miss network structure**:
   - Simple depth/thread_size don't capture topology
   - User history not considered
   - Node importance in conversation ignored

3. **KG captures semantic relationships**:
   - Ontology-based reasoning about events
   - Veracity labels propagate through threads
   - User-event associations

### Expected Improvements

| Feature Set         | Expected Recall Gain |
| ------------------- | -------------------- |
| Node Centrality     | +3-5%                |
| User Reputation     | +5-8%                |
| Thread Structure    | +2-4%                |
| Full Graph Features | +8-12%               |

## Implementation Priority

1. **High**: Node centrality (easy to extract, clear impact)
2. **High**: User reputation (leverages existing user_id)
3. **Medium**: Thread structure (some overlap with existing features)
4. **Low**: Source authority (requires additional data processing)

## Risk Mitigation

- **Feature correlation**: Check for multicollinearity with existing features
- **Overfitting**: Use regularization, limit number of new features
- **Computation cost**: Pre-compute graph features, cache results
- **Missing data**: Handle posts not in KG gracefully (default values)
