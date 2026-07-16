"""
evaluation/metrics.py
=====================
Offline evaluation metrics for recommender systems.

Metrics:
  Precision@k  — of the top-k items recommended, what fraction are relevant?
  Recall@k     — of all relevant items for this user, what fraction did we retrieve?
  NDCG@k       — Normalised Discounted Cumulative Gain (rewards relevant items ranked higher)
  Coverage     — what fraction of the item catalogue is recommended to at least one user?
  Novelty      — how unpopular (on average) are the recommended items?

All metrics are macro-averaged across users (mean over users, not interactions).
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from tqdm import tqdm


# ── Per-user metrics ──────────────────────────────────────────

def precision_at_k(recommended: List[int], relevant: set, k: int) -> float:
    """Fraction of top-k items that are relevant."""
    if not relevant or k == 0:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / k


def recall_at_k(recommended: List[int], relevant: set, k: int) -> float:
    """Fraction of relevant items that appear in top-k."""
    if not relevant or k == 0:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: List[int], relevant: set, k: int) -> float:
    """
    NDCG@k: rewards relevant items at higher ranks more than lower ranks.
    DCG@k  = sum_{i=1}^{k} rel_i / log2(i+1)
    IDCG@k = DCG of a perfect ranking (all relevant items at top)
    NDCG@k = DCG@k / IDCG@k
    """
    if not relevant or k == 0:
        return 0.0

    dcg = sum(
        1.0 / np.log2(rank + 2)
        for rank, item in enumerate(recommended[:k])
        if item in relevant
    )
    n_ideal = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(rank + 2) for rank in range(n_ideal))

    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(recommended: List[int], relevant: set, k: int) -> float:
    """Binary: 1 if any relevant item appears in top-k, else 0."""
    return float(any(item in relevant for item in recommended[:k]))


# ── Full evaluation loop ──────────────────────────────────────

def evaluate(
    recommendations: Dict[int, List[int]],
    test: pd.DataFrame,
    train: pd.DataFrame,
    k_values: List[int] = [5, 10, 20],
) -> pd.DataFrame:
    """
    Evaluate a set of recommendations across all users in the test set.

    Parameters
    ----------
    recommendations : dict {user_idx → [item_idx, ...]}  (pre-computed)
    test            : test split (each user's held-out interaction)
    train           : train split (used to compute item popularity for novelty)
    k_values        : list of cutoffs to evaluate at

    Returns
    -------
    pd.DataFrame with one row per k value, columns = metric names + 'k'
    """

    # Ground truth: one relevant item per user (leave-one-out)
    user_ground_truth: Dict[int, set] = (
        test.groupby("user_idx")["movie_idx"]
        .apply(set).to_dict()
    )

    # Item popularity (interaction count) for novelty calculation
    item_popularity = train.groupby("movie_idx").size().to_dict()
    max_pop = max(item_popularity.values()) if item_popularity else 1

    results = []
    for k in k_values:
        metrics_per_user = {
            "precision": [],
            "recall":    [],
            "ndcg":      [],
            "hit_rate":  [],
        }

        all_recommended = set()

        for user_idx, recs in recommendations.items():
            relevant = user_ground_truth.get(user_idx, set())
            if not relevant:
                continue

            metrics_per_user["precision"].append(precision_at_k(recs, relevant, k))
            metrics_per_user["recall"].append(recall_at_k(recs, relevant, k))
            metrics_per_user["ndcg"].append(ndcg_at_k(recs, relevant, k))
            metrics_per_user["hit_rate"].append(hit_rate_at_k(recs, relevant, k))
            all_recommended.update(recs[:k])

        n_items = train["movie_idx"].nunique()
        coverage = len(all_recommended) / n_items if n_items > 0 else 0.0

        # Novelty: mean self-information of recommended items
        # Higher novelty = more niche items recommended
        novelty_scores = []
        for user_idx, recs in recommendations.items():
            for item in recs[:k]:
                pop  = item_popularity.get(item, 1)
                novelty_scores.append(-np.log2(pop / max_pop + 1e-10))
        novelty = np.mean(novelty_scores) if novelty_scores else 0.0

        results.append({
            "k":         k,
            "precision": np.mean(metrics_per_user["precision"]),
            "recall":    np.mean(metrics_per_user["recall"]),
            "ndcg":      np.mean(metrics_per_user["ndcg"]),
            "hit_rate":  np.mean(metrics_per_user["hit_rate"]),
            "coverage":  coverage,
            "novelty":   novelty,
            "n_users":   len(metrics_per_user["precision"]),
        })

    df = pd.DataFrame(results)
    for col in ["precision", "recall", "ndcg", "hit_rate", "coverage", "novelty"]:
        df[col] = df[col].round(4)
    return df


def print_results(name: str, df: pd.DataFrame):
    """Pretty-print evaluation results."""
    print(f"\n  {'─'*55}")
    print(f"  {name}")
    print(f"  {'─'*55}")
    print(f"  {'k':>4}  {'Prec':>7}  {'Recall':>7}  {'NDCG':>7}  {'Hit@k':>7}  {'Cov':>6}  {'Nov':>6}")
    print(f"  {'─'*55}")
    for _, row in df.iterrows():
        print(f"  {int(row['k']):>4}  "
              f"{row['precision']:>7.4f}  "
              f"{row['recall']:>7.4f}  "
              f"{row['ndcg']:>7.4f}  "
              f"{row['hit_rate']:>7.4f}  "
              f"{row['coverage']:>6.3f}  "
              f"{row['novelty']:>6.3f}")
