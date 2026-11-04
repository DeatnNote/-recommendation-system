"""Top-K ranking metrics with binary relevance, and paired bootstrap intervals over users."""
from __future__ import annotations

import numpy as np


def recall_at_k(ranked, relevant: set, k: int) -> float:
    if not relevant:
        return np.nan
    hits = sum(1 for i in ranked[:k] if i in relevant)
    return hits / min(len(relevant), k)


def ndcg_at_k(ranked, relevant: set, k: int) -> float:
    if not relevant:
        return np.nan
    dcg = sum(1.0 / np.log2(r + 2) for r, i in enumerate(ranked[:k]) if i in relevant)
    idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(relevant), k)))
    return dcg / idcg


def hit_at_k(ranked, relevant: set, k: int) -> float:
    return float(any(i in relevant for i in ranked[:k]))


def per_user(recs: dict, labels: dict, users, k_list=(10, 50)) -> dict:
    """recs: user -> ranked item list. Returns metric name -> array aligned with `users`."""
    out = {}
    for k in k_list:
        out[f"recall@{k}"] = np.array([recall_at_k(recs[u], labels[u], k) for u in users])
    out["ndcg@10"] = np.array([ndcg_at_k(recs[u], labels[u], 10) for u in users])
    out["hit@10"] = np.array([hit_at_k(recs[u], labels[u], 10) for u in users])
    return out


def bootstrap_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Mean and percentile CI, resampling users."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def coverage(recs: dict, users, n_items: int, k: int = 10) -> float:
    """Share of the catalog that appears in at least one user's top-k."""
    return len({i for u in users for i in recs[u][:k]}) / n_items
