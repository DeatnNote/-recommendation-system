"""Stage 2: multi-source candidates, ranking features, and a LightGBM LambdaRank model.

Candidates for each user are the union of three retrieval sources (two-tower, recent popularity,
BPR-MF), de-duplicated and with already-rated items removed. The ranker then re-orders those
candidates using features that the retrieval models can't express on their own: freshness
(what's popular right now), item quality, and how well an item's genres match the user's taste.

Every feature is computed only from events before the snapshot's cutoff.
"""
from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from .baselines import BPRMF, ItemKNN, Popularity, mask_seen, top_k
from .data import Catalog, Snapshot
from .two_tower import Retriever

DAY = 86400
PER_SOURCE = 100

FEATURES = [
    "tt_score", "tt_rank", "bpr_score", "bpr_rank", "knn_score",
    "pop_7d", "pop_30d", "pop_all", "pop_trend", "item_days_since_first", "item_mean_rating", "item_year",
    "genre_affinity", "user_n_hist", "user_mean_rating", "user_days_since_last",
    "user_gender", "user_age", "user_occupation",
    "src_tt", "src_pop", "src_bpr",
]


@dataclass
class Stage1:
    """All stage-1 models fit on one snapshot."""
    pop: Popularity
    knn: ItemKNN
    bpr: BPRMF
    tt: Retriever


def fit_stage1(cat: Catalog, snap: Snapshot, tt: Retriever, seed: int) -> Stage1:
    return Stage1(pop=Popularity(7).fit(snap, cat.n_users, cat.n_items),
                  knn=ItemKNN(200).fit(snap, cat.n_users, cat.n_items),
                  bpr=BPRMF(epochs=10, reg=1e-3, seed=seed).fit(snap, cat.n_users, cat.n_items),
                  tt=tt)


class ItemStats:
    def __init__(self, cat: Catalog, snap: Snapshot):
        h, c = snap.history, snap.cutoff
        n = cat.n_items
        cnt = lambda d: np.bincount(d.i.values, minlength=n).astype(np.float32)
        self.pop_7d = np.log1p(cnt(h[h.t >= c - 7 * DAY]))
        self.pop_30d = np.log1p(cnt(h[h.t >= c - 30 * DAY]))
        all_cnt = cnt(h)
        self.pop_all = np.log1p(all_cnt)
        self.pop_trend = np.expm1(self.pop_7d) / (np.expm1(self.pop_30d) / 30 * 7 + 1.0)
        first = h.groupby("i").t.min()
        self.days_since_first = np.full(n, 9999.0, np.float32)
        self.days_since_first[first.index.values] = (c - first.values) / DAY
        rsum = np.bincount(h.i.values, weights=h.rating.values, minlength=n)
        prior, k = h.rating.mean(), 20.0              # shrink small-sample means toward the global mean
        self.mean_rating = ((rsum + prior * k) / (all_cnt + k)).astype(np.float32)
        self.year = cat.year_bucket.astype(np.float32)


class UserStats:
    def __init__(self, cat: Catalog, snap: Snapshot):
        h, c = snap.history, snap.cutoff
        g = h.groupby("u")
        self.n_hist = np.log1p(g.size()).to_dict()
        self.mean_rating = g.rating.mean().to_dict()
        self.days_since_last = ((c - g.t.max()) / DAY).to_dict()
        G = cat.genres
        prof = np.zeros((cat.n_users, G.shape[1]), np.float32)
        np.add.at(prof, h.u.values, G[h.i.values])
        self.genre_profile = prof / (np.linalg.norm(prof, axis=1, keepdims=True) + 1e-9)


def candidates(s1: Stage1, snap: Snapshot, users, per_source: int = PER_SOURCE):
    """Union of the top `per_source` unseen items from each retrieval source."""
    tt = top_k(mask_seen(s1.tt.score(users), users, snap), per_source)
    pop = top_k(mask_seen(s1.pop.score(users), users, snap), per_source)
    bpr = top_k(mask_seen(s1.bpr.score(users), users, snap), per_source)
    out = {}
    for r, u in enumerate(users):
        out[u] = {"tt": set(tt[r]), "pop": set(pop[r]), "bpr": set(bpr[r])}
    return out


def _rank_of(scores_row, items):
    order = np.argsort(-scores_row)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return rank[items]


def build_features(cat: Catalog, s1: Stage1, snap: Snapshot, users, cands: dict, stats=None,
                   with_labels: bool = True) -> pd.DataFrame:
    istat, ustat = stats if stats is not None else (ItemStats(cat, snap), UserStats(cat, snap))
    tt_all = mask_seen(s1.tt.score(users), users, snap)
    bpr_all = mask_seen(s1.bpr.score(users), users, snap)
    knn_all = s1.knn.score(users)
    rows = []
    for r, u in enumerate(users):
        c = cands[u]
        items = np.array(sorted(c["tt"] | c["pop"] | c["bpr"]))
        f = pd.DataFrame({"user": u, "item": items})
        f["tt_score"] = tt_all[r, items]
        f["tt_rank"] = _rank_of(tt_all[r], items)
        f["bpr_score"] = bpr_all[r, items]
        f["bpr_rank"] = _rank_of(bpr_all[r], items)
        f["knn_score"] = knn_all[r, items]
        f["pop_7d"] = istat.pop_7d[items]
        f["pop_30d"] = istat.pop_30d[items]
        f["pop_all"] = istat.pop_all[items]
        f["pop_trend"] = istat.pop_trend[items]
        f["item_days_since_first"] = istat.days_since_first[items]
        f["item_mean_rating"] = istat.mean_rating[items]
        f["item_year"] = istat.year[items]
        f["genre_affinity"] = cat.genres[items] @ ustat.genre_profile[u] / (
            np.linalg.norm(cat.genres[items], axis=1) + 1e-9)
        f["user_n_hist"] = ustat.n_hist.get(u, 0.0)
        f["user_mean_rating"] = ustat.mean_rating.get(u, np.nan)
        f["user_days_since_last"] = ustat.days_since_last.get(u, np.nan)
        f["user_gender"] = cat.gender[u]
        f["user_age"] = cat.age[u]
        f["user_occupation"] = cat.occupation[u]
        f["src_tt"] = np.isin(items, list(c["tt"])).astype(np.int8)
        f["src_pop"] = np.isin(items, list(c["pop"])).astype(np.int8)
        f["src_bpr"] = np.isin(items, list(c["bpr"])).astype(np.int8)
        rows.append(f)
    df = pd.concat(rows, ignore_index=True)
    if with_labels:
        df["label"] = [int(i in snap.labels.get(u, ())) for u, i in zip(df.user.values, df.item.values)]
    return df


RANKER_PARAMS = dict(objective="lambdarank", learning_rate=0.03, num_leaves=31, min_child_samples=50,
                     subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1,
                     lambdarank_truncation_level=20)
MAX_TREES = 600


def _groups(df):
    return df.groupby("user", sort=False).size().values


def train_ranker(df: pd.DataFrame, features=FEATURES, seed: int = 0, folds: int = 5):
    """LambdaRank on per-user groups. The number of trees is picked by 5-fold cross-validation over users
    (mean NDCG@10 curve across folds), then the model is refit on all users with that many trees.

    An earlier version early-stopped on a single 20% holdout (~100 users). That was too noisy: it picked
    82, 10 and 34 trees across three seeds, and the 10-tree model scored below the popularity baseline.
    """
    df = df[df.groupby("user").label.transform("max") > 0]          # groups with no positive teach nothing
    users = df.user.unique()
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(rng.permutation(users), np.arange(len(users)) % folds))
    f = df.user.map(fold_of).values
    curves = []
    for k in range(folds):
        tr, ho = df[f != k], df[f == k]
        m = lgb.LGBMRanker(n_estimators=MAX_TREES, random_state=seed, **RANKER_PARAMS)
        m.fit(tr[features], tr.label, group=_groups(tr), eval_X=(ho[features],), eval_y=(ho.label,),
              eval_group=[_groups(ho)], eval_at=[10])
        curves.append(m.evals_result_["valid_0"]["ndcg@10"])
    mean_curve = np.mean(curves, axis=0)
    n_trees = int(np.argmax(mean_curve)) + 1
    model = lgb.LGBMRanker(n_estimators=n_trees, random_state=seed, **RANKER_PARAMS)
    model.fit(df[features], df.label, group=_groups(df))
    model.cv_ndcg10_ = float(mean_curve[n_trees - 1])
    model.n_trees_ = n_trees
    return model


def order_by_score(items, scores):
    """Highest score first; exact ties (common with tree models) broken by item id so order is deterministic."""
    return items[np.lexsort((items, -np.round(scores, 6)))]


def rank(model, df: pd.DataFrame, features=FEATURES, k: int = 200) -> dict:
    df = df.assign(score=model.predict(df[features]))
    out = {}
    for u, g in df.groupby("user", sort=False):
        out[u] = list(order_by_score(g.item.values, g.score.values)[:k])
    return out
