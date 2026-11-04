"""MovieLens-1M loading and a global time-based split.

Every model and feature only sees events strictly before its cutoff. The evaluation window is
the next slice of real time, so nothing from the future leaks into training (a random or
leave-one-out split would let a model learn from ratings made after the ones it is tested on).

Split (by global event-time quantile):
    fit    : t <  q80             -> stage-1 models and ranker features for the ranker's training data
    valid  : q80 <= t < q90       -> labels used to train the ranker
    test   : t >= q90             -> final evaluation; stage-1 models are refit on everything before q90
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
POSITIVE_RATING = 4          # a future rating of 4 or 5 counts as a relevant recommendation
HISTORY_LEN = 50             # most recent items fed to the user tower


def load_raw(raw: Path = RAW):
    """Reads the official GroupLens files (ratings.dat etc., '::'-separated) if present,
    otherwise the tab-separated CSV conversion of the same dataset."""
    if (raw / "ratings.dat").exists():
        rd = lambda f, cols: pd.read_csv(raw / f, sep="::", engine="python", names=cols, encoding="latin-1")
        r = rd("ratings.dat", ["user_id", "movie_id", "rating", "timestamp"])
        m = rd("movies.dat", ["movie_id", "title", "genres"])
        u = rd("users.dat", ["user_id", "gender", "age", "occupation", "zipcode"])
        return r, m, u
    r = pd.read_csv(raw / "ratings.csv", sep="\t", index_col=0)[["user_id", "movie_id", "rating", "timestamp"]]
    m = pd.read_csv(raw / "movies.csv", sep="\t", index_col=0, encoding="latin-1")[["movie_id", "title", "genres"]]
    u = pd.read_csv(raw / "users.csv", sep="\t", index_col=0)[["user_id", "gender", "age", "occupation", "zipcode"]]
    return r, m, u


@dataclass
class Catalog:
    """Static item and user attributes, indexed by dense ids (item 0..n_items-1, user 0..n_users-1)."""
    item_ids: np.ndarray            # dense -> movie_id
    item_index: dict                # movie_id -> dense
    genres: np.ndarray              # (n_items, n_genres) multi-hot float32
    genre_names: list
    year_bucket: np.ndarray         # (n_items,) int, release year in 5-year buckets
    user_ids: np.ndarray
    user_index: dict
    gender: np.ndarray              # (n_users,) int
    age: np.ndarray                 # (n_users,) int bucket
    occupation: np.ndarray          # (n_users,) int

    @property
    def n_items(self):
        return len(self.item_ids)

    @property
    def n_users(self):
        return len(self.user_ids)


def build_catalog(m: pd.DataFrame, u: pd.DataFrame) -> Catalog:
    m = m.sort_values("movie_id").reset_index(drop=True)
    names = sorted({g for gs in m.genres for g in gs.split("|")})
    gi = {g: k for k, g in enumerate(names)}
    G = np.zeros((len(m), len(names)), np.float32)
    for row, gs in enumerate(m.genres):
        for g in gs.split("|"):
            G[row, gi[g]] = 1.0
    year = m.title.str.extract(r"\((\d{4})\)\s*$")[0].astype(float).fillna(1990).astype(int)
    yb = np.clip((year.values - 1915) // 5, 0, 20)
    u = u.sort_values("user_id").reset_index(drop=True)
    ages = {a: k for k, a in enumerate(sorted(u.age.unique()))}
    return Catalog(
        item_ids=m.movie_id.values, item_index={mid: k for k, mid in enumerate(m.movie_id.values)},
        genres=G, genre_names=names, year_bucket=yb.astype(np.int64),
        user_ids=u.user_id.values, user_index={uid: k for k, uid in enumerate(u.user_id.values)},
        gender=(u.gender.values == "M").astype(np.int64), age=u.age.map(ages).values.astype(np.int64),
        occupation=u.occupation.values.astype(np.int64))


@dataclass
class Snapshot:
    """Everything known at one cutoff time, plus the labels from the window after it."""
    cutoff: int
    history: pd.DataFrame                      # events before cutoff (dense ids, sorted by time)
    labels: dict = field(default_factory=dict)  # user -> set of items rated >= 4 in the window
    seen: dict = field(default_factory=dict)    # user -> set of items rated before cutoff
    cold_label_items: int = 0                   # positives on items with no history (unreachable by CF)

    def eval_users(self):
        """Users with history before the cutoff and at least one positive in the window."""
        return np.array(sorted(u for u in self.labels if u in self.seen))


def to_dense(r: pd.DataFrame, cat: Catalog) -> pd.DataFrame:
    d = pd.DataFrame({"u": r.user_id.map(cat.user_index).values, "i": r.movie_id.map(cat.item_index).values,
                      "rating": r.rating.values, "t": r.timestamp.values})
    return d.sort_values(["t", "u", "i"], kind="mergesort").reset_index(drop=True)


def make_snapshot(d: pd.DataFrame, cutoff: int, end: int | None) -> Snapshot:
    hist = d[d.t < cutoff]
    win = d[(d.t >= cutoff) & ((d.t < end) if end is not None else True)]
    seen = hist.groupby("u").i.agg(set).to_dict()
    pos = win[win.rating >= POSITIVE_RATING]
    # an item the user already rated before the cutoff is never a valid recommendation
    labels = {}
    for u, items in pos.groupby("u").i:
        s = set(items) - seen.get(u, set())
        if s:
            labels[u] = s
    known = set(hist.i.unique())
    cold = sum(1 for s in labels.values() for i in s if i not in known)
    return Snapshot(cutoff=int(cutoff), history=hist.reset_index(drop=True), labels=labels, seen=seen,
                    cold_label_items=cold)


def load_splits():
    r, m, u = load_raw()
    cat = build_catalog(m, u)
    d = to_dense(r, cat)
    q80, q90 = np.quantile(d.t.values, [0.8, 0.9]).astype(int)
    valid = make_snapshot(d, q80, q90)     # ranker training: features before q80, labels in [q80, q90)
    test = make_snapshot(d, q90, None)     # final evaluation: features before q90, labels after
    return cat, valid, test
