"""
src/dataset.py
==============
Data loading, preprocessing, and train/val/test splitting
for the MovieLens-style recommendation dataset.

Split strategy: leave-one-out (last interaction per user → test,
second-to-last → val, remainder → train). Standard in RecSys literature.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Dict
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# ── Loading ───────────────────────────────────────────────────

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))
    movies  = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))
    users   = pd.read_csv(os.path.join(DATA_DIR, "users.csv"))
    return ratings, movies, users


# ── ID remapping to contiguous 0-based indices ───────────────

def remap_ids(ratings: pd.DataFrame) -> Tuple[pd.DataFrame, Dict, Dict]:
    """
    Remap userId and movieId to contiguous 0-based indices.
    Required for embedding layers (nn.Embedding expects 0..N-1).
    Returns remapped dataframe and two lookup dicts.
    """
    user_ids  = sorted(ratings["userId"].unique())
    movie_ids = sorted(ratings["movieId"].unique())

    user2idx  = {uid: idx for idx, uid in enumerate(user_ids)}
    movie2idx = {mid: idx for idx, mid in enumerate(movie_ids)}

    ratings = ratings.copy()
    ratings["user_idx"]  = ratings["userId"].map(user2idx)
    ratings["movie_idx"] = ratings["movieId"].map(movie2idx)

    return ratings, user2idx, movie2idx


# ── Leave-one-out split ───────────────────────────────────────

def leave_one_out_split(
    ratings: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split strategy used in Neural Collaborative Filtering (He et al. 2017):
      - Test:  each user's most recent interaction
      - Val:   each user's second most recent interaction
      - Train: everything else

    Only users with ≥3 interactions are kept (need at least one
    for train, one for val, one for test).
    """
    ratings = ratings.sort_values(["user_idx", "timestamp"])

    # Filter users with fewer than 3 ratings
    user_counts = ratings.groupby("user_idx").size()
    valid_users = user_counts[user_counts >= 3].index
    ratings     = ratings[ratings["user_idx"].isin(valid_users)].copy()

    # Rank interactions per user (most recent = highest rank)
    ratings["rank"] = ratings.groupby("user_idx")["timestamp"].rank(
        method="first", ascending=True
    ).astype(int)
    ratings["max_rank"] = ratings.groupby("user_idx")["rank"].transform("max")

    test  = ratings[ratings["rank"] == ratings["max_rank"]].copy()
    val   = ratings[ratings["rank"] == ratings["max_rank"] - 1].copy()
    train = ratings[ratings["rank"] <  ratings["max_rank"] - 1].copy()

    print(f"  Train: {len(train):>8,} interactions ({len(train)/len(ratings)*100:.1f}%)")
    print(f"  Val:   {len(val):>8,} interactions")
    print(f"  Test:  {len(test):>8,} interactions")
    print(f"  Users: {ratings['user_idx'].nunique():,}  |  Movies: {ratings['movie_idx'].nunique():,}")

    return train, val, test


# ── Implicit feedback conversion ──────────────────────────────

def to_implicit(df: pd.DataFrame, threshold: float = 3.5) -> pd.DataFrame:
    """
    Convert explicit ratings to implicit feedback.
    Rating >= threshold → positive interaction (label=1).
    Ratings below threshold are dropped (not treated as negatives).
    Standard practice: negative samples are drawn randomly at train time.
    """
    return df[df["rating"] >= threshold].copy()


# ── PyTorch Dataset ───────────────────────────────────────────

class RatingsDataset(Dataset):
    """
    BPR (Bayesian Personalised Ranking) dataset.
    Each sample = (user, positive_item, negative_item).
    Negative items sampled uniformly from unseen items per user.
    """

    def __init__(
        self,
        interactions: pd.DataFrame,
        n_movies: int,
        n_negatives: int = 4,
    ):
        self.n_movies    = n_movies
        self.n_negatives = n_negatives

        self.users  = interactions["user_idx"].values
        self.items  = interactions["movie_idx"].values

        # Build per-user positive set for fast negative sampling
        self.user_positives: Dict[int, set] = (
            interactions.groupby("user_idx")["movie_idx"]
            .apply(set).to_dict()
        )

        # Expand to (user, pos, neg) triples
        self._build_samples()

    def _sample_negative(self, user: int) -> int:
        pos = self.user_positives.get(user, set())
        while True:
            neg = np.random.randint(0, self.n_movies)
            if neg not in pos:
                return neg

    def _build_samples(self):
        users, pos_items, neg_items = [], [], []
        for u, i in zip(self.users, self.items):
            for _ in range(self.n_negatives):
                users.append(u)
                pos_items.append(i)
                neg_items.append(self._sample_negative(u))
        self.sample_users = np.array(users,     dtype=np.int64)
        self.sample_pos   = np.array(pos_items, dtype=np.int64)
        self.sample_neg   = np.array(neg_items, dtype=np.int64)

    def __len__(self):
        return len(self.sample_users)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sample_users[idx], dtype=torch.long),
            torch.tensor(self.sample_pos[idx],   dtype=torch.long),
            torch.tensor(self.sample_neg[idx],   dtype=torch.long),
        )


class PointwiseDataset(Dataset):
    """
    Pointwise dataset for explicit rating prediction.
    Used by the Matrix Factorization baseline.
    """
    def __init__(self, interactions: pd.DataFrame):
        self.users   = torch.tensor(interactions["user_idx"].values,  dtype=torch.long)
        self.items   = torch.tensor(interactions["movie_idx"].values,  dtype=torch.long)
        self.ratings = torch.tensor(interactions["rating"].values / 5.0, dtype=torch.float)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]
