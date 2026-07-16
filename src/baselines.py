"""
src/baselines.py
================
Baseline recommender models:
  1. PopularityBaseline  — recommend the most-rated items globally
  2. MatrixFactorization — classic MF via gradient descent (pointwise MSE)

Both implement the same .fit() / .recommend() interface as the two-tower model,
so the evaluation module can treat all models identically.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Dict
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════
# 1. Popularity Baseline
# ══════════════════════════════════════════════════════════════

class PopularityBaseline:
    """
    Recommends the globally most-rated items.
    No personalisation — same list for every user.
    Useful as a lower-bound baseline and for coverage/novelty comparison.
    """

    def __init__(self):
        self.popular_items: List[int] = []
        self.item_counts:   Dict[int, int] = {}

    def fit(self, train: pd.DataFrame, **kwargs) -> "PopularityBaseline":
        counts = train.groupby("movie_idx").size().sort_values(ascending=False)
        self.item_counts   = counts.to_dict()
        self.popular_items = counts.index.tolist()
        print(f"  PopularityBaseline fitted. Top item: movie_idx={self.popular_items[0]}"
              f" ({self.item_counts[self.popular_items[0]]:,} ratings)")
        return self

    def recommend(
        self,
        user_idx: int,
        n: int = 10,
        exclude: set = None,
    ) -> List[int]:
        exclude = exclude or set()
        recs = [i for i in self.popular_items if i not in exclude]
        return recs[:n]

    def recommend_batch(
        self,
        user_indices: List[int],
        train: pd.DataFrame,
        n: int = 10,
    ) -> Dict[int, List[int]]:
        user_seen = train.groupby("user_idx")["movie_idx"].apply(set).to_dict()
        return {
            u: self.recommend(u, n=n, exclude=user_seen.get(u, set()))
            for u in user_indices
        }


# ══════════════════════════════════════════════════════════════
# 2. Matrix Factorization (MF)
# ══════════════════════════════════════════════════════════════

class MatrixFactorizationModel(nn.Module):
    """
    Classic Matrix Factorization:
      score(u, i) = dot(U[u], V[i]) + b_u[u] + b_i[i] + global_bias
    Trained with MSE loss on normalised explicit ratings.
    """

    def __init__(self, n_users: int, n_movies: int, embedding_dim: int = 64):
        super().__init__()
        self.user_emb    = nn.Embedding(n_users,  embedding_dim)
        self.item_emb    = nn.Embedding(n_movies, embedding_dim)
        self.user_bias   = nn.Embedding(n_users,  1)
        self.item_bias   = nn.Embedding(n_movies, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        nn.init.normal_(self.user_emb.weight,  std=0.01)
        nn.init.normal_(self.item_emb.weight,  std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        u = self.user_emb(user_idx)
        v = self.item_emb(item_idx)
        dot = (u * v).sum(dim=1)
        b_u = self.user_bias(user_idx).squeeze()
        b_i = self.item_bias(item_idx).squeeze()
        return torch.sigmoid(dot + b_u + b_i + self.global_bias)

    def get_user_embedding(self, user_idx: int) -> np.ndarray:
        with torch.no_grad():
            return self.user_emb(torch.tensor([user_idx])).numpy()[0]

    def get_item_embeddings(self) -> np.ndarray:
        with torch.no_grad():
            return self.item_emb.weight.numpy()


class MatrixFactorization:
    """
    Wrapper around MatrixFactorizationModel — handles training loop,
    checkpointing, and the recommend() interface.
    """

    def __init__(
        self,
        n_users:       int,
        n_movies:      int,
        embedding_dim: int   = 64,
        lr:            float = 0.001,
        weight_decay:  float = 1e-5,
        batch_size:    int   = 2048,
        epochs:        int   = 5,
        device:        str   = "cpu",
    ):
        self.n_movies  = n_movies
        self.epochs    = epochs
        self.batch_size = batch_size
        self.device    = torch.device(device)

        self.model = MatrixFactorizationModel(n_users, n_movies, embedding_dim)
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.loss_fn = nn.MSELoss()
        self.train_losses: List[float] = []

    def fit(self, train_loader: DataLoader, **kwargs) -> "MatrixFactorization":
        self.model.train()
        print(f"  Training MatrixFactorization ({self.epochs} epochs)…")

        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            n_batches  = 0
            for users, items, ratings in tqdm(train_loader, desc=f"  Epoch {epoch}/{self.epochs}", leave=False):
                users, items, ratings = (
                    users.to(self.device),
                    items.to(self.device),
                    ratings.to(self.device),
                )
                self.optimizer.zero_grad()
                preds = self.model(users, items)
                loss  = self.loss_fn(preds, ratings)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                n_batches  += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            self.train_losses.append(avg_loss)
            print(f"    Epoch {epoch}/{self.epochs}  loss={avg_loss:.4f}")

        return self

    def recommend(
        self,
        user_idx: int,
        n: int = 10,
        exclude: set = None,
    ) -> List[int]:
        exclude = exclude or set()
        self.model.eval()
        with torch.no_grad():
            u_tensor = torch.tensor([user_idx] * self.n_movies, dtype=torch.long)
            i_tensor = torch.arange(self.n_movies, dtype=torch.long)
            scores   = self.model(u_tensor, i_tensor).numpy()
        scores_idx = np.argsort(-scores)
        return [i for i in scores_idx if i not in exclude][:n]

    def recommend_batch(
        self,
        user_indices: List[int],
        train: pd.DataFrame,
        n: int = 10,
    ) -> Dict[int, List[int]]:
        user_seen = train.groupby("user_idx")["movie_idx"].apply(set).to_dict()
        return {
            u: self.recommend(u, n=n, exclude=user_seen.get(u, set()))
            for u in tqdm(user_indices, desc="  MF recommendations", leave=False)
        }
