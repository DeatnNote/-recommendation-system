"""
src/two_tower.py
================
Two-Tower Neural Retrieval Model.

Architecture (mirrors production systems at YouTube, Pinterest, Netflix):
  User Tower:   user_embedding → BatchNorm → Dense(256) → ReLU → Dense(128) → L2-norm
  Item Tower:   item_embedding → BatchNorm → Dense(256) → ReLU → Dense(128) → L2-norm
  Score:        cosine_similarity(user_vec, item_vec)

Training:
  Loss: BPR (Bayesian Personalised Ranking) — maximises margin between
        positive and negative items in embedding space.
  BPR: -log(sigmoid(score(u, pos) - score(u, neg)))

Retrieval:
  Item vectors are pre-computed and indexed in FAISS.
  At inference: compute user vector → FAISS nearest-neighbour search → top-k items.
  Target: <50ms retrieval latency.

References:
  - Yi et al. (2019) — Sampling-Bias-Corrected Neural Modeling (YouTube)
  - Covington et al. (2016) — Deep Neural Networks for YouTube Recommendations
  - He et al. (2017) — Neural Collaborative Filtering
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import faiss
import os
from typing import List, Dict, Optional
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════

class UserTower(nn.Module):
    def __init__(self, n_users: int, embedding_dim: int, output_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(n_users, embedding_dim)
        self.net = nn.Sequential(
            nn.BatchNorm1d(embedding_dim),
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim),
        )
        nn.init.normal_(self.embedding.weight, std=0.01)

    def forward(self, user_idx: torch.Tensor) -> torch.Tensor:
        x = self.embedding(user_idx)
        x = self.net(x)
        return F.normalize(x, dim=1)          # L2 norm → unit sphere


class ItemTower(nn.Module):
    def __init__(self, n_movies: int, embedding_dim: int, output_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(n_movies, embedding_dim)
        self.net = nn.Sequential(
            nn.BatchNorm1d(embedding_dim),
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim),
        )
        nn.init.normal_(self.embedding.weight, std=0.01)

    def forward(self, item_idx: torch.Tensor) -> torch.Tensor:
        x = self.embedding(item_idx)
        x = self.net(x)
        return F.normalize(x, dim=1)


class TwoTowerModel(nn.Module):
    def __init__(
        self,
        n_users:       int,
        n_movies:      int,
        embedding_dim: int = 128,
        output_dim:    int = 64,
    ):
        super().__init__()
        self.user_tower = UserTower(n_users,  embedding_dim, output_dim)
        self.item_tower = ItemTower(n_movies, embedding_dim, output_dim)
        self.output_dim = output_dim

    def forward(
        self,
        user_idx:     torch.Tensor,
        pos_item_idx: torch.Tensor,
        neg_item_idx: torch.Tensor,
    ):
        user_vec = self.user_tower(user_idx)
        pos_vec  = self.item_tower(pos_item_idx)
        neg_vec  = self.item_tower(neg_item_idx)
        pos_score = (user_vec * pos_vec).sum(dim=1)
        neg_score = (user_vec * neg_vec).sum(dim=1)
        return pos_score, neg_score

    def get_user_vector(self, user_idx: int) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            t = torch.tensor([user_idx], dtype=torch.long)
            return self.user_tower(t).numpy()[0]

    def get_all_item_vectors(self, batch_size: int = 512) -> np.ndarray:
        self.eval()
        n = self.item_tower.embedding.num_embeddings
        vecs = []
        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                t   = torch.arange(start, end, dtype=torch.long)
                vecs.append(self.item_tower(t).numpy())
        return np.vstack(vecs).astype(np.float32)


# ══════════════════════════════════════════════════════════════
# BPR Loss
# ══════════════════════════════════════════════════════════════

def bpr_loss(pos_score: torch.Tensor, neg_score: torch.Tensor) -> torch.Tensor:
    """
    Bayesian Personalised Ranking loss.
    Maximises the margin between positive and negative item scores.
    -log(sigmoid(pos - neg)) → 0 when pos >> neg.
    """
    return -F.logsigmoid(pos_score - neg_score).mean()


# ══════════════════════════════════════════════════════════════
# Two-Tower Wrapper (fit + FAISS index + recommend)
# ══════════════════════════════════════════════════════════════

class TwoTower:
    def __init__(
        self,
        n_users:       int,
        n_movies:      int,
        embedding_dim: int   = 128,
        output_dim:    int   = 64,
        lr:            float = 0.001,
        weight_decay:  float = 1e-5,
        batch_size:    int   = 2048,
        epochs:        int   = 5,
        device:        str   = "cpu",
    ):
        self.n_movies   = n_movies
        self.epochs     = epochs
        self.batch_size = batch_size
        self.device     = torch.device(device)

        self.model = TwoTowerModel(n_users, n_movies, embedding_dim, output_dim)
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=2, gamma=0.5
        )

        self.train_losses: List[float] = []
        self.index: Optional[faiss.IndexFlatIP] = None   # FAISS inner-product index

    def fit(self, train_loader: DataLoader, **kwargs) -> "TwoTower":
        print(f"  Training TwoTower ({self.epochs} epochs, BPR loss)…")
        self.model.train()

        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            n_batches  = 0
            for users, pos_items, neg_items in tqdm(
                train_loader, desc=f"  Epoch {epoch}/{self.epochs}", leave=False
            ):
                users     = users.to(self.device)
                pos_items = pos_items.to(self.device)
                neg_items = neg_items.to(self.device)

                self.optimizer.zero_grad()
                pos_score, neg_score = self.model(users, pos_items, neg_items)
                loss = bpr_loss(pos_score, neg_score)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches  += 1

            self.scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)
            self.train_losses.append(avg_loss)
            print(f"    Epoch {epoch}/{self.epochs}  BPR loss={avg_loss:.4f}  "
                  f"lr={self.scheduler.get_last_lr()[0]:.5f}")

        self._build_faiss_index()
        return self

    def _build_faiss_index(self):
        """Pre-compute all item vectors and build a FAISS inner-product index."""
        print("  Building FAISS index…")
        item_vecs = self.model.get_all_item_vectors()   # shape: (n_movies, output_dim)
        dim = item_vecs.shape[1]

        # IndexFlatIP = exact inner product search on L2-normalised vectors = cosine similarity
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(item_vecs)
        print(f"  FAISS index built: {self.index.ntotal:,} items, dim={dim}")

    def recommend(
        self,
        user_idx: int,
        n: int = 10,
        exclude: set = None,
    ) -> List[int]:
        """
        Retrieve top-n items for a user via FAISS nearest-neighbour search.
        Excludes already-seen items (exclude set).
        """
        assert self.index is not None, "Call fit() before recommend()"
        exclude = exclude or set()

        user_vec = self.model.get_user_vector(user_idx).reshape(1, -1).astype(np.float32)

        # Over-fetch to account for exclusions
        k = min(n + len(exclude) + 50, self.n_movies)
        _, item_indices = self.index.search(user_vec, k)
        candidates = item_indices[0].tolist()

        return [i for i in candidates if i not in exclude][:n]

    def recommend_batch(
        self,
        user_indices: List[int],
        train: pd.DataFrame,
        n: int = 10,
    ) -> Dict[int, List[int]]:
        user_seen = train.groupby("user_idx")["movie_idx"].apply(set).to_dict()
        return {
            u: self.recommend(u, n=n, exclude=user_seen.get(u, set()))
            for u in tqdm(user_indices, desc="  TwoTower recommendations", leave=False)
        }

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "train_losses": self.train_losses,
        }, path)
        if self.index is not None:
            faiss.write_index(self.index, path.replace(".pt", ".faiss"))
        print(f"  Model saved → {path}")

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.train_losses = checkpoint.get("train_losses", [])
        faiss_path = path.replace(".pt", ".faiss")
        if os.path.exists(faiss_path):
            self.index = faiss.read_index(faiss_path)
        return self
