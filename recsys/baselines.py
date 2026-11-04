"""Baselines a two-stage system has to beat: popularity, item-kNN, and BPR matrix factorization.

All are fit only on a Snapshot's history (events before its cutoff).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from .data import Snapshot

DAY = 86400


def interaction_matrix(snap: Snapshot, n_users: int, n_items: int) -> sp.csr_matrix:
    h = snap.history.drop_duplicates(["u", "i"])
    return sp.csr_matrix((np.ones(len(h), np.float32), (h.u.values, h.i.values)), shape=(n_users, n_items))


def mask_seen(scores: np.ndarray, users, snap: Snapshot) -> np.ndarray:
    scores = scores.copy()
    for row, u in enumerate(users):
        s = snap.seen.get(u)
        if s:
            scores[row, list(s)] = -np.inf
    return scores


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    part = np.argpartition(-scores, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1)
    return np.take_along_axis(part, order, axis=1)


class Popularity:
    """Most-watched items in the last `window_days` before the cutoff (recency matters on ML-1M)."""

    def __init__(self, window_days: int = 30):
        self.window_days = window_days

    def fit(self, snap: Snapshot, n_users: int, n_items: int):
        h = snap.history
        recent = h[h.t >= snap.cutoff - self.window_days * DAY]
        self.scores = np.bincount(recent.i.values, minlength=n_items).astype(np.float32)
        self.scores += 1e-6 * np.bincount(h.i.values, minlength=n_items)   # tie-break by all-time count
        return self

    def score(self, users) -> np.ndarray:
        return np.tile(self.scores, (len(users), 1))


class ItemKNN:
    """Item-item cosine similarity on the binary interaction matrix, keeping each item's top neighbors."""

    def __init__(self, neighbors: int = 200):
        self.neighbors = neighbors

    def fit(self, snap: Snapshot, n_users: int, n_items: int):
        X = interaction_matrix(snap, n_users, n_items)
        norms = np.sqrt(np.asarray(X.sum(axis=0)).ravel()) + 1e-9
        S = (X.T @ X).toarray() / norms[:, None] / norms[None, :]
        np.fill_diagonal(S, 0.0)
        if self.neighbors < n_items:
            drop = np.argpartition(-S, self.neighbors, axis=1)[:, self.neighbors:]
            np.put_along_axis(S, drop, 0.0, axis=1)
        self.S = sp.csr_matrix(S.astype(np.float32))      # kept sparse: top neighbors only, so scoring is cheap
        self.X = X
        return self

    def score(self, users) -> np.ndarray:
        return (self.X[users] @ self.S).toarray()


class BPRMF:
    """Matrix factorization trained with the BPR pairwise loss and uniform negative sampling."""

    def __init__(self, dim: int = 64, epochs: int = 30, lr: float = 0.01, reg: float = 1e-5, seed: int = 0):
        self.dim, self.epochs, self.lr, self.reg, self.seed = dim, epochs, lr, reg, seed

    def fit(self, snap: Snapshot, n_users: int, n_items: int):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        h = snap.history.drop_duplicates(["u", "i"])
        u_all = torch.tensor(h.u.values)
        i_all = torch.tensor(h.i.values)
        pos_keys = np.sort(h.u.values.astype(np.int64) * n_items + h.i.values)
        P = torch.nn.Embedding(n_users, self.dim)
        Q = torch.nn.Embedding(n_items, self.dim)
        b = torch.nn.Embedding(n_items, 1)
        for e in (P, Q):
            torch.nn.init.normal_(e.weight, std=0.05)
        torch.nn.init.zeros_(b.weight)
        opt = torch.optim.Adam([P.weight, Q.weight, b.weight], lr=self.lr)
        n, bs = len(h), 4096
        for _ in range(self.epochs):
            perm = torch.tensor(rng.permutation(n))
            for s in range(0, n, bs):
                idx = perm[s:s + bs]
                u, i = u_all[idx], i_all[idx]
                j = torch.tensor(rng.integers(0, n_items, len(idx)))
                # resample negatives that happen to be positives (one pass is enough at this density)
                keys = u.numpy().astype(np.int64) * n_items + j.numpy()
                pos = np.minimum(np.searchsorted(pos_keys, keys), len(pos_keys) - 1)
                clash = pos_keys[pos] == keys
                if clash.any():
                    j[torch.tensor(clash)] = torch.tensor(rng.integers(0, n_items, int(clash.sum())))
                pu, qi, qj = P(u), Q(i), Q(j)
                x = (pu * (qi - qj)).sum(1) + (b(i) - b(j)).squeeze(1)
                loss = -torch.nn.functional.logsigmoid(x).mean() + self.reg * (
                    pu.pow(2).sum() + qi.pow(2).sum() + qj.pow(2).sum()) / len(idx)
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.P = P.weight.detach().numpy()
        self.Q = Q.weight.detach().numpy()
        self.b = b.weight.detach().numpy().ravel()
        return self

    def score(self, users) -> np.ndarray:
        return self.P[users] @ self.Q.T + self.b[None, :]
