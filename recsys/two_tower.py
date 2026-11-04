"""Stage 1: two-tower retrieval.

User tower: user id, gender, age, occupation, and the mean of the user's most recent items.
Item tower: item id, genres, release-year bucket.
Training: each past interaction is a positive; the user's history is only the items *before* it,
so training sees the same kind of input the model gets at serving time. Loss is in-batch sampled
softmax with log-frequency (logQ) correction, so popular items aren't over-penalized just for
appearing as negatives more often.
Serving: item vectors go into a FAISS inner-product index; each user retrieves their top candidates.
"""
from __future__ import annotations

import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import HISTORY_LEN, Catalog, Snapshot

DIM = 64


class TwoTower(nn.Module):
    def __init__(self, cat: Catalog, dim: int = DIM, tau: float = 0.05):
        super().__init__()
        n_i, n_u, n_g = cat.n_items, cat.n_users, cat.genres.shape[1]
        self.pad = n_i
        self.tau = tau
        # user tower
        self.u_id = nn.Embedding(n_u, 32)
        self.u_gender = nn.Embedding(2, 4)
        self.u_age = nn.Embedding(int(cat.age.max()) + 1, 4)
        self.u_occ = nn.Embedding(int(cat.occupation.max()) + 1, 8)
        self.hist = nn.Embedding(n_i + 1, 64, padding_idx=n_i)
        self.u_mlp = nn.Sequential(nn.Linear(32 + 4 + 4 + 8 + 64 + 2, 128), nn.ReLU(), nn.Linear(128, dim))
        # item tower
        self.i_id = nn.Embedding(n_i, 32)
        self.i_genre = nn.Linear(n_g, 16)
        self.i_year = nn.Embedding(int(cat.year_bucket.max()) + 1, 8)
        self.i_mlp = nn.Sequential(nn.Linear(32 + 16 + 8, 128), nn.ReLU(), nn.Linear(128, dim))
        self.register_buffer("genres", torch.tensor(cat.genres))
        self.register_buffer("year", torch.tensor(cat.year_bucket))
        self.register_buffer("gender", torch.tensor(cat.gender))
        self.register_buffer("age", torch.tensor(cat.age))
        self.register_buffer("occ", torch.tensor(cat.occupation))

    def user_vec(self, users, hist, example_age=None):
        mask = (hist != self.pad).float().unsqueeze(-1)
        pooled = (self.hist(hist) * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        if example_age is None:                       # serving: the request happens "now"
            example_age = torch.zeros(len(users))
        a = example_age.unsqueeze(1)
        x = torch.cat([self.u_id(users), self.u_gender(self.gender[users]), self.u_age(self.age[users]),
                       self.u_occ(self.occ[users]), pooled, a, a * a], dim=1)
        return F.normalize(self.u_mlp(x), dim=1)

    def item_vec(self, items):
        x = torch.cat([self.i_id(items), self.i_genre(self.genres[items]), self.i_year(self.year[items])], dim=1)
        return F.normalize(self.i_mlp(x), dim=1)


def _sequences(snap: Snapshot, n_users: int, pad: int):
    """Per-user time-ordered item sequences, left-padded with HISTORY_LEN pad tokens, concatenated."""
    h = snap.history  # already sorted by time
    seqs = h.groupby("u").i.apply(np.array)
    offsets = np.zeros(n_users, np.int64)
    parts, pos = [], 0
    for u in range(n_users):
        s = seqs.get(u, np.array([], np.int64))
        offsets[u] = pos
        parts.append(np.concatenate([np.full(HISTORY_LEN, pad), s]))
        pos += HISTORY_LEN + len(s)
    return np.concatenate(parts).astype(np.int64), offsets, seqs


def history_window(flat, offsets, users, positions):
    """Items at sequence positions [p - HISTORY_LEN, p) for each (user, p); pads where history is short."""
    base = offsets[users] + positions                       # index of the first of the 50 slots
    return flat[base[:, None] + np.arange(HISTORY_LEN)[None, :]]


def train(cat: Catalog, snap: Snapshot, seed: int = 0, epochs: int = 5, batch: int = 1024, lr: float = 2e-3,
          tau: float = 0.1, loss_type: str = "full", use_age: bool = True, verbose: bool = False) -> "Retriever":
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = TwoTower(cat, tau=tau)
    flat, offsets, seqs = _sequences(snap, cat.n_users, model.pad)
    users = np.concatenate([np.full(len(s), u) for u, s in seqs.items()])
    positions = np.concatenate([np.arange(len(s)) for s in seqs.values])
    targets = np.concatenate(list(seqs.values))
    # "example age" (Covington et al., YouTube DNN, 2016): how long before the cutoff each training
    # interaction happened. The model can learn that what people watch drifts over time; at serving the
    # age is 0, which points the user vector at what is current. Measured in 30-day units, log-scaled.
    times = np.concatenate(list(snap.history.groupby("u").t.apply(np.array).values))
    ages = np.log1p((snap.cutoff - times) / (30 * 86400)).astype(np.float32)
    if not use_age:
        ages[:] = 0.0
    freq = np.bincount(targets, minlength=cat.n_items).astype(np.float64)
    logq = torch.tensor(np.log((freq + 1) / (freq + 1).sum()), dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    all_items = torch.arange(cat.n_items)
    n = len(targets)
    for ep in range(epochs):
        perm = rng.permutation(n)
        total = 0.0
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            u = torch.tensor(users[idx])
            hist = torch.tensor(history_window(flat, offsets, users[idx], positions[idx]))
            it = torch.tensor(targets[idx])
            uv = model.user_vec(u, hist, torch.tensor(ages[idx]))
            if loss_type == "full":
                # the catalog is small (3.7k items), so score every item: an exact softmax, no sampling bias
                logits = uv @ model.item_vec(all_items).T / model.tau
                loss = F.cross_entropy(logits, it)
            else:
                iv = model.item_vec(it)
                logits = uv @ iv.T / model.tau - logq[it][None, :]
                same = it[None, :] == it[:, None]
                logits = logits.masked_fill(same & ~torch.eye(len(idx), dtype=torch.bool), float("-inf"))
                loss = F.cross_entropy(logits, torch.arange(len(idx)))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if verbose:
            print(f"  two-tower epoch {ep + 1}: loss {total / n:.3f}", flush=True)
    model.eval()
    return Retriever(model, cat, snap)


class Retriever:
    """A trained two-tower model frozen at a snapshot: item index built, user histories known."""

    def __init__(self, model: TwoTower, cat: Catalog, snap: Snapshot):
        self.model, self.cat, self.snap = model, cat, snap
        with torch.no_grad():
            self.item_vecs = model.item_vec(torch.arange(cat.n_items)).numpy().astype(np.float32)
        self.index = faiss.IndexFlatIP(self.item_vecs.shape[1])
        self.index.add(self.item_vecs)
        last = snap.history.groupby("u").i.apply(lambda s: s.values[-HISTORY_LEN:])
        self.recent = {u: v for u, v in last.items()}

    def user_vecs(self, users) -> np.ndarray:
        hist = np.full((len(users), HISTORY_LEN), self.model.pad, np.int64)
        for r, u in enumerate(users):
            h = self.recent.get(u, np.array([], np.int64))
            if len(h):
                hist[r, -len(h):] = h
        with torch.no_grad():
            return self.model.user_vec(torch.tensor(np.asarray(users)), torch.tensor(hist)).numpy()

    def score(self, users) -> np.ndarray:
        """Dense scores over the whole catalog (used for evaluation and ranker features)."""
        return self.user_vecs(users) @ self.item_vecs.T

    def retrieve(self, user: int, k: int):
        """FAISS top-k for one user, excluding items they've already rated. Returns (items, scores)."""
        seen = self.snap.seen.get(user, set())
        D, I = self.index.search(self.user_vecs([user]), min(k + len(seen), self.cat.n_items))
        keep = [(i, d) for i, d in zip(I[0], D[0]) if i not in seen][:k]
        return np.array([i for i, _ in keep]), np.array([d for _, d in keep], np.float32)

    # FAISS indexes don't pickle; rebuild from the stored item vectors on load
    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop("index", None)
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self.index = faiss.IndexFlatIP(self.item_vecs.shape[1])
        self.index.add(self.item_vecs)
