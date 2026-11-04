"""FastAPI service for the two-stage recommender.

Per request: two-tower user vector -> FAISS top-100, plus recent-popularity top-100 and BPR top-100
-> de-duplicated candidates -> ranking features -> LightGBM -> top-k.

Run:   uvicorn recsys.serve:app --port 8000
Then:  curl localhost:8000/recommend/1?k=10
Requires artifacts/service.pkl, written by `python -m recsys.pipeline`.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException

from .data import load_raw
from .ranker import FEATURES, PER_SOURCE, ItemStats, UserStats, build_features, order_by_score

ARTIFACT = Path("artifacts") / "service.pkl"


class Recommender:
    def __init__(self, path: Path = ARTIFACT):
        with open(path, "rb") as f:
            b = pickle.load(f)
        self.cat, self.snap, self.s1, self.ranker = b["cat"], b["snap"], b["stage1"], b["ranker"]
        self.stats = (ItemStats(self.cat, self.snap), UserStats(self.cat, self.snap))
        self.pop_order = np.argsort(-self.s1.pop.scores)
        movies = load_raw()[1]
        self.titles = dict(zip(movies.movie_id, movies.title))

    def candidates(self, u: int) -> dict:
        seen = self.snap.seen.get(u, set())
        tt_items, _ = self.s1.tt.retrieve(u, PER_SOURCE)                 # FAISS
        pop = [i for i in self.pop_order[:PER_SOURCE + len(seen)] if i not in seen][:PER_SOURCE]
        bpr_scores = self.s1.bpr.score([u])[0]
        if seen:
            bpr_scores[list(seen)] = -np.inf
        bpr = np.argpartition(-bpr_scores, PER_SOURCE)[:PER_SOURCE]
        return {"tt": set(tt_items.tolist()), "pop": set(pop), "bpr": set(bpr.tolist())}

    def recommend(self, movielens_user_id: int, k: int = 10):
        if movielens_user_id not in self.cat.user_index:
            raise KeyError(movielens_user_id)
        u = self.cat.user_index[movielens_user_id]
        df = build_features(self.cat, self.s1, self.snap, [u], {u: self.candidates(u)}, stats=self.stats,
                            with_labels=False)
        scores = self.ranker.predict(df[FEATURES])
        score_of = dict(zip(df.item.values, scores))
        top = order_by_score(df.item.values, scores)[:k]
        return [{"movie_id": int(self.cat.item_ids[i]), "title": self.titles.get(int(self.cat.item_ids[i]), ""),
                 "score": float(score_of[i])} for i in top]


app = FastAPI(title="Two-stage movie recommender")
_rec: Recommender | None = None


def get() -> Recommender:
    global _rec
    if _rec is None:
        _rec = Recommender()
    return _rec


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    get()
    return {"status": "ready"}


@app.get("/recommend/{user_id}")
def recommend(user_id: int, k: int = 10):
    if not 1 <= k <= 100:
        raise HTTPException(400, "k must be between 1 and 100")
    try:
        return {"user_id": user_id, "items": get().recommend(user_id, k)}
    except KeyError:
        raise HTTPException(404, f"unknown user {user_id}")
