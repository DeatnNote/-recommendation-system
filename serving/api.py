"""
serving/api.py
==============
FastAPI REST API for the recommendation system.
Loads the trained Two-Tower model + FAISS index at startup
and serves recommendations with <50ms target latency.

Run:
  uvicorn serving.api:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
  GET /recommend/{user_id}?n=10   → top-n movie recommendations
  GET /similar/{movie_id}?n=10    → similar movies via item-item retrieval
  GET /health                     → health check + model info
  GET /stats                      → dataset + model statistics
"""

import time
import json
import os
import sys
import numpy as np
import pandas as pd
import torch
import faiss

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from src.dataset  import load_data, remap_ids
from src.two_tower import TwoTowerModel

# ── App ───────────────────────────────────────────────────────
app = FastAPI(
    title="Movie Recommendation API",
    description="Two-Tower neural retrieval system with FAISS indexing",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ──────────────────────────────────────────────
MODEL:       Optional[TwoTowerModel] = None
FAISS_INDEX: Optional[faiss.IndexFlatIP] = None
MOVIES:      Optional[pd.DataFrame] = None
RATINGS:     Optional[pd.DataFrame] = None
USER2IDX:    dict = {}
MOVIE2IDX:   dict = {}
IDX2MOVIE:   dict = {}
USER_SEEN:   dict = {}


# ── Startup ───────────────────────────────────────────────────
@app.on_event("startup")
async def load_model():
    global MODEL, FAISS_INDEX, MOVIES, RATINGS, USER2IDX, MOVIE2IDX, IDX2MOVIE, USER_SEEN

    print("Loading data…")
    ratings, movies, _ = load_data()
    ratings, USER2IDX, MOVIE2IDX = remap_ids(ratings)
    IDX2MOVIE = {v: k for k, v in MOVIE2IDX.items()}

    MOVIES  = movies
    RATINGS = ratings

    # Build user seen-item sets for exclusion
    USER_SEEN = ratings.groupby("user_idx")["movie_idx"].apply(set).to_dict()

    N_USERS  = len(USER2IDX)
    N_MOVIES = len(MOVIE2IDX)

    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "two_tower.pt")
    faiss_path = model_path.replace(".pt", ".faiss")

    if not os.path.exists(model_path):
        raise RuntimeError(
            f"Model not found at {model_path}. Run pipeline.py first."
        )

    print("Loading Two-Tower model…")
    MODEL = TwoTowerModel(N_USERS, N_MOVIES, embedding_dim=128, output_dim=64)
    checkpoint = torch.load(model_path, map_location="cpu")
    MODEL.load_state_dict(checkpoint["model_state"])
    MODEL.eval()

    print("Loading FAISS index…")
    FAISS_INDEX = faiss.read_index(faiss_path)
    print(f"  FAISS: {FAISS_INDEX.ntotal:,} items indexed")

    print("API ready ✓")


# ── Response schemas ──────────────────────────────────────────
class MovieRec(BaseModel):
    movie_id:    int
    movie_idx:   int
    title:       str
    genres:      str
    score:       float

class RecommendResponse(BaseModel):
    user_id:       int
    user_idx:      int
    recommendations: List[MovieRec]
    latency_ms:    float
    model:         str

class SimilarResponse(BaseModel):
    movie_id:    int
    similar:     List[MovieRec]
    latency_ms:  float


# ── Helper ────────────────────────────────────────────────────
def idx_to_movie(movie_idx: int, score: float = 0.0) -> MovieRec:
    original_id = IDX2MOVIE.get(movie_idx, -1)
    row = MOVIES[MOVIES["movieId"] == original_id]
    if row.empty:
        return MovieRec(movie_id=original_id, movie_idx=movie_idx,
                        title="Unknown", genres="", score=round(score, 4))
    row = row.iloc[0]
    return MovieRec(
        movie_id=int(original_id),
        movie_idx=movie_idx,
        title=row["title"],
        genres=row["genres"],
        score=round(float(score), 4),
    )


# ── Endpoints ─────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model":  "TwoTower (BPR + FAISS)",
        "n_items_indexed": FAISS_INDEX.ntotal if FAISS_INDEX else 0,
        "n_users": len(USER2IDX),
    }


@app.get("/stats")
def stats():
    if RATINGS is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "n_users":   len(USER2IDX),
        "n_movies":  len(MOVIE2IDX),
        "n_ratings": len(RATINGS),
        "avg_rating": round(float(RATINGS["rating"].mean()), 3),
        "matrix_density_pct": round(len(RATINGS) / (len(USER2IDX) * len(MOVIE2IDX)) * 100, 3),
    }


@app.get("/recommend/{user_id}", response_model=RecommendResponse)
def recommend(
    user_id: int,
    n: int = Query(default=10, ge=1, le=50),
    exclude_seen: bool = Query(default=True),
):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    user_idx = USER2IDX.get(user_id)
    if user_idx is None:
        raise HTTPException(status_code=404, detail=f"user_id {user_id} not found")

    t0 = time.perf_counter()

    # Get user vector
    with torch.no_grad():
        u_vec = MODEL.user_tower(
            torch.tensor([user_idx], dtype=torch.long)
        ).numpy().astype(np.float32)

    # FAISS search — over-fetch to account for exclusions
    exclude = USER_SEEN.get(user_idx, set()) if exclude_seen else set()
    k_fetch = min(n + len(exclude) + 50, FAISS_INDEX.ntotal)
    scores, indices = FAISS_INDEX.search(u_vec, k_fetch)

    recs = []
    for score, idx in zip(scores[0], indices[0]):
        if exclude_seen and idx in exclude:
            continue
        recs.append(idx_to_movie(int(idx), float(score)))
        if len(recs) >= n:
            break

    latency_ms = (time.perf_counter() - t0) * 1000

    return RecommendResponse(
        user_id=user_id,
        user_idx=user_idx,
        recommendations=recs,
        latency_ms=round(latency_ms, 2),
        model="TwoTower",
    )


@app.get("/similar/{movie_id}", response_model=SimilarResponse)
def similar_items(
    movie_id: int,
    n: int = Query(default=10, ge=1, le=50),
):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    movie_idx = MOVIE2IDX.get(movie_id)
    if movie_idx is None:
        raise HTTPException(status_code=404, detail=f"movie_id {movie_id} not found")

    t0 = time.perf_counter()

    with torch.no_grad():
        item_vec = MODEL.item_tower(
            torch.tensor([movie_idx], dtype=torch.long)
        ).numpy().astype(np.float32)

    scores, indices = FAISS_INDEX.search(item_vec, n + 1)

    similar = []
    for score, idx in zip(scores[0], indices[0]):
        if int(idx) == movie_idx:     # exclude the query item itself
            continue
        similar.append(idx_to_movie(int(idx), float(score)))
        if len(similar) >= n:
            break

    latency_ms = (time.perf_counter() - t0) * 1000

    return SimilarResponse(
        movie_id=movie_id,
        similar=similar,
        latency_ms=round(latency_ms, 2),
    )
