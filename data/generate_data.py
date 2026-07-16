"""
generate_data.py
================
Generates synthetic data mirroring the MovieLens 1M schema.
  - 6,040 users  (same as ML-1M)
  - 3,706 movies (same as ML-1M)
  - ~1,000,209 ratings (same as ML-1M)

Realistic distributions:
  - Movie popularity follows a power law (few blockbusters, many niche films)
  - User activity follows a power law (few heavy users, many casual ones)
  - Ratings skewed positive (mean ~3.58, matching ML-1M)

Outputs:
  data/ratings.csv   — userId, movieId, rating, timestamp
  data/movies.csv    — movieId, title, genres
  data/users.csv     — userId, gender, age, occupation, zip

Run:
  python data/generate_data.py
"""

import numpy as np
import pandas as pd
import os, time

np.random.seed(42)

# ── Config (matching ML-1M exactly) ──────────────────────────
N_USERS   = 6_040
N_MOVIES  = 3_706
N_RATINGS = 1_000_209
OUT_DIR   = os.path.dirname(os.path.abspath(__file__))

# ── Genre pool ────────────────────────────────────────────────
GENRES = ["Action","Adventure","Animation","Children's","Comedy",
          "Crime","Documentary","Drama","Fantasy","Film-Noir",
          "Horror","Musical","Mystery","Romance","Sci-Fi",
          "Thriller","War","Western"]

# ── Occupation codes (ML-1M) ──────────────────────────────────
OCCUPATIONS = list(range(21))  # 0–20

# ── Age brackets (ML-1M) ─────────────────────────────────────
AGE_GROUPS = [1, 18, 25, 35, 45, 50, 56]

def build_movies() -> pd.DataFrame:
    print("  Building movies…")
    genres_per_movie = []
    for _ in range(N_MOVIES):
        n = np.random.choice([1, 2, 3], p=[0.50, 0.38, 0.12])
        chosen = np.random.choice(GENRES, size=n, replace=False)
        genres_per_movie.append("|".join(sorted(chosen)))

    # Decade distribution (1930s–2000s, more recent = more movies)
    decades = np.random.choice(
        range(1930, 2001, 10),
        size=N_MOVIES,
        p=[0.02,0.03,0.05,0.08,0.12,0.18,0.28,0.24]
    )
    years = decades + np.random.randint(0, 10, size=N_MOVIES)

    titles = [f"Movie {i+1} ({years[i]})" for i in range(N_MOVIES)]

    return pd.DataFrame({
        "movieId": range(1, N_MOVIES + 1),
        "title":   titles,
        "genres":  genres_per_movie,
        "year":    years,
    })


def build_users() -> pd.DataFrame:
    print("  Building users…")
    return pd.DataFrame({
        "userId":     range(1, N_USERS + 1),
        "gender":     np.random.choice(["M", "F"], size=N_USERS, p=[0.72, 0.28]),
        "age":        np.random.choice(AGE_GROUPS, size=N_USERS,
                                       p=[0.10,0.24,0.21,0.17,0.13,0.07,0.08]),
        "occupation": np.random.choice(OCCUPATIONS, size=N_USERS),
        "zip":        [f"{np.random.randint(10000,99999):05d}" for _ in range(N_USERS)],
    })


def build_ratings(movies: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    print("  Building ratings (~1M rows)…")

    movie_ids = movies["movieId"].values
    user_ids  = users["userId"].values

    # ── Power-law popularity weights for movies ───────────────
    movie_rank   = np.arange(1, N_MOVIES + 1, dtype=float)
    movie_pop    = 1.0 / (movie_rank ** 0.7)
    movie_pop   /= movie_pop.sum()

    # ── Power-law activity weights for users ──────────────────
    user_rank    = np.arange(1, N_USERS + 1, dtype=float)
    user_act     = 1.0 / (user_rank ** 0.5)
    user_act    /= user_act.sum()

    # ── Sample users and movies ───────────────────────────────
    sampled_users  = np.random.choice(user_ids,  size=N_RATINGS, p=user_act)
    sampled_movies = np.random.choice(movie_ids, size=N_RATINGS, p=movie_pop)

    # ── Rating scores: skewed positive (ML-1M mean = 3.58) ───
    # Mix of Gaussian and mode-at-4
    raw_ratings = np.clip(
        np.random.normal(loc=3.6, scale=1.1, size=N_RATINGS), 1, 5
    ).round().astype(int)
    raw_ratings = np.clip(raw_ratings, 1, 5)

    # ── Timestamps: spread over 1998–2003 ────────────────────
    t_start = int(pd.Timestamp("1998-01-01").timestamp())
    t_end   = int(pd.Timestamp("2003-12-31").timestamp())
    timestamps = np.random.randint(t_start, t_end, size=N_RATINGS)

    df = pd.DataFrame({
        "userId":    sampled_users,
        "movieId":   sampled_movies,
        "rating":    raw_ratings,
        "timestamp": timestamps,
    })

    # ── Remove duplicate (user, movie) pairs — keep highest rating ─
    print("  Deduplicating…")
    df = df.sort_values("rating", ascending=False)
    df = df.drop_duplicates(subset=["userId", "movieId"], keep="first")
    df = df.sort_values(["userId", "timestamp"]).reset_index(drop=True)

    print(f"  Final ratings: {len(df):,}")
    return df


def main():
    print("=" * 50)
    print("  MOVIELENS-STYLE DATA GENERATOR")
    print("=" * 50)
    t0 = time.time()

    movies = build_movies()
    users  = build_users()
    ratings = build_ratings(movies, users)

    # ── Save ─────────────────────────────────────────────────
    movies.to_csv(os.path.join(OUT_DIR, "movies.csv"),  index=False)
    users.to_csv(os.path.join(OUT_DIR,  "users.csv"),   index=False)
    ratings.to_csv(os.path.join(OUT_DIR, "ratings.csv"), index=False)

    print(f"\n  ✓ movies.csv  → {len(movies):,} rows")
    print(f"  ✓ users.csv   → {len(users):,} rows")
    print(f"  ✓ ratings.csv → {len(ratings):,} rows")
    print(f"  ✓ Done in {time.time()-t0:.1f}s")
    print(f"\n  Avg rating:      {ratings['rating'].mean():.3f}")
    print(f"  Ratings/user:    {len(ratings)/len(users):.1f}")
    print(f"  Ratings/movie:   {len(ratings)/len(movies):.1f}")
    print(f"  Matrix density:  {len(ratings)/(len(users)*len(movies))*100:.2f}%")


if __name__ == "__main__":
    main()
