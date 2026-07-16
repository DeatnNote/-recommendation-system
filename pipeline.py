"""
pipeline.py
===========
End-to-end pipeline: load → preprocess → train baselines → train two-tower
→ evaluate all three → save results and model.

Run:
  python pipeline.py

Outputs:
  models/two_tower.pt        — saved two-tower model weights
  models/two_tower.faiss     — FAISS item index
  models/mf.pt               — saved MF model weights
  results/eval_results.csv   — offline metrics for all models
  results/training_curves.csv — loss curves
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import os, json, time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.dataset   import load_data, remap_ids, leave_one_out_split, to_implicit, RatingsDataset, PointwiseDataset
from src.baselines  import PopularityBaseline, MatrixFactorization
from src.two_tower  import TwoTower
from evaluation.metrics import evaluate, print_results

os.makedirs("models",  exist_ok=True)
os.makedirs("results", exist_ok=True)

# ── Config ────────────────────────────────────────────────────
EMBEDDING_DIM = 128
OUTPUT_DIM    = 64
BATCH_SIZE    = 2048
EPOCHS        = 5
LR            = 0.001
N_NEGATIVES   = 4
EVAL_SAMPLE   = 1000   # users to evaluate on (faster than full set)
K_VALUES      = [5, 10, 20]
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("  RECOMMENDATION SYSTEM PIPELINE")
print("=" * 60)
print(f"  Device: {DEVICE}")

# ── 1. Load & preprocess ──────────────────────────────────────
print("\n[1] LOADING DATA")
print("─" * 40)
ratings, movies, users = load_data()
ratings, user2idx, movie2idx = remap_ids(ratings)

N_USERS  = len(user2idx)
N_MOVIES = len(movie2idx)
print(f"  Users:   {N_USERS:,}")
print(f"  Movies:  {N_MOVIES:,}")
print(f"  Ratings: {len(ratings):,}")

# ── 2. Split ──────────────────────────────────────────────────
print("\n[2] SPLITTING DATA (leave-one-out)")
print("─" * 40)
train, val, test = leave_one_out_split(ratings)

# Convert to implicit for two-tower (BPR) training
train_implicit = to_implicit(train)
print(f"  Implicit train (rating ≥ 3.5): {len(train_implicit):,}")

# ── 3. DataLoaders ────────────────────────────────────────────
bpr_dataset = RatingsDataset(train_implicit, N_MOVIES, n_negatives=N_NEGATIVES)
bpr_loader  = DataLoader(bpr_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

mf_dataset  = PointwiseDataset(train)
mf_loader   = DataLoader(mf_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# ── 4. Evaluation user sample ─────────────────────────────────
eval_users = test["user_idx"].unique()
np.random.seed(42)
eval_users = np.random.choice(eval_users, size=min(EVAL_SAMPLE, len(eval_users)), replace=False)
eval_users = eval_users.tolist()

# ── 5. Train & evaluate Popularity Baseline ───────────────────
print("\n[3] POPULARITY BASELINE")
print("─" * 40)
t0  = time.time()
pop = PopularityBaseline()
pop.fit(train)
pop_recs = pop.recommend_batch(eval_users, train, n=max(K_VALUES))
pop_results = evaluate(pop_recs, test, train, k_values=K_VALUES)
print_results("Popularity Baseline", pop_results)
print(f"  Time: {time.time()-t0:.1f}s")

# ── 6. Train & evaluate Matrix Factorization ──────────────────
print("\n[4] MATRIX FACTORIZATION (baseline)")
print("─" * 40)
t0 = time.time()
mf = MatrixFactorization(
    n_users=N_USERS, n_movies=N_MOVIES,
    embedding_dim=64, lr=LR, batch_size=BATCH_SIZE,
    epochs=EPOCHS, device=DEVICE,
)
mf.fit(mf_loader)
torch.save({"model_state": mf.model.state_dict(), "train_losses": mf.train_losses},
           "models/mf.pt")
mf_recs = mf.recommend_batch(eval_users, train, n=max(K_VALUES))
mf_results = evaluate(mf_recs, test, train, k_values=K_VALUES)
print_results("Matrix Factorization", mf_results)
print(f"  Time: {time.time()-t0:.1f}s")

# ── 7. Train & evaluate Two-Tower ────────────────────────────
print("\n[5] TWO-TOWER MODEL")
print("─" * 40)
t0 = time.time()
tt = TwoTower(
    n_users=N_USERS, n_movies=N_MOVIES,
    embedding_dim=EMBEDDING_DIM, output_dim=OUTPUT_DIM,
    lr=LR, batch_size=BATCH_SIZE,
    epochs=EPOCHS, device=DEVICE,
)
tt.fit(bpr_loader)
tt.save("models/two_tower.pt")
tt_recs = tt.recommend_batch(eval_users, train, n=max(K_VALUES))
tt_results = evaluate(tt_recs, test, train, k_values=K_VALUES)
print_results("Two-Tower (BPR + FAISS)", tt_results)
print(f"  Time: {time.time()-t0:.1f}s")

# ── 8. Comparison summary ────────────────────────────────────
print("\n[6] COMPARISON SUMMARY (k=10)")
print("─" * 40)
k10_pop = pop_results[pop_results["k"] == 10].iloc[0]
k10_mf  =  mf_results[mf_results["k"]  == 10].iloc[0]
k10_tt  =  tt_results[tt_results["k"]  == 10].iloc[0]

print(f"  {'Model':<28} {'NDCG@10':>9} {'Recall@10':>10} {'Hit@10':>8} {'Coverage':>9}")
print(f"  {'─'*65}")
for name, row in [("Popularity Baseline", k10_pop),
                  ("Matrix Factorization", k10_mf),
                  ("Two-Tower (BPR+FAISS)", k10_tt)]:
    print(f"  {name:<28} {row['ndcg']:>9.4f} {row['recall']:>10.4f} "
          f"{row['hit_rate']:>8.4f} {row['coverage']:>9.4f}")

ndcg_lift_vs_pop = (k10_tt["ndcg"] - k10_pop["ndcg"]) / max(k10_pop["ndcg"], 1e-10) * 100
ndcg_lift_vs_mf  = (k10_tt["ndcg"] - k10_mf["ndcg"])  / max(k10_mf["ndcg"],  1e-10) * 100
print(f"\n  Two-Tower NDCG@10 lift vs Popularity: {ndcg_lift_vs_pop:+.1f}%")
print(f"  Two-Tower NDCG@10 lift vs MF:         {ndcg_lift_vs_mf:+.1f}%")

# ── 9. Save results ───────────────────────────────────────────
pop_results["model"] = "popularity"
mf_results["model"]  = "matrix_factorization"
tt_results["model"]  = "two_tower"
all_results = pd.concat([pop_results, mf_results, tt_results], ignore_index=True)
all_results.to_csv("results/eval_results.csv", index=False)

loss_curves = pd.DataFrame({
    "epoch": range(1, EPOCHS + 1),
    "mf_loss": mf.train_losses,
    "tt_loss": tt.train_losses,
})
loss_curves.to_csv("results/training_curves.csv", index=False)

summary = {
    "n_users":  N_USERS,
    "n_movies": N_MOVIES,
    "n_ratings": len(ratings),
    "device":   DEVICE,
    "epochs":   EPOCHS,
    "two_tower_ndcg_at_10": float(k10_tt["ndcg"]),
    "mf_ndcg_at_10":        float(k10_mf["ndcg"]),
    "pop_ndcg_at_10":       float(k10_pop["ndcg"]),
    "ndcg_lift_vs_pop_pct": round(ndcg_lift_vs_pop, 2),
    "ndcg_lift_vs_mf_pct":  round(ndcg_lift_vs_mf, 2),
}
with open("results/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 60)
print("  Results saved → results/")
print("  Models  saved → models/")
print("=" * 60)
