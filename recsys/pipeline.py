"""End-to-end experiment: baselines vs. two-tower vs. two-stage, on a time-based MovieLens-1M split.

Protocol (per seed):
  1. Fit stage-1 models on events before q80; build candidates + features for users active in
     [q80, q90); label = rated >= 4 in that window; train the LambdaRank ranker.
  2. Refit stage-1 models on events before q90; build candidates + features for test users;
     apply the ranker trained in step 1; evaluate on ratings >= 4 after q90.
Hyperparameters were chosen on the [q80, q90) window only; the test window was not used for tuning.

Usage: python -m recsys.pipeline            (3 seeds, about 11 minutes on 2 CPU cores)
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from . import two_tower
from .baselines import mask_seen, top_k
from .data import load_raw, load_splits
from .metrics import bootstrap_ci, coverage, per_user
from .ranker import FEATURES, build_features, candidates, fit_stage1, rank, train_ranker

RESULTS = Path("results")
ARTIFACTS = Path("artifacts")
SEEDS = (0, 1, 2)
NO_TT = [f for f in FEATURES if f not in ("tt_score", "tt_rank", "src_tt")]
TT_EPOCHS = 2


def full_catalog_recs(model, snap, users, k=200):
    r = top_k(mask_seen(model.score(users), users, snap), k)
    return {u: list(r[j]) for j, u in enumerate(users)}


def candidate_recall(cands, labels, users):
    """Share of each user's relevant items that made it into the candidate set (the ranker's ceiling)."""
    out = []
    for u in users:
        pool = cands[u]["tt"] | cands[u]["pop"] | cands[u]["bpr"]
        out.append(len(labels[u] & pool) / len(labels[u]))
    return np.array(out), float(np.mean([len(cands[u]["tt"] | cands[u]["pop"] | cands[u]["bpr"]) for u in users]))


def run_seed(cat, valid, test, seed, save_artifacts=False):
    t0 = time.time()
    log = lambda m: print(f"[seed {seed} | {time.time() - t0:5.0f}s] {m}", flush=True)
    # ---- ranker training data (everything before q80; labels in [q80, q90)) ----
    tt_v = two_tower.train(cat, valid, seed=seed, epochs=TT_EPOCHS)
    s1_v = fit_stage1(cat, valid, tt_v, seed)
    vu = valid.eval_users()
    df_v = build_features(cat, s1_v, valid, vu, candidates(s1_v, valid, vu))
    log(f"ranker training rows {len(df_v)}, users {len(vu)}, positive rate {df_v.label.mean():.4f}")
    ranker = train_ranker(df_v, FEATURES, seed=seed)
    log(f"ranker trained ({ranker.n_trees_} trees, CV NDCG@10 {ranker.cv_ndcg10_:.3f})")
    # ablation: is the two-tower pulling its weight? drop its candidates and its features
    no_tt = lambda df: df[(df.src_pop == 1) | (df.src_bpr == 1)]
    ranker_no_tt = train_ranker(no_tt(df_v), NO_TT, seed=seed)
    # ---- test (everything before q90; labels after) ----
    tt_t = two_tower.train(cat, test, seed=seed, epochs=TT_EPOCHS)
    s1_t = fit_stage1(cat, test, tt_t, seed)
    tu = test.eval_users()
    cands = candidates(s1_t, test, tu)
    df_t = build_features(cat, s1_t, test, tu, cands)
    recs = {
        "Popularity (last 7 days)": full_catalog_recs(s1_t.pop, test, tu),
        "Item-kNN": full_catalog_recs(s1_t.knn, test, tu),
        "BPR matrix factorization": full_catalog_recs(s1_t.bpr, test, tu),
        "Two-tower retrieval only": full_catalog_recs(s1_t.tt, test, tu),
        "Two-stage: candidates + ranker": rank(ranker, df_t, FEATURES),
        "Two-stage without the two-tower": rank(ranker_no_tt, no_tt(df_t), NO_TT),
    }
    per = {name: per_user(r, test.labels, tu) for name, r in recs.items()}
    cov = {name: coverage(r, tu, cat.n_items) for name, r in recs.items()}
    cand_rec, cand_size = candidate_recall(cands, test.labels, tu)
    pool_no_tt = float(np.mean([len(test.labels[u] & (cands[u]["pop"] | cands[u]["bpr"])) / len(test.labels[u])
                                for u in tu]))
    source_rec = {src: float(np.mean([len(test.labels[u] & cands[u][src]) / len(test.labels[u]) for u in tu]))
                  for src in ("tt", "pop", "bpr")}
    imp = dict(zip(FEATURES, ranker.booster_.feature_importance("gain").tolist()))
    log("evaluated")
    if save_artifacts:
        ARTIFACTS.mkdir(exist_ok=True)
        with open(ARTIFACTS / "service.pkl", "wb") as f:
            pickle.dump({"cat": cat, "snap": test, "stage1": s1_t, "ranker": ranker}, f)
        log("saved serving artifacts")
    return {"users": tu, "per_user": per, "coverage": cov, "candidate_recall": cand_rec,
            "candidate_pool_size": cand_size, "source_recall": source_rec, "pool_no_tt": pool_no_tt, "importance": imp,
            "ranker_trees": int(ranker.n_trees_)}


def main(seeds=SEEDS):
    cat, valid, test = load_splits()
    RESULTS.mkdir(exist_ok=True)
    runs = [run_seed(cat, valid, test, s, save_artifacts=(s == seeds[0])) for s in seeds]
    users = runs[0]["users"]
    methods = list(runs[0]["per_user"])
    metrics = list(runs[0]["per_user"][methods[0]])
    out = {
        "data": {"ratings": int(len(load_raw()[0])), "n_users": cat.n_users, "n_items": cat.n_items,
                 "test_cutoff_utc": int(test.cutoff), "valid_cutoff_utc": int(valid.cutoff),
                 "test_users": int(len(users)), "test_relevant_items": int(sum(len(test.labels[u]) for u in users)),
                 "test_relevant_on_cold_items": int(test.cold_label_items),
                 "ranker_training_users": int(len(valid.eval_users()))},
        "seeds": list(seeds),
        "methods": {},
    }
    for m in methods:
        # per-user metric averaged over seeds, then bootstrapped over users
        res = {}
        for k in metrics:
            avg = np.mean([r["per_user"][m][k] for r in runs], axis=0)
            mean, lo, hi = bootstrap_ci(avg)
            res[k] = {"mean": mean, "ci95": [lo, hi],
                      "seed_means": [float(np.mean(r["per_user"][m][k])) for r in runs]}
        res["coverage@10"] = float(np.mean([r["coverage"][m] for r in runs]))
        out["methods"][m] = res
    # paired comparison: two-stage vs. the strongest single model on NDCG@10
    singles = [m for m in methods if not m.startswith("Two-stage")]
    best = max(singles, key=lambda m: out["methods"][m]["ndcg@10"]["mean"])
    pairs = {}
    for k in ("ndcg@10", "recall@10", "recall@50"):
        d = np.mean([r["per_user"]["Two-stage: candidates + ranker"][k] - r["per_user"][best][k] for r in runs], axis=0)
        mean, lo, hi = bootstrap_ci(d)
        base = out["methods"][best][k]["mean"]
        pairs[k] = {"diff": mean, "ci95": [lo, hi], "relative": mean / base}
    out["two_stage_vs_best_single"] = {"best_single": best, "paired": pairs}
    out["two_tower_contribution"] = {}
    for k in ("ndcg@10", "recall@10", "recall@50"):
        d = np.mean([r["per_user"]["Two-stage: candidates + ranker"][k] - r["per_user"]["Two-stage without the two-tower"][k]
                     for r in runs], axis=0)
        mean, lo, hi = bootstrap_ci(d)
        out["two_tower_contribution"][k] = {"diff": mean, "ci95": [lo, hi]}
    out["candidates"] = {"per_source": 100, "mean_pool_size": float(np.mean([r["candidate_pool_size"] for r in runs])),
                         "recall_of_pool": float(np.mean([r["candidate_recall"].mean() for r in runs])),
                         "recall_by_source": {s: float(np.mean([r["source_recall"][s] for r in runs]))
                                              for s in ("tt", "pop", "bpr")},
                         "recall_of_pool_without_two_tower": float(np.mean([r["pool_no_tt"] for r in runs]))}
    imp = {f: float(np.mean([r["importance"][f] for r in runs])) for f in FEATURES}
    tot = sum(imp.values())
    out["ranker"] = {"importance_share": {f: v / tot for f, v in sorted(imp.items(), key=lambda x: -x[1])},
                     "trees": [r["ranker_trees"] for r in runs]}
    (RESULTS / "results.json").write_text(json.dumps(out, indent=2))
    print("wrote results/results.json")


if __name__ == "__main__":
    main(tuple(int(s) for s in sys.argv[1:]) or SEEDS)
