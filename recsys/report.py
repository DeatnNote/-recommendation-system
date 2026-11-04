"""Build RESULTS.md from results/*.json so no number is transcribed by hand.

Usage: python -m recsys.report
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

R = Path("results")
NAMES = {"tt": "Two-tower (FAISS)", "pop": "Recent popularity", "bpr": "BPR-MF"}
FEATURE_LABELS = {
    "tt_score": "two-tower score", "tt_rank": "two-tower rank", "bpr_score": "BPR score", "bpr_rank": "BPR rank",
    "knn_score": "item-kNN score", "pop_7d": "ratings, last 7 days", "pop_30d": "ratings, last 30 days",
    "pop_all": "ratings, all time", "pop_trend": "7-day vs. 30-day trend", "item_days_since_first": "days since first rating",
    "item_mean_rating": "item mean rating (shrunk)", "item_year": "release year", "genre_affinity": "genre match with user",
    "user_n_hist": "user history length", "user_mean_rating": "user mean rating",
    "user_days_since_last": "days since user's last rating", "user_gender": "user gender", "user_age": "user age group",
    "user_occupation": "user occupation", "src_tt": "came from two-tower", "src_pop": "came from popularity",
    "src_bpr": "came from BPR",
}


def build() -> str:
    r = json.loads((R / "results.json").read_text())
    d = r["data"]
    day = lambda t: dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%d")
    out = ["# Results", "",
           "All numbers are generated from `results/*.json` by `python -m recsys.report`.", "",
           "## Setup", "",
           f"- MovieLens-1M: {d['ratings']:,} ratings, {d['n_users']:,} users, {d['n_items']:,} movies.",
           f"- Time-based split. Ranker trained on users active between {day(d['valid_cutoff_utc'])} and "
           f"{day(d['test_cutoff_utc'])} ({d['ranker_training_users']} users); tested on ratings after "
           f"{day(d['test_cutoff_utc'])}.",
           f"- Test set: {d['test_users']:,} users with history before the cutoff, {d['test_relevant_items']:,} "
           f"relevant (rated 4 or 5) items. {d['test_relevant_on_cold_items']} of those are on movies with no "
           "earlier ratings, which no model here can recommend.",
           f"- Seeds: {', '.join(map(str, r['seeds']))}. Per-user metrics are averaged over seeds, then "
           "bootstrapped over users (2,000 resamples) for 95% intervals.", "",
           "## Test-window accuracy", "",
           "| Method | NDCG@10 | Recall@10 | Recall@50 | Hit@10 | Catalog coverage@10 |",
           "|---|---|---|---|---|---|"]
    for m, v in r["methods"].items():
        cell = lambda k: f"{v[k]['mean']:.3f} ({v[k]['ci95'][0]:.3f}–{v[k]['ci95'][1]:.3f})"
        out.append(f"| {m} | {cell('ndcg@10')} | {cell('recall@10')} | {cell('recall@50')} | "
                   f"{v['hit@10']['mean']:.3f} | {100 * v['coverage@10']:.1f}% |")
    p = r["two_stage_vs_best_single"]
    out += ["", f"Recall@K divides hits by min(relevant items, K). Coverage is the share of the catalog "
                "appearing in at least one test user's top 10.", "",
            f"### Two-stage vs. the strongest single model ({p['best_single']}), paired by user", "",
            "| Metric | Difference | 95% interval | Relative |", "|---|---|---|---|"]
    for k, x in p["paired"].items():
        out.append(f"| {k} | {x['diff']:+.4f} | {x['ci95'][0]:+.4f} to {x['ci95'][1]:+.4f} | {100 * x['relative']:+.1f}% |")
    out += ["", "### Does the two-tower add anything? Two-stage with vs. without it, paired by user", "",
            "| Metric | Difference | 95% interval |", "|---|---|---|"]
    for k, x in r["two_tower_contribution"].items():
        out.append(f"| {k} | {x['diff']:+.4f} | {x['ci95'][0]:+.4f} to {x['ci95'][1]:+.4f} |")
    out += ["", "### Per-seed NDCG@10", "", "| Method | " + " | ".join(f"seed {s}" for s in r["seeds"]) + " |",
            "|---|" + "---|" * len(r["seeds"])]
    for m, v in r["methods"].items():
        out.append(f"| {m} | " + " | ".join(f"{x:.3f}" for x in v["ndcg@10"]["seed_means"]) + " |")
    c = r["candidates"]
    out += ["", "## Candidate generation", "",
            f"Each source contributes its top {c['per_source']} unseen items; the union averages "
            f"{c['mean_pool_size']:.0f} candidates per user.", "",
            "| Source | Share of a user's relevant items it retrieves |", "|---|---|"]
    for s, x in c["recall_by_source"].items():
        out.append(f"| {NAMES[s]} alone (top {c['per_source']}) | {x:.3f} |")
    out += [f"| Union of all three | {c['recall_of_pool']:.3f} |",
            f"| Union without the two-tower | {c['recall_of_pool_without_two_tower']:.3f} |", "",
            "The union is the ceiling for the ranker: it can only reorder what retrieval found.", "",
            "## What the ranker relies on", "",
            f"LightGBM LambdaRank; number of trees chosen by 5-fold cross-validation over training users (per seed: "
            f"{', '.join(map(str, r['ranker']['trees']))}). Share of total split gain:", "",
            "| Feature | Share |", "|---|---|"]
    for f, x in list(r["ranker"]["importance_share"].items())[:10]:
        out.append(f"| {FEATURE_LABELS.get(f, f)} | {100 * x:.1f}% |")
    lat = R / "latency.json"
    if lat.exists():
        L = json.loads(lat.read_text())
        out += ["", "## Serving latency", "",
                f"{L['requests']} requests for top {L['k']} ({L['note']}), on {L['cpu_cores']} CPU cores: "
                f"p50 {L['p50_ms']:.1f} ms, p95 {L['p95_ms']:.1f} ms, p99 {L['p99_ms']:.1f} ms."]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    Path("RESULTS.md").write_text(build())
    print("wrote RESULTS.md")
