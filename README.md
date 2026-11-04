# Two-stage movie recommender: retrieval + learned ranking, evaluated on a time-based split

> **This repo replaces v1** (a two-tower model on resampled MovieLens data, still in the git history). v1's README called
> the two-tower "best", but its own `results/eval_results.csv` showed it scoring below a popularity baseline (NDCG@10
> 0.0044 vs. 0.0535). v2 rebuilds the evaluation on the real MovieLens-1M data with a time-based split and reports what
> it found, including that the two-tower alone is still the weakest model.

A recommender built the way large feeds are built: several retrieval sources each propose
candidates, then a learned ranker orders them. It's trained and evaluated on the real
MovieLens-1M dataset (1,000,209 ratings), with a time-based split so no model ever sees ratings
made after the ones it's tested on, and it's served through a FastAPI endpoint.

```
user ──► two-tower user vector ──► FAISS top-100 ─┐
     ──► recent popularity (last 7 days) top-100 ─┼─► ~224 unseen candidates ──► 22 features ──► LightGBM LambdaRank ──► top-k
     ──► BPR matrix factorization top-100 ────────┘
```

## Results (test window, 1,159 users, 3 seeds)

Full tables with 95% intervals: [`RESULTS.md`](RESULTS.md), generated from `results/*.json`.

| Method | NDCG@10 | Recall@10 | Recall@50 | Catalog coverage@10 |
|---|---|---|---|---|
| Popularity (last 7 days) | 0.210 | 0.204 | 0.217 | 3.0% |
| Item-kNN | 0.185 | 0.179 | 0.194 | 7.1% |
| BPR matrix factorization | 0.152 | 0.151 | 0.201 | 39.4% |
| Two-tower retrieval only | 0.133 | 0.130 | 0.170 | 31.1% |
| **Two-stage: candidates + ranker** | **0.233** | **0.234** | **0.284** | 12.5% |
| Two-stage without the two-tower | 0.226 | 0.226 | 0.269 | 10.3% |

- **The two-stage system beats the strongest single model by 10.8% on NDCG@10** (paired by user:
  +0.023, 95% interval +0.015 to +0.031) and by 31% on Recall@50, while recommending four times
  as much of the catalog.
- **The strongest single model is plain recent popularity.** On a time-based split, what people
  rate next is heavily driven by what's current, and every personalized model alone, including
  the two-tower, scored below it. The ranker wins by combining personalization with freshness: its
  top features are ratings in the last 7 days, release year, and item quality.
- **The two-tower is the weakest model alone but still earns its place.** It raises the share of
  relevant items reaching the ranker from 0.388 to 0.453, and removing it lowers NDCG@10 by 0.007
  (95% interval 0.003 to 0.011).
- **Serving:** p50 11.4 ms, p99 18.7 ms per request for the full path (FAISS retrieval, features,
  ranker, HTTP layer), in-process on 2 CPU cores.

## What is and isn't shown here

- **Real data, offline evaluation only.** No live users and no online A/B test. Offline gains
  often shrink online.
- **MovieLens ratings aren't views.** A rating of 4 or 5 is treated as a relevant recommendation.
- **The two-tower underperforms on its own, and tuning didn't fix it.** I tried an exact softmax
  over the catalog and in-batch sampled softmax with log-frequency correction, temperatures
  0.05–0.2, an "example age" feature (Covington et al., 2016), and dropping the user-id embedding.
  All landed at 0.32–0.35 validation Recall@200, against 0.46 for recent popularity.
- **The ranker learns from 492 users**, the ones active in the four-week window before the test
  cutoff (MovieLens-1M users rate in bursts, so few are active in any given window).
- **Small catalog.** 3,883 movies, so FAISS uses an exact flat index. At a large catalog this
  would need an approximate index (IVF or HNSW) and a recall-vs-latency trade-off.
- **No cold start for new movies.** 45 of the 52,107 relevant test items are movies with no
  earlier ratings, which no model here can recommend.

## Problems caught during development

1. **Unstable ranker size.** Early stopping on a single 20% holdout (~100 users) picked 82, 10,
   and 34 trees across three seeds, and the 10-tree model scored below the popularity baseline.
   The number of trees is now chosen from the mean NDCG@10 curve across 5 folds of users. Per-seed
   NDCG@10 is now 0.230–0.235.
2. **The API and the offline evaluation disagreed for 6 of 200 users.** Tree models give exact
   score ties, and the two code paths sorted ties differently. Ties are now broken by item id in
   both paths, and a test checks that the API returns the same top 10 as the batch evaluation.
3. **123 ms median latency.** Profiling showed 104 ms of each request going to item-kNN scoring,
   which copied a dense 60 MB similarity matrix on every call. Storing only each item's top
   neighbors as a sparse matrix brought the median to 11.4 ms.

## How it works

| Component | File |
|---|---|
| Data loading and time-based split | `recsys/data.py` |
| Popularity, item-kNN, BPR-MF baselines | `recsys/baselines.py` |
| Two-tower model, FAISS retrieval | `recsys/two_tower.py` |
| Candidates, features, LambdaRank ranker | `recsys/ranker.py` |
| Experiment runner (3 seeds) | `recsys/pipeline.py` |
| FastAPI service, latency benchmark | `recsys/serve.py`, `recsys/bench.py` |
| Report generator | `recsys/report.py` |

**Split.** Ratings are cut at the 80th and 90th percentiles of time. Stage-1 models are fit on
events before the 80th percentile, and the ranker learns from labels between the 80th and 90th.
For the test, stage-1 models are refit on everything before the 90th percentile, and the trained
ranker scores ratings after it. All hyperparameters were chosen on the validation window only.

**Two-tower.** The user tower takes the user id, demographics, and the mean of the 50 most recent
items. The item tower takes the item id, genres, and release year. Each training example only sees
the items the user rated *before* it, matching what the model gets at serving time.

**Ranker features.** The 22 features are retrieval scores and ranks (two-tower, BPR, item-kNN),
recent and all-time popularity and trend, days since an item's first rating, shrunk mean rating,
release year, genre match with the user's history, user activity, demographics, and which
sources proposed the item. All are
computed only from events before the cutoff.

**Metrics.** NDCG@10, Recall@10 and @50 (hits divided by min(relevant, K)), Hit@10, and catalog
coverage. Per-user metrics are averaged over three seeds, then bootstrapped over users.

## How to run

```bash
python -m venv venv && . venv/bin/activate      # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt

python scripts/get_data.py       # downloads MovieLens-1M from GroupLens into data/raw/
python -m recsys.pipeline        # 3 seeds, about 11 minutes on 2 CPU cores
python -m recsys.bench           # latency
python -m recsys.report          # regenerate RESULTS.md
pytest -q                        # 16 tests

uvicorn recsys.serve:app --port 8000
curl "localhost:8000/recommend/1?k=10"
```

The dataset isn't included in this repo (GroupLens terms). The committed results were produced
from a tab-separated conversion of MovieLens-1M with the official counts (1,000,209 ratings, 6,040
users, 3,883 movies). The loader reads either that format or GroupLens' own `.dat` files, and a
test checks that both load identically. Exact numbers may still differ slightly by machine,
because PyTorch and LightGBM training aren't bit-for-bit reproducible across hardware.
