# Results

All numbers are generated from `results/*.json` by `python -m recsys.report`.

## Setup

- MovieLens-1M: 1,000,209 ratings, 6,040 users, 3,883 movies.
- Time-based split. Ranker trained on users active between 2000-12-02 and 2000-12-29 (492 users); tested on ratings after 2000-12-29.
- Test set: 1,159 users with history before the cutoff, 52,107 relevant (rated 4 or 5) items. 45 of those are on movies with no earlier ratings, which no model here can recommend.
- Seeds: 0, 1, 2. Per-user metrics are averaged over seeds, then bootstrapped over users (2,000 resamples) for 95% intervals.

## Test-window accuracy

| Method | NDCG@10 | Recall@10 | Recall@50 | Hit@10 | Catalog coverage@10 |
|---|---|---|---|---|---|
| Popularity (last 7 days) | 0.210 (0.197–0.223) | 0.204 (0.191–0.217) | 0.217 (0.206–0.227) | 0.665 | 3.0% |
| Item-kNN | 0.185 (0.172–0.199) | 0.179 (0.167–0.193) | 0.194 (0.183–0.205) | 0.569 | 7.1% |
| BPR matrix factorization | 0.152 (0.142–0.162) | 0.151 (0.141–0.160) | 0.201 (0.192–0.210) | 0.567 | 39.4% |
| Two-tower retrieval only | 0.133 (0.123–0.143) | 0.130 (0.121–0.140) | 0.170 (0.161–0.181) | 0.482 | 31.1% |
| Two-stage: candidates + ranker | 0.233 (0.220–0.245) | 0.234 (0.222–0.246) | 0.284 (0.272–0.295) | 0.734 | 12.5% |
| Two-stage without the two-tower | 0.226 (0.214–0.238) | 0.226 (0.214–0.238) | 0.269 (0.258–0.280) | 0.717 | 10.3% |

Recall@K divides hits by min(relevant items, K). Coverage is the share of the catalog appearing in at least one test user's top 10.

### Two-stage vs. the strongest single model (Popularity (last 7 days)), paired by user

| Metric | Difference | 95% interval | Relative |
|---|---|---|---|
| ndcg@10 | +0.0228 | +0.0151 to +0.0305 | +10.8% |
| recall@10 | +0.0301 | +0.0219 to +0.0384 | +14.8% |
| recall@50 | +0.0670 | +0.0580 to +0.0758 | +30.9% |

### Does the two-tower add anything? Two-stage with vs. without it, paired by user

| Metric | Difference | 95% interval |
|---|---|---|
| ndcg@10 | +0.0068 | +0.0030 to +0.0106 |
| recall@10 | +0.0085 | +0.0042 to +0.0130 |
| recall@50 | +0.0147 | +0.0109 to +0.0188 |

### Per-seed NDCG@10

| Method | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| Popularity (last 7 days) | 0.210 | 0.210 | 0.210 |
| Item-kNN | 0.185 | 0.185 | 0.185 |
| BPR matrix factorization | 0.151 | 0.152 | 0.151 |
| Two-tower retrieval only | 0.131 | 0.135 | 0.133 |
| Two-stage: candidates + ranker | 0.234 | 0.230 | 0.235 |
| Two-stage without the two-tower | 0.225 | 0.219 | 0.235 |

## Candidate generation

Each source contributes its top 100 unseen items; the union averages 224 candidates per user.

| Source | Share of a user's relevant items it retrieves |
|---|---|
| Two-tower (FAISS) alone (top 100) | 0.226 |
| Recent popularity alone (top 100) | 0.258 |
| BPR-MF alone (top 100) | 0.262 |
| Union of all three | 0.453 |
| Union without the two-tower | 0.388 |

The union is the ceiling for the ranker: it can only reorder what retrieval found.

## What the ranker relies on

LightGBM LambdaRank; number of trees chosen by 5-fold cross-validation over training users (per seed: 27, 88, 78). Share of total split gain:

| Feature | Share |
|---|---|
| ratings, last 7 days | 13.4% |
| release year | 12.4% |
| item mean rating (shrunk) | 12.4% |
| BPR rank | 8.7% |
| item-kNN score | 6.3% |
| days since first rating | 5.9% |
| BPR score | 5.1% |
| days since user's last rating | 5.0% |
| genre match with user | 4.8% |
| user history length | 4.0% |

## Serving latency

500 requests for top 10 (in-process FastAPI TestClient, single request at a time, no network hop), on 2 CPU cores: p50 11.4 ms, p95 15.1 ms, p99 18.7 ms.
