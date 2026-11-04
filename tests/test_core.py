import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from recsys.baselines import Popularity, mask_seen, top_k
from recsys.data import HISTORY_LEN, POSITIVE_RATING, load_splits, make_snapshot
from recsys.metrics import bootstrap_ci, ndcg_at_k, recall_at_k
from recsys.two_tower import _sequences, history_window

ART = Path("artifacts/service.pkl")
HAS_DATA = any((Path("data/raw") / f).exists() for f in ("ratings.csv", "ratings.dat"))
needs_data = pytest.mark.skipif(not HAS_DATA, reason="MovieLens-1M not downloaded: python scripts/get_data.py")


@pytest.fixture(scope="module")
def splits():
    if not HAS_DATA:
        pytest.skip("MovieLens-1M not downloaded: python scripts/get_data.py")
    return load_splits()


# ---------- metrics ----------
def test_ndcg_hand_computed():
    # relevant {a, c}; ranked a, b, c -> DCG = 1 + 1/log2(4); IDCG = 1 + 1/log2(3)
    expected = (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(["a", "b", "c"], {"a", "c"}, 10) == pytest.approx(expected)
    assert ndcg_at_k(["a", "c", "b"], {"a", "c"}, 10) == pytest.approx(1.0)
    assert ndcg_at_k(["b", "d"], {"a"}, 10) == 0.0


def test_recall_is_normalized_by_min_of_relevant_and_k():
    assert recall_at_k([1, 2, 3], {1, 9}, 2) == 0.5
    assert recall_at_k([1, 2], {1, 2, 3, 4}, 2) == 1.0      # 2 hits out of min(4, 2)


def test_bootstrap_ci_brackets_mean():
    x = np.random.default_rng(0).normal(1.0, 1.0, 500)
    m, lo, hi = bootstrap_ci(x)
    assert lo < m < hi and lo < 1.0 < hi


# ---------- split: nothing from the future ----------
def test_history_strictly_before_cutoff(splits):
    _, valid, test = splits
    assert valid.history.t.max() < valid.cutoff
    assert test.history.t.max() < test.cutoff
    assert valid.cutoff < test.cutoff


def test_labels_exclude_already_rated_items(splits):
    _, valid, test = splits
    for snap in (valid, test):
        for u, items in snap.labels.items():
            assert not (items & snap.seen.get(u, set()))


def test_validation_labels_come_from_before_test_cutoff():
    d = pd.DataFrame({"u": [0, 0, 0, 1], "i": [1, 2, 3, 1], "rating": [5, 5, 5, 5], "t": [10, 20, 30, 25]})
    snap = make_snapshot(d, cutoff=15, end=25)
    assert snap.labels == {0: {2}}            # t=25 and t=30 are outside [15, 25)
    assert set(snap.history.t) == {10}


def test_low_ratings_are_not_labels():
    d = pd.DataFrame({"u": [0, 0, 0], "i": [1, 2, 3], "rating": [5, POSITIVE_RATING - 1, 5], "t": [1, 5, 6]})
    snap = make_snapshot(d, cutoff=2, end=None)
    assert snap.labels == {0: {3}}


# ---------- two-tower training input ----------
def test_training_history_only_contains_earlier_items():
    d = pd.DataFrame({"u": [0, 0, 0, 1], "i": [5, 6, 7, 8], "rating": 5, "t": [1, 2, 3, 4]})
    snap = make_snapshot(d, cutoff=100, end=None)
    pad = 99
    flat, offsets, _ = _sequences(snap, 2, pad)
    h = history_window(flat, offsets, np.array([0, 0, 0, 1]), np.array([0, 1, 2, 0]))
    assert h.shape == (4, HISTORY_LEN)
    assert set(h[0]) == {pad}                         # first item: no history
    assert list(h[1][-1:]) == [5] and 6 not in h[1]   # second item sees only the first
    assert list(h[2][-2:]) == [5, 6] and 7 not in h[2]
    assert set(h[3]) == {pad}                         # other user's items never leak in


# ---------- baselines ----------
def test_seen_items_are_never_recommended(splits):
    cat, _, test = splits
    users = test.eval_users()[:50]
    pop = Popularity(7).fit(test, cat.n_users, cat.n_items)
    recs = top_k(mask_seen(pop.score(users), users, test), 50)
    for r, u in enumerate(users):
        assert not (set(recs[r]) & test.seen[u])


def test_popularity_uses_only_recent_window():
    d = pd.DataFrame({"u": [0, 1, 2, 3], "i": [0, 0, 0, 1], "rating": 5,
                      "t": [0, 0, 0, 100 * 86400]})
    snap = make_snapshot(d, cutoff=101 * 86400, end=None)
    pop = Popularity(7).fit(snap, 4, 2)
    assert pop.scores[1] > pop.scores[0]     # item 0 is more watched overall, but not in the last week


# ---------- ranker features are computed from history only ----------
def test_item_stats_ignore_future_events(splits):
    from recsys.ranker import ItemStats
    cat, _, test = splits
    s = ItemStats(cat, test)
    h = test.history
    expected = np.log1p(np.bincount(h.i.values, minlength=cat.n_items))
    assert np.allclose(s.pop_all, expected)


# ---------- serving (needs artifacts from `python -m recsys.pipeline`) ----------
@pytest.mark.skipif(not ART.exists(), reason="run the pipeline first")
def test_faiss_retrieval_matches_exact_scores():
    from recsys.serve import Recommender
    rec = Recommender()
    users = rec.snap.eval_users()[:20]
    dense = top_k(mask_seen(rec.s1.tt.score(users), users, rec.snap), 100)
    for r, u in enumerate(users):
        items, _ = rec.s1.tt.retrieve(u, 100)
        assert set(items) == set(dense[r])


@pytest.mark.skipif(not ART.exists(), reason="run the pipeline first")
def test_api_returns_unseen_items_and_handles_errors():
    from fastapi.testclient import TestClient
    from recsys.serve import app, get
    c = TestClient(app)
    assert c.get("/healthz").json() == {"status": "ok"}
    rec = get()
    uid = int(rec.cat.user_ids[rec.snap.eval_users()[0]])
    body = c.get(f"/recommend/{uid}?k=10").json()
    assert len(body["items"]) == 10
    seen_movies = {int(rec.cat.item_ids[i]) for i in rec.snap.seen[rec.cat.user_index[uid]]}
    assert not ({it["movie_id"] for it in body["items"]} & seen_movies)
    scores = [it["score"] for it in body["items"]]
    assert scores == sorted(scores, reverse=True)
    assert c.get("/recommend/999999").status_code == 404
    assert c.get(f"/recommend/{uid}?k=0").status_code == 400


@needs_data
def test_official_dat_format_loads_identically(tmp_path):
    """The repo ships a loader for GroupLens' own '::' files; check it matches the CSV conversion."""
    from recsys.data import load_raw
    r, m, u = load_raw()
    sub_r = r.head(2000)
    (tmp_path / "ratings.dat").write_text("\n".join("::".join(map(str, x)) for x in sub_r.itertuples(index=False)))
    (tmp_path / "movies.dat").write_bytes("\n".join("::".join(map(str, x)) for x in m.itertuples(index=False)).encode("latin-1"))
    (tmp_path / "users.dat").write_text("\n".join("::".join(map(str, x)) for x in u.itertuples(index=False)))
    r2, m2, u2 = load_raw(tmp_path)
    pd.testing.assert_frame_equal(sub_r.reset_index(drop=True), r2, check_dtype=False)
    pd.testing.assert_frame_equal(m.reset_index(drop=True), m2, check_dtype=False)
    pd.testing.assert_frame_equal(u.reset_index(drop=True), u2, check_dtype=False)


@pytest.mark.skipif(not ART.exists(), reason="run the pipeline first")
def test_serving_matches_offline_pipeline():
    """No train/serve skew: the API's per-request path returns the same top 10 as the batch evaluation."""
    from recsys.ranker import FEATURES, build_features, candidates, rank
    from recsys.serve import get
    rec = get()
    users = rec.snap.eval_users()[:15]
    df = build_features(rec.cat, rec.s1, rec.snap, users, candidates(rec.s1, rec.snap, users), with_labels=False)
    offline = rank(rec.ranker, df, FEATURES, k=10)
    for u in users:
        online = [rec.cat.item_index[x["movie_id"]] for x in rec.recommend(int(rec.cat.user_ids[u]), 10)]
        assert online == list(offline[u])


def test_ties_are_broken_by_item_id():
    from recsys.ranker import order_by_score
    items = np.array([7, 3, 5, 1])
    scores = np.array([0.5, 0.9, 0.5, 0.1])
    assert list(order_by_score(items, scores)) == [3, 5, 7, 1]
