"""Latency of the full request path (retrieval + features + ranker + HTTP layer), measured in-process.

Usage: python -m recsys.bench     -> results/latency.json
This runs FastAPI's TestClient in the same process, so it includes request parsing and JSON
serialization but no real network hop. Numbers are from whatever machine runs it.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from .serve import app, get


def main(n: int = 500, k: int = 10):
    c = TestClient(app)
    rec = get()
    users = [int(rec.cat.user_ids[u]) for u in rec.snap.eval_users()[:n]]
    for uid in users[:20]:                     # warm-up
        c.get(f"/recommend/{uid}?k={k}")
    times = []
    for uid in users:
        t = time.perf_counter()
        r = c.get(f"/recommend/{uid}?k={k}")
        times.append((time.perf_counter() - t) * 1000)
        assert r.status_code == 200
    t = np.array(times)
    out = {"requests": len(t), "k": k, "p50_ms": float(np.percentile(t, 50)), "p95_ms": float(np.percentile(t, 95)),
           "p99_ms": float(np.percentile(t, 99)), "cpu_cores": os.cpu_count(),
           "note": "in-process FastAPI TestClient, single request at a time, no network hop"}
    Path("results").mkdir(exist_ok=True)
    Path("results/latency.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
