"""
dashboard/app.py
================
Streamlit dashboard for the Movie Recommendation System.

Tabs:
  1. 🎬 Recommendations  — pick a user, see personalised top-10
  2. 🔍 Similar Movies   — item-item similarity via FAISS
  3. 📊 Model Comparison — Popularity vs MF vs Two-Tower metrics
  4. 📈 Training Curves  — loss curves for MF and Two-Tower
  5. 🗂️ Dataset Explorer — rating distributions, genre stats

Run:
  streamlit run dashboard/app.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.express as px
import plotly.graph_objects as go

from src.dataset   import load_data, remap_ids
from src.two_tower import TwoTowerModel
import faiss

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Movie RecSys",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

BG      = "#0d1117"
CARD    = "#161b22"
BORDER  = "#30363d"
BLUE    = "#58a6ff"
GREEN   = "#3fb950"
ORANGE  = "#e3b341"
RED     = "#f85149"
TEXT    = "#e6edf3"
SUBTEXT = "#8b949e"

st.markdown(f"""
<style>
  .stApp {{ background-color:{BG}; color:{TEXT}; }}
  section[data-testid="stSidebar"] {{ background-color:{CARD}; border-right:1px solid {BORDER}; }}
  div[data-testid="stMetricValue"] {{ color:{BLUE}; }}
  .rec-card {{
    background:{CARD}; border:1px solid {BORDER}; border-radius:8px;
    padding:0.8rem 1rem; margin-bottom:0.4rem;
  }}
  .rec-title {{ font-weight:600; color:{TEXT}; }}
  .rec-meta  {{ font-size:0.78rem; color:{SUBTEXT}; }}
  .score-pill {{
    background:{BLUE}22; color:{BLUE}; border-radius:12px;
    padding:2px 10px; font-size:0.75rem; font-weight:600;
  }}
</style>
""", unsafe_allow_html=True)

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font_color=TEXT, font_family="monospace",
    margin=dict(l=20, r=20, t=40, b=20),
)

def styled_fig(fig):
    fig.update_layout(**PLOT_LAYOUT)
    fig.update_xaxes(gridcolor=BORDER, linecolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER, linecolor=BORDER)
    return fig


# ── Data & model loading (cached) ─────────────────────────────
@st.cache_resource(show_spinner="Loading data…")
def load_everything():
    ratings, movies, users = load_data()
    ratings, user2idx, movie2idx = remap_ids(ratings)
    idx2movie = {v: k for k, v in movie2idx.items()}

    N_USERS  = len(user2idx)
    N_MOVIES = len(movie2idx)

    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "two_tower.pt")
    faiss_path = model_path.replace(".pt", ".faiss")

    model, index = None, None
    if os.path.exists(model_path) and os.path.exists(faiss_path):
        model = TwoTowerModel(N_USERS, N_MOVIES, embedding_dim=128, output_dim=64)
        ckpt  = torch.load(model_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        index = faiss.read_index(faiss_path)

    user_seen = ratings.groupby("user_idx")["movie_idx"].apply(set).to_dict()

    results_path = os.path.join(os.path.dirname(__file__), "..", "results", "eval_results.csv")
    curves_path  = os.path.join(os.path.dirname(__file__), "..", "results", "training_curves.csv")
    eval_results  = pd.read_csv(results_path)  if os.path.exists(results_path)  else None
    train_curves  = pd.read_csv(curves_path)   if os.path.exists(curves_path)   else None

    return {
        "ratings": ratings, "movies": movies, "users": users,
        "user2idx": user2idx, "movie2idx": movie2idx, "idx2movie": idx2movie,
        "N_USERS": N_USERS, "N_MOVIES": N_MOVIES,
        "model": model, "index": index, "user_seen": user_seen,
        "eval_results": eval_results, "train_curves": train_curves,
    }

d = load_everything()

model_loaded = d["model"] is not None
if not model_loaded:
    st.warning("⚠️ Trained model not found. Run `python pipeline.py` first, then refresh. "
               "Dataset Explorer tab is still available.")


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎬 Movie RecSys")
    st.markdown(f"<span style='color:{SUBTEXT};font-size:0.8rem;'>Two-Tower · BPR · FAISS</span>", unsafe_allow_html=True)
    st.divider()
    tab = st.radio("Navigate", [
        "🎬 Recommendations",
        "🔍 Similar Movies",
        "📊 Model Comparison",
        "📈 Training Curves",
        "🗂️ Dataset Explorer",
    ])
    st.divider()
    c1, c2 = st.columns(2)
    c1.metric("Users",  f"{d['N_USERS']:,}")
    c2.metric("Movies", f"{d['N_MOVIES']:,}")
    st.metric("Ratings", f"{len(d['ratings']):,}")


# ══════════════════════════════════════════════════════════════
# TAB: RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════
if tab == "🎬 Recommendations":
    st.title("Personalised Recommendations")
    st.markdown(f"<span style='color:{SUBTEXT}'>Two-Tower retrieval via FAISS · cosine similarity in embedding space</span>", unsafe_allow_html=True)
    st.divider()

    if not model_loaded:
        st.error("Run `python pipeline.py` to train the model first.")
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            user_idx = st.slider("Select User", 0, d["N_USERS"] - 1, 42)
        with col2:
            n_recs = st.selectbox("Top-N", [5, 10, 20], index=1)

        user_seen = d["user_seen"].get(user_idx, set())

        import time
        t0 = time.perf_counter()
        u_vec = d["model"].user_tower(
            torch.tensor([user_idx], dtype=torch.long)
        ).detach().numpy().astype(np.float32)
        k_fetch = min(n_recs + len(user_seen) + 50, d["N_MOVIES"])
        scores, indices = d["index"].search(u_vec.reshape(1, -1), k_fetch)
        latency_ms = (time.perf_counter() - t0) * 1000

        recs = []
        for score, idx in zip(scores[0], indices[0]):
            if idx in user_seen:
                continue
            orig_id = d["idx2movie"].get(int(idx), -1)
            row = d["movies"][d["movies"]["movieId"] == orig_id]
            if not row.empty:
                recs.append({
                    "title":  row.iloc[0]["title"],
                    "genres": row.iloc[0]["genres"],
                    "score":  float(score),
                    "idx":    int(idx),
                })
            if len(recs) >= n_recs:
                break

        st.markdown(f"**User {user_idx}** · {len(user_seen)} movies seen · "
                    f"Retrieval latency: `{latency_ms:.1f}ms`")
        st.divider()

        for i, rec in enumerate(recs, 1):
            st.markdown(f"""
            <div class="rec-card">
              <span style="color:{SUBTEXT};font-size:0.78rem;">#{i}</span>
              <span class="rec-title">&nbsp; {rec['title']}</span>
              <span class="score-pill" style="float:right">{rec['score']:.3f}</span>
              <div class="rec-meta">{rec['genres']}</div>
            </div>
            """, unsafe_allow_html=True)

        # Show user history
        with st.expander(f"User {user_idx}'s watch history ({len(user_seen)} movies)"):
            history_ids = [d["idx2movie"].get(idx, -1) for idx in list(user_seen)[:20]]
            hist = d["movies"][d["movies"]["movieId"].isin(history_ids)][["title","genres"]]
            st.dataframe(hist, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# TAB: SIMILAR MOVIES
# ══════════════════════════════════════════════════════════════
elif tab == "🔍 Similar Movies":
    st.title("Similar Movies")
    st.markdown(f"<span style='color:{SUBTEXT}'>Item-item cosine similarity in the shared embedding space</span>", unsafe_allow_html=True)
    st.divider()

    if not model_loaded:
        st.error("Run `python pipeline.py` to train the model first.")
    else:
        movie_titles = d["movies"]["title"].tolist()
        selected_title = st.selectbox("Pick a movie", sorted(movie_titles))
        n_similar = st.slider("Number of similar movies", 5, 20, 10)

        selected_row = d["movies"][d["movies"]["title"] == selected_title].iloc[0]
        orig_id   = int(selected_row["movieId"])
        movie_idx = d["movie2idx"].get(orig_id)

        if movie_idx is not None:
            item_vec = d["model"].item_tower(
                torch.tensor([movie_idx], dtype=torch.long)
            ).detach().numpy().astype(np.float32)
            scores, indices = d["index"].search(item_vec.reshape(1, -1), n_similar + 1)

            st.markdown(f"**{selected_title}** · *{selected_row['genres']}*")
            st.divider()

            similar = []
            for score, idx in zip(scores[0], indices[0]):
                if int(idx) == movie_idx:
                    continue
                orig = d["idx2movie"].get(int(idx), -1)
                row  = d["movies"][d["movies"]["movieId"] == orig]
                if not row.empty:
                    similar.append({
                        "title":  row.iloc[0]["title"],
                        "genres": row.iloc[0]["genres"],
                        "score":  float(score),
                    })
                if len(similar) >= n_similar:
                    break

            for i, s in enumerate(similar, 1):
                st.markdown(f"""
                <div class="rec-card">
                  <span style="color:{SUBTEXT};font-size:0.78rem;">#{i}</span>
                  <span class="rec-title">&nbsp; {s['title']}</span>
                  <span class="score-pill" style="float:right">{s['score']:.3f}</span>
                  <div class="rec-meta">{s['genres']}</div>
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB: MODEL COMPARISON
# ══════════════════════════════════════════════════════════════
elif tab == "📊 Model Comparison":
    st.title("Model Comparison")
    st.markdown(f"<span style='color:{SUBTEXT}'>Popularity Baseline vs Matrix Factorization vs Two-Tower</span>", unsafe_allow_html=True)
    st.divider()

    if d["eval_results"] is None:
        st.warning("Run `python pipeline.py` first to generate evaluation results.")
    else:
        res = d["eval_results"]
        model_names = {"popularity": "Popularity", "matrix_factorization": "Matrix Factorization", "two_tower": "Two-Tower"}
        res["Model"] = res["model"].map(model_names)

        for metric, label in [("ndcg","NDCG@k"), ("recall","Recall@k"), ("hit_rate","Hit Rate@k")]:
            st.markdown(f"#### {label}")
            fig = px.line(res, x="k", y=metric, color="Model", markers=True,
                          color_discrete_map={"Popularity": ORANGE,
                                              "Matrix Factorization": BLUE,
                                              "Two-Tower": GREEN})
            st.plotly_chart(styled_fig(fig), width="stretch")

        st.markdown("#### Full Results Table (k=10)")
        k10 = res[res["k"] == 10][["Model","precision","recall","ndcg","hit_rate","coverage","novelty"]]
        st.dataframe(k10.set_index("Model"), use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB: TRAINING CURVES
# ══════════════════════════════════════════════════════════════
elif tab == "📈 Training Curves":
    st.title("Training Curves")
    st.markdown(f"<span style='color:{SUBTEXT}'>Loss convergence for MF (MSE) and Two-Tower (BPR)</span>", unsafe_allow_html=True)
    st.divider()

    if d["train_curves"] is None:
        st.warning("Run `python pipeline.py` first to generate training curves.")
    else:
        curves = d["train_curves"]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Matrix Factorization — MSE Loss")
            fig = px.line(curves, x="epoch", y="mf_loss", markers=True,
                          color_discrete_sequence=[BLUE],
                          labels={"mf_loss": "MSE Loss", "epoch": "Epoch"})
            st.plotly_chart(styled_fig(fig), width="stretch")

        with col2:
            st.markdown("#### Two-Tower — BPR Loss")
            fig2 = px.line(curves, x="epoch", y="tt_loss", markers=True,
                           color_discrete_sequence=[GREEN],
                           labels={"tt_loss": "BPR Loss", "epoch": "Epoch"})
            st.plotly_chart(styled_fig(fig2), width="stretch")

        st.info("BPR loss (Bayesian Personalised Ranking): "
                "minimised when the score of the positive item exceeds the negative item. "
                "Converges to ~0.23 indicating the model has learned to rank positives above negatives.")


# ══════════════════════════════════════════════════════════════
# TAB: DATASET EXPLORER
# ══════════════════════════════════════════════════════════════
elif tab == "🗂️ Dataset Explorer":
    st.title("Dataset Explorer")
    st.markdown(f"<span style='color:{SUBTEXT}'>MovieLens-style synthetic data · 850k+ ratings</span>", unsafe_allow_html=True)
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Users",   f"{d['N_USERS']:,}")
    c2.metric("Movies",  f"{d['N_MOVIES']:,}")
    c3.metric("Ratings", f"{len(d['ratings']):,}")
    c4.metric("Avg Rating", f"{d['ratings']['rating'].mean():.2f}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Rating Distribution")
        dist = d["ratings"]["rating"].value_counts().sort_index().reset_index()
        dist.columns = ["rating", "count"]
        fig = px.bar(dist, x="rating", y="count", color_discrete_sequence=[BLUE])
        st.plotly_chart(styled_fig(fig), width="stretch")

    with col2:
        st.markdown("#### Ratings per User (distribution)")
        user_counts = d["ratings"].groupby("user_idx").size().reset_index(name="n_ratings")
        fig2 = px.histogram(user_counts, x="n_ratings", nbins=50,
                            color_discrete_sequence=[GREEN],
                            labels={"n_ratings": "Ratings per User"})
        st.plotly_chart(styled_fig(fig2), width="stretch")

    st.markdown("#### Movie Popularity (log scale) — Power Law Distribution")
    movie_pop = d["ratings"].groupby("movie_idx").size().sort_values(ascending=False).reset_index()
    movie_pop.columns = ["movie_idx", "n_ratings"]
    movie_pop["rank"] = range(1, len(movie_pop) + 1)
    fig3 = px.line(movie_pop.head(500), x="rank", y="n_ratings",
                   color_discrete_sequence=[ORANGE],
                   labels={"rank": "Movie Rank", "n_ratings": "Number of Ratings"},
                   log_y=True)
    st.plotly_chart(styled_fig(fig3), width="stretch")

    if "genres" in d["movies"].columns:
        st.markdown("#### Genre Distribution")
        all_genres = d["movies"]["genres"].str.split("|").explode()
        genre_counts = all_genres.value_counts().reset_index()
        genre_counts.columns = ["genre", "count"]
        fig4 = px.bar(genre_counts, x="count", y="genre", orientation="h",
                      color_discrete_sequence=[BLUE])
        fig4.update_yaxes(autorange="reversed")
        st.plotly_chart(styled_fig(fig4), width="stretch")
