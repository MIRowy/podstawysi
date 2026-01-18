import os
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, abort

app = Flask(__name__)

# ====== ŚCIEŻKI (PythonAnywhere: najlepiej absolutne) ======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "saved_model.keras")  # folder z model.save()
PREPARED_DIR = os.path.join(BASE_DIR, "prepared")             # train/val/test z pkt 2
MOVIES_CSV = os.path.join(BASE_DIR, "movies.csv")             # MovieLens movies.csv

# ====== ŁADOWANIE ARTEFAKTÓW (robimy raz) ======
model = tf.keras.models.load_model(MODEL_DIR)

train_df = pd.read_csv(os.path.join(PREPARED_DIR, "train.csv"))
val_df   = pd.read_csv(os.path.join(PREPARED_DIR, "val.csv"))
test_df  = pd.read_csv(os.path.join(PREPARED_DIR, "test.csv"))

movies = pd.read_csv(MOVIES_CSV)

# movieId -> title
movieId_to_title = dict(zip(movies["movieId"], movies["title"]))

# mapowania z danych (nie potrzebujesz LabelEncoderów)
all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

# userId -> user_enc
userId_to_enc = (
    all_df[["userId", "user_enc"]]
    .drop_duplicates(subset=["userId"])
    .set_index("userId")["user_enc"]
)

# movie_enc -> movieId
enc_to_movieId = (
    all_df[["movie_enc", "movieId"]]
    .drop_duplicates(subset=["movie_enc"])
    .set_index("movie_enc")["movieId"]
)

# liczności
n_movies = int(all_df["movie_enc"].nunique())

# do “nie polecaj tego co już oceniał”
seen_df = pd.concat([train_df, val_df], ignore_index=True)


def recommend_top_k_by_userId(userId: int, k: int = 10) -> pd.DataFrame:
    if userId not in userId_to_enc.index:
        raise KeyError(f"Nieznany userId={userId} (brak w danych po filtracji).")

    user_enc = int(userId_to_enc.loc[userId])

    seen_movies = set(seen_df.loc[seen_df["user_enc"] == user_enc, "movie_enc"].tolist())
    candidates = np.array([m for m in range(n_movies) if m not in seen_movies], dtype=np.int32)

    if len(candidates) == 0:
        return pd.DataFrame(columns=["title", "pred_rating", "movieId", "movie_enc"])

    user_arr = np.full(shape=(len(candidates),), fill_value=user_enc, dtype=np.int32)

    preds = model.predict(
        {"user": user_arr, "movie": candidates},
        batch_size=4096,
        verbose=0
    ).reshape(-1)

    k_eff = min(k, len(preds))
    top_idx = np.argpartition(-preds, kth=k_eff - 1)[:k_eff]
    top_sorted = top_idx[np.argsort(-preds[top_idx])]

    recs = pd.DataFrame({
        "movie_enc": candidates[top_sorted],
        "pred_rating": preds[top_sorted],
    })

    recs["movieId"] = recs["movie_enc"].map(enc_to_movieId)
    recs["title"] = recs["movieId"].map(movieId_to_title).fillna("(brak tytułu)")

    recs = recs[["title", "pred_rating", "movieId", "movie_enc"]].reset_index(drop=True)
    return recs


@app.route("/")
def home():
    return """
    <h2>Movie Recommender</h2>
    <p>Wejdź na <code>/&lt;userId&gt;</code>, np. <a href="/1">/1</a></p>
    """


@app.route("/<int:userId>")
def by_user(userId: int):
    try:
        recs = recommend_top_k_by_userId(userId, k=10)
    except KeyError as e:
        abort(404, str(e))

    table_html = recs.to_html(index=False, escape=True)
    return f"""
    <h2>Rekomendacje dla userId={userId}</h2>
    {table_html}
    <p><a href="/">Powrót</a></p>
    """