import os
import pickle
import pandas as pd
import streamlit as st
from hybrid_model import HybridRecommendationSystem_Custom   # ✅ your model class

# ----------------- Setup -----------------
DATA_DIR = os.path.dirname(__file__)

# Load data
ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))
books = pd.read_csv(os.path.join(DATA_DIR, "books.csv"))

# Load saved model
with open(os.path.join(DATA_DIR, "hybrid_model.pkl"), "rb") as f:
    hybrid_system = pickle.load(f)

# ----------------- Helpers -----------------
if "recs" not in st.session_state:
    st.session_state.recs = None

def fetch_recommendations(hybrid_system, user_id, books_df, top_n=10):
    """
    Returns a DataFrame with: book_id, title, authors, score (if available).
    Works with either .get_hybrid_recommendations() or .recommend().
    """
    # Case 1: Your custom method
    if hasattr(hybrid_system, "get_hybrid_recommendations"):
        df = hybrid_system.get_hybrid_recommendations(user_id, top_n=top_n)
        if "hybrid_score" in df.columns and "score" not in df.columns:
            df = df.rename(columns={"hybrid_score": "score"})
        return df

    # Case 2: Generic recommend()
    if hasattr(hybrid_system, "recommend"):
        items = books_df['book_id'].tolist()  # use book_id consistently
        recommendations = hybrid_system.recommend(user_id, items, top_n=top_n)

        if isinstance(recommendations, list) and len(recommendations) > 0:
            rows = [{"book_id": int(bid), "score": float(scr)} for bid, scr in recommendations]
            df = pd.DataFrame(rows)
            return df.merge(books_df[["book_id", "title", "authors"]], on="book_id", how="left")
        return pd.DataFrame()

    st.error("Model has neither 'get_hybrid_recommendations' nor 'recommend'.")
    return pd.DataFrame()

# ----------------- UI -----------------
st.title("📚 Hybrid Book Recommendation System")

user_id = st.sidebar.selectbox("Select User ID:", ratings["user_id"].unique())
top_n = st.sidebar.slider("Number of recommendations:", 5, 20, 10)

if st.sidebar.button("Recommend"):
    st.session_state.recs = fetch_recommendations(hybrid_system, user_id, books, top_n=top_n)

# ----------------- Display -----------------
recs = st.session_state.recs

if recs is None:
    st.info("Pick a user and click **Recommend** to see suggestions.")
elif recs.empty:
    st.warning("No model recommendations found. Showing popular books instead.")
    top_books = (
        ratings.groupby("book_id")["rating"].mean().sort_values(ascending=False).head(top_n).index.tolist()
    )
    fallback = books[books["book_id"].isin(top_books)][["book_id", "title", "authors"]]
    for i, row in enumerate(fallback.itertuples(), 1):
        st.write(f"{i}. 📖 {row.title} — {getattr(row, 'authors', 'Unknown')}")
else:
    st.subheader(f"Top {min(top_n, len(recs))} recommendations for user {user_id}")
    for i, row in enumerate(recs.itertuples(), 1):
        title = getattr(row, "title", f"Book {getattr(row, 'book_id', 'N/A')}")
        authors = getattr(row, "authors", "Unknown")
        score = getattr(row, "score", getattr(row, "cf_score", getattr(row, "content_score", None)))
        if score is not None:
            st.write(f"{i}. 📖 **{title}** — {authors} _(score: {score:.3f})_")
        else:
            st.write(f"{i}. 📖 **{title}** — {authors}")

# ----------------- Feedback -----------------
st.markdown("---")
st.subheader("💡 Feedback")

if recs is not None and not recs.empty:
    feedback_options = recs["book_id"].tolist()
else:
    feedback_options = books["book_id"].head(50).tolist()

col1, col2 = st.columns(2)
with col1:
    book_id_fb = st.selectbox("Pick a book to rate:", feedback_options, key="fb_book")
with col2:
    fb_choice = st.radio("Did you like this recommendation?", ["👍 Yes", "👎 No"], horizontal=True, key="fb_like")

if st.button("Submit Feedback"):
    rating_val = 5 if fb_choice == "👍 Yes" else 1
    pd.DataFrame([[user_id, book_id_fb, rating_val]], columns=["user_id", "book_id", "rating"])\
        .to_csv(os.path.join(DATA_DIR, "ratings.csv"), mode="a", header=False, index=False)
    st.success(f"Saved feedback for user {user_id}, book {book_id_fb}: {rating_val}⭐")
