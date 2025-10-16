import streamlit as st
import pandas as pd
import os
import model
from model import load_data, preprocess, train_model, load_model, save_meta, get_recommendations_from_titles, MODEL_PATH, META_PATH

# ----------------- Page Setup -----------------
st.set_page_config(page_title="📚 Book Recommendation System", layout="wide")
st.title("📚 Book Recommendation System (Item-Item Cosine)")

# ----------------- Sidebar -----------------
st.sidebar.header("⚙️ Settings")
ratings_path = st.sidebar.text_input("Ratings CSV path", value="Dataset/ratings.csv")
books_path = st.sidebar.text_input("Books CSV path", value="Dataset/books.csv")
top_n = st.sidebar.slider("Number of recommendations", min_value=3, max_value=20, value=8)
min_user_ratings = st.sidebar.number_input("Min ratings per user", min_value=1, value=1, step=1)
min_book_ratings = st.sidebar.number_input("Min ratings per book", min_value=1, value=1, step=1)
retrain_button = st.sidebar.button("🔄 Retrain model")

# ----------------- Data Loading -----------------
@st.cache_data(show_spinner=True)
def load_and_prepare(ratings_path, books_path, min_user_ratings, min_book_ratings):
    ratings, books = load_data(ratings_path, books_path)
    merged, user_item, books_meta = preprocess(
        ratings, books,
        min_user_ratings=int(min_user_ratings),
        min_book_ratings=int(min_book_ratings)
    )
    return ratings, books, merged, user_item, books_meta

# check if files exist before loading
if not os.path.exists(ratings_path):
    st.error(f"Ratings file not found: `{ratings_path}`")
    st.stop()

if not os.path.exists(books_path):
    st.error(f"Books file not found: `{books_path}`")
    st.stop()

try:
    ratings, books, merged, user_item, books_meta = load_and_prepare(
        ratings_path, books_path, min_user_ratings, min_book_ratings
    )
    st.write("### Preview of Ratings Data", ratings.head())
    st.write("### Preview of Books Data", books.head())
except Exception as e:
    st.error(f" Error loading data: {e}")
    st.stop()

# ----------------- Model Loading/Training -----------------
similarity_df, saved_meta = load_model(MODEL_PATH, META_PATH)

if similarity_df is None or retrain_button:
    with st.spinner("Training similarity model... Please wait ⏳"):
        similarity_df = train_model(user_item, save_path=MODEL_PATH)
        save_meta(books_meta, META_PATH)
        saved_meta = books_meta
    st.success("✅ Model trained and saved successfully!")
else:
    if saved_meta is None:
        saved_meta = books_meta

# ----------------- User Input -----------------
st.markdown("### ✍️ Enter book titles you like")
st.markdown("Separate multiple titles with **`;`** or **`,`**")
user_input = st.text_input("Example: The Hobbit; Dune; Harry Potter")
get_button = st.button("🚀 Get Recommendations")

# ----------------- Recommendation Output -----------------
if get_button:
    if not user_input.strip():
        st.warning("⚠️ Please enter at least one book title.")
    else:
        titles = [t.strip() for t in user_input.replace(";", ",").split(",") if t.strip()]
        try:
            recs, suggestions = get_recommendations_from_titles(
                titles, similarity_df, saved_meta, top_n=top_n
            )

            if recs.empty:
                st.info("ℹ️ No exact matches found for your titles.")
                if suggestions:
                    st.write("🔍 Did you mean:")
                    for q, s in suggestions.items():
                        st.write(f"- **{q}** → {s if s else 'No close match'}")
                else:
                    st.write("Try checking spelling or another title.")
            else:
                st.success(f"🎯 Top {len(recs)} Recommendations")
                for _, row in recs.iterrows():
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        if pd.notna(row.get("image_url")) and str(row.get("image_url")).strip():
                            st.image(row["image_url"], width=100)
                        else:
                            st.write("📕 No image")
                    with col2:
                        st.markdown(f"**{row['title']}**")
                        if pd.notna(row.get("authors")):
                            st.markdown(f"👤 {row['authors']}")
                        if pd.notna(row.get("average_rating")):
                            st.markdown(f"⭐ Avg Rating: {row['average_rating']}")
                        st.markdown(f"🔗 Similarity Score: `{row['similarity_score']:.4f}`")
                        st.markdown("---")

        except Exception as e:
            st.error(f" Error generating recommendations: {e}")
