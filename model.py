import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches
import mlflow

MODEL_PATH = "similarity_model.pkl"
META_PATH = "books_meta.pkl"

def _check_required_columns(df, column_groups, filename: str):
    
    for possible_names, display_name in column_groups:
        if not any(col in df.columns for col in possible_names):
            raise ValueError(f"{filename} must contain a column for {display_name}")


def _standardize_columns(df, mapping):
    df = df.copy()
    df_cols = {c.lower(): c for c in df.columns}
    rename_map = {}
    for std_name, candidates in mapping.items():
        for cand in candidates:
            if cand in df_cols:
                rename_map[df_cols[cand]] = std_name
                break
    return df.rename(columns=rename_map)

def load_data(ratings_path="Dataset/ratings.csv", books_path="Dataset/books.csv"):
    """
    Loads ratings and books CSVs.
    Tries common column-name fallbacks for merging.
    """
    ratings = pd.read_csv(ratings_path)
    books = pd.read_csv(books_path)

    # Ensure expected columns exist
    _check_required_columns(
        ratings,
        [
            (("user_id", "userid", "user"), "user id"),
            (("book_id", "bookid", "book"), "book id"),
            (("rating", "ratings", "score"), "rating"),
        ],
        "ratings.csv"
    )

    ratings = _standardize_columns(ratings, {
        "user_id": ["user_id", "userid", "user"],
        "book_id": ["book_id", "bookid", "book"],
        "rating": ["rating", "ratings", "score"]
    })

    books = _standardize_columns(books, {
        "book_id": ["book_id", "bookid", "book", "id", "goodreads_book_id"],
        "title": ["title", "original_title", "book_title"],
        "authors": ["authors", "author", "book_authors", "author_name"],
        "image_url": ["image_url", "image", "small_image_url"],
        "average_rating": ["average_rating", "avg_rating", "avg"]
    })

    # Ensure required columns exist now
    if "book_id" not in ratings.columns or "user_id" not in ratings.columns or "rating" not in ratings.columns:
        raise ValueError("ratings.csv missing required columns after standardization.")
    if "book_id" not in books.columns or "title" not in books.columns:
        raise ValueError("books.csv missing required columns after standardization.")

    return ratings, books

def preprocess(ratings, books, min_user_ratings=1, min_book_ratings=1):
    """
    Merge ratings and books, drop missing, encode IDs, filter low-activity users/books.
    Returns merged_df, user_item_matrix, books_meta_df
    """
    # Merge on book_id
    merged = ratings.merge(books, on="book_id", how="inner")
    # Drop rows with missing essential values
    merged = merged.dropna(subset=["user_id", "book_id", "rating", "title"])

    # Optionally filter users/books with very few ratings to reduce sparsity
    if min_user_ratings > 1:
        user_counts = merged["user_id"].value_counts()
        keep_users = user_counts[user_counts >= min_user_ratings].index
        merged = merged[merged["user_id"].isin(keep_users)]
    if min_book_ratings > 1:
        book_counts = merged["book_id"].value_counts()
        keep_books = book_counts[book_counts >= min_book_ratings].index
        merged = merged[merged["book_id"].isin(keep_books)]

    # Encode user_id and book_id to integer indices
    merged = merged.copy()
    merged["user_enc"], _ = pd.factorize(merged["user_id"])
    merged["book_enc"], _ = pd.factorize(merged["book_id"])

    # Build user-item matrix: rows = users, columns = books (use title as column for readability)

    merged["title_clean"] = merged["title"].astype(str)
    user_item = merged.pivot_table(
    index="user_enc", columns="title_clean", values="rating", aggfunc="mean"
    ) 

    # Prepare books metadata (one row per title)
    books_meta = merged[["title_clean", "book_id", "authors", "image_url", "average_rating"]].drop_duplicates("title_clean").set_index("title_clean")

    return merged, user_item, books_meta

def train_model(user_item, save_path=MODEL_PATH):
    
    # Transpose: columns are book titles (user_item already columns=title). Compute cosine on columns
    item_matrix = user_item.values.T  # shape (n_books, n_users)
    
    # Compute cosine similarity (n_books x n_books)
    sim = cosine_similarity(item_matrix)
    
    titles = user_item.columns.tolist()
    similarity_df = pd.DataFrame(sim, index=titles, columns=titles)

    # Try MLflow logging (safe: won't crash if mlflow not configured)
    try:
        mlflow.start_run()
        mlflow.log_param("model_type", "item_item_cosine")
        mlflow.log_param("n_books", len(titles))
        mlflow.log_param("n_users", user_item.shape[0])
        mlflow.log_metric("sample_mean_similarity", float(similarity_df.to_numpy().mean()))
        mlflow.end_run()
    except Exception:
        # silently ignore mlflow problems
        pass

    # Save similarity matrix
    joblib.dump(similarity_df, save_path)
    return similarity_df

def load_model(similarity_path=MODEL_PATH, meta_path=META_PATH):
    
    if not os.path.exists(similarity_path):
        return None, None
    similarity_df = joblib.load(similarity_path)
    if os.path.exists(meta_path):
        books_meta = joblib.load(meta_path)
    else:
        books_meta = None
    return similarity_df, books_meta

def save_meta(books_meta, meta_path=META_PATH):
    joblib.dump(books_meta, meta_path)

def find_best_title_match(query_title, titles_index, n_matches=3, cutoff=0.6):
    
    if not query_title or not isinstance(query_title, str):
        return None, []

    q = query_title.strip().lower()
    # exact match (case-insensitive)
    for t in titles_index:
        if t.strip().lower() == q:
            return t, []

    # fuzzy suggestions using difflib
    candidates = list(titles_index)
    matches = get_close_matches(query_title, candidates, n=n_matches, cutoff=cutoff)
    return (matches[0] if matches else None), matches

def get_recommendations_from_titles(input_titles, similarity_df, books_meta, top_n=10):
 
    if similarity_df is None:
        raise ValueError("Model not loaded. Train or load a model first.")

    # validate titles and compute aggregated similarity
    all_titles = similarity_df.index.tolist()
    found_titles = []
    suggestions = {}

    for t in input_titles:
        matched, sugg = find_best_title_match(t, all_titles, n_matches=5)
        if matched:
            found_titles.append(matched)
        else:
            suggestions[t] = sugg

    if not found_titles:
        return pd.DataFrame(), suggestions

    # Aggregate similarity: average similarity vector across liked books
    sim_sub = similarity_df.loc[found_titles]  # rows = liked books
    
    # mean across rows -> average similarity of every book to the liked set
    agg_scores = sim_sub.mean(axis=0)
    # remove the liked books themselves
    agg_scores = agg_scores.drop(index=found_titles, errors="ignore")
    top_scores = agg_scores.sort_values(ascending=False).head(top_n)

    # Build result DataFrame with metadata
    recs = pd.DataFrame({
        "title": top_scores.index,
        "similarity_score": top_scores.values
    }).set_index("title")

    # Join metadata if available
    if books_meta is not None:
        recs = recs.join(books_meta, how="left", on="title", validate="one_to_one")
    else:
        recs["authors"] = np.nan
        recs["image_url"] = np.nan
        recs["average_rating"] = np.nan

    recs = recs.reset_index()
    return recs, suggestions
