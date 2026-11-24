
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---------------------------------
# PAGE CONFIGURATION
# ---------------------------------

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# ---------------------------------
# PROJECT PATHS
# ---------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "processed"

# ---------------------------------
# LOAD SAVED FILES
# ---------------------------------

@st.cache_resource
def load_data():

    movies = pd.read_csv(
        DATA_DIR / "movies_processed.csv"
    )

    tfidf_matrix = joblib.load(
        MODEL_DIR / "tfidf_matrix.pkl"
    )

    return movies, tfidf_matrix


movies, tfidf_matrix = load_data()

# ---------------------------------
# RECOMMENDATION FUNCTION
# ---------------------------------

def recommend_movies(movie_name, top_n=10):

    matches = movies[
        movies["title"].str.lower() == movie_name.lower()
    ]

    if matches.empty:
        return pd.DataFrame()

    movie_index = matches.index[0]

    # Calculate similarity against selected movie
    scores = tfidf_matrix @ tfidf_matrix[movie_index].T

    scores = scores.toarray().flatten()

    # Sort by similarity
    similar_indices = scores.argsort()[::-1]

    recommendations = []

    for index in similar_indices:

        if index == movie_index:
            continue

        recommendations.append({
            "Movie": movies.iloc[index]["title"],
            "Similarity Score": round(
                float(scores[index]), 4
            )
        })

        if len(recommendations) == top_n:
            break

    return pd.DataFrame(recommendations)


# ---------------------------------
# APPLICATION INTERFACE
# ---------------------------------

st.title("🎬 Movie Recommendation System")

st.write(
    "Discover movies similar to your favourite films "
    "using Content-Based Machine Learning."
)

st.divider()

# Movie selection
movie_list = sorted(movies["title"].dropna().unique())

selected_movie = st.selectbox(
    "Choose your favourite movie",
    movie_list
)

# Recommendation count
top_n = st.slider(
    "Number of recommendations",
    min_value=5,
    max_value=20,
    value=10
)

# Recommendation button
if st.button("Recommend Movies", type="primary"):

    recommendations = recommend_movies(
        selected_movie,
        top_n
    )

    st.subheader(
        f"Movies similar to {selected_movie}"
    )

    if recommendations.empty:

        st.warning("No recommendations found.")

    else:

        st.dataframe(
            recommendations,
            use_container_width=True,
            hide_index=True
        )

        st.success(
            "Movie recommendations generated successfully!"
        )

st.divider()

st.caption(
    "Powered by Python | TF-IDF | Cosine Similarity | Streamlit"
)
