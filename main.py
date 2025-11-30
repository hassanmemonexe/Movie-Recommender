import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------
# 1. Load Dataset
# ---------------------------------------------------------
@st.cache_data
def load_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")  # Ensure movies.csv is in same folder
    movies = movies[['title', 'overview']]
    movies.dropna(inplace=True)
    return movies

movies = load_data()

# ---------------------------------------------------------
# 2. TF-IDF Vectorization
# ---------------------------------------------------------
@st.cache_data
def compute_tfidf(movies):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['overview'])
    similarity = cosine_similarity(tfidf_matrix)
    return similarity

similarity = compute_tfidf(movies)

# ---------------------------------------------------------
# 3. Recommendation Function
# ---------------------------------------------------------
def recommend(movie_name):
    movie_name = movie_name.strip()

    if movie_name not in movies['title'].values:
        return ["Movie not found in dataset"]

    index = movies[movies['title'] == movie_name].index[0]
    distances = similarity[index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]  # skip the selected movie itself

    recommendations = [movies.iloc[i[0]].title for i in movie_list]
    return recommendations

# ---------------------------------------------------------
# 4. Streamlit UI
# ---------------------------------------------------------
st.set_page_config(page_title="Movie Recommendation System", layout="centered")
st.title("🎬 Movie Recommendation System")
st.write("Select a movie and get recommendations based on similarity.")

movie_name = st.selectbox(
    "Choose a movie",
    movies['title'].values
)

if st.button("Recommend Movies"):
    st.subheader("Recommended Movies:")
    recommendations = recommend(movie_name)
    for m in recommendations:
        st.write("✔️ " + m)
