import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------
# 1. Load Dataset
# Dataset must have: title, overview, poster_path, imdb_id
# poster_path is TMDB poster path, imdb_id is IMDb code
# ---------------------------------------------------------
@st.cache_data
def load_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    movies = movies[['title', 'overview', 'poster_path', 'imdb_id']]
    movies.dropna(inplace=True)
    return movies

movies = load_data()

# ---------------------------------------------------------
# 2. TF-IDF Vectorization & Cosine Similarity
# ---------------------------------------------------------
@st.cache_data
def compute_similarity(movies):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['overview'])
    return cosine_similarity(tfidf_matrix)

similarity = compute_similarity(movies)

# ---------------------------------------------------------
# 3. Recommendation Function
# ---------------------------------------------------------
def recommend(movie_name):
    movie_name = movie_name.strip()
    if movie_name not in movies['title'].values:
        return []

    idx = movies[movies['title'] == movie_name].index[0]
    distances = similarity[idx]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]  # skip the selected movie

    recommendations = []
    for i in movie_list:
        movie = movies.iloc[i[0]]
        poster_url = f"https://image.tmdb.org/t/p/w200{movie['poster_path']}"
        imdb_url = f"https://www.imdb.com/title/{movie['imdb_id']}/"
        recommendations.append({
            "title": movie['title'],
            "poster": poster_url,
            "imdb": imdb_url
        })
    return recommendations

# ---------------------------------------------------------
# 4. Streamlit UI
# ---------------------------------------------------------
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommendation System with Posters")

movie_name = st.selectbox("Choose a movie", movies['title'].values)

if st.button("Recommend Movies"):
    recs = recommend(movie_name)
    if not recs:
        st.write("Movie not found.")
    else:
        st.subheader("Recommended Movies:")
        for r in recs:
            st.markdown(f"**[{r['title']}]({r['imdb']})**")
            st.image(r['poster'], width=150)
