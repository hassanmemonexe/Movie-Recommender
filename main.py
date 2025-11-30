import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
API_KEY = "c07dc22b62ee42838a91ed1b93f8b2ae" 

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
@st.cache_data
def load_data():
    # Load the CSV file
    movies = pd.read_csv("tmdb_5000_movies.csv")
    
    # Keep only necessary columns
    movies = movies[['id', 'title', 'overview']] 
    
    # Drop missing values and reset index to match matrix rows
    movies.dropna(inplace=True)
    movies.reset_index(drop=True, inplace=True)
    
    return movies

movies = load_data()

# ---------------------------------------------------------
# 2. VECTORIZATION (THE BRAIN)
# ---------------------------------------------------------
@st.cache_data
def compute_tfidf(movies):
    # Convert text to numerical vectors, ignoring common English words
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['overview'])
    
    # Calculate cosine similarity between all vectors
    similarity = cosine_similarity(tfidf_matrix)
    return similarity

similarity = compute_tfidf(movies)

# ---------------------------------------------------------
# 3. HELPER: FETCH POSTERS FROM API
# ---------------------------------------------------------
def fetch_details(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
        data = requests.get(url).json()
        
        # Get Poster URL
        poster_path = data.get('poster_path')
        full_path = "https://via.placeholder.com/500x750?text=No+Image"
        if poster_path:
            full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
            
        # Get IMDb URL
        imdb_id = data.get('imdb_id')
        imdb_url = "https://www.imdb.com/"
        if imdb_id:
            imdb_url = f"https://www.imdb.com/title/{imdb_id}/"

        return full_path, imdb_url
    except:
        return "https://via.placeholder.com/500x750?text=Error", "https://www.imdb.com/"

# ---------------------------------------------------------
# 4. RECOMMENDATION LOGIC
# ---------------------------------------------------------
def recommend(movie_name):
    # Find the index of the movie in the dataframe
    if movie_name not in movies['title'].values:
        return [], [], []

    index = movies[movies['title'] == movie_name].index[0]
    
    # Get similarity scores for this movie
    distances = similarity[index]

    # Sort movies based on similarity score (highest first)
    # enumerate keeps track of the original index (movie ID)
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    rec_names = []
    rec_posters = []
    rec_links = []

    for i in movie_list:
        movie_id = movies.iloc[i[0]].id
        
        # Fetch external data via API
        poster, link = fetch_details(movie_id)
        
        rec_names.append(movies.iloc[i[0]].title)
        rec_posters.append(poster)
        rec_links.append(link)

    return rec_names, rec_posters, rec_links

# ---------------------------------------------------------
# 5. UI (STREAMLIT)
# ---------------------------------------------------------
st.set_page_config(page_title="Movie Recommender", layout="wide")

st.title("🎬 Movie Recommendation System")
st.write("Select a movie to see similar recommendations.")

selected_movie = st.selectbox("Type or select a movie from the dropdown", movies['title'].values)

if st.button("Recommend Movies"):
    names, posters, links = recommend(selected_movie)
    
    st.subheader(f"Because you liked '{selected_movie}':")
    
    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            # HTML to make the image clickable
            html_code = f"""
                <a href="{links[i]}" target="_blank">
                    <img src="{posters[i]}" style="width:100%; border-radius:10px; margin-bottom: 10px;">
                </a>
                <div style="text-align:center;"><b>{names[i]}</b></div>
            """
            st.markdown(html_code, unsafe_allow_html=True)
