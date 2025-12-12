import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
API_KEY = "c07dc22b62ee42838a91ed1b93f8b2ae" # TMDB API Key

# ---------------------------------------------------------
# 1. DATA LOADING & PREPROCESSING
# ---------------------------------------------------------
@st.cache_data
def load_data():
    # Load movies dataset
    movies = pd.read_csv("tmdb_5000_movies.csv")
    
    all_tags = []
    
    # Process each movie to create a unified 'tags' feature
    for i in range(len(movies)):
        
        # Parse stringified lists (e.g., "['Action', 'Comedy']")
        genres_list = eval(movies['genres'][i]) 
        keywords_list = eval(movies['keywords'][i])
        
        # Handle missing overview
        overview = str(movies['overview'][i])
        if overview == "nan":
            overview = ""
        
        # Extract genre names
        my_genres = []
        for item in genres_list:
            # Replace spaces with underscores for single-token handling
            name = item['name'].replace(" ", "_")
            my_genres.append(name)
            
        # Extract keyword names
        my_keywords = []
        for item in keywords_list:
            name = item['name'].replace(" ", "_")
            my_keywords.append(name)
        
        # Create tag string
        genre_str = " ".join(my_genres)
        keyword_str = " ".join(my_keywords)
        full_tag = overview + " " + genre_str + " " + keyword_str
        
        all_tags.append(full_tag)

    movies['tags'] = all_tags
    
    # Select relevant columns and clean missing values
    movies = movies[['id', 'title', 'tags']]
    movies.dropna(inplace=True)
    movies.reset_index(drop=True, inplace=True)
    
    return movies

movies = load_data()

# ---------------------------------------------------------
# 2. FEATURE EXTRACTION & SIMILARITY
# ---------------------------------------------------------
@st.cache_data
def compute_similarity(movies):
    # Initialize TF-IDF Vectorizer (remove English stop words)
    tfidf = TfidfVectorizer(stop_words='english')
    
    # Generate TF-IDF matrix from 'tags'
    matrix = tfidf.fit_transform(movies['tags'])
    
    # Compute Cosine Similarity matrix
    similarity = cosine_similarity(matrix)
    return similarity

similarity = compute_similarity(movies)

# ---------------------------------------------------------
# 3. API UTILITIES
# ---------------------------------------------------------
def fetch_details(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
    
    try:
        data = requests.get(url).json()
        
        # Fetch poster path
        poster_path = data.get('poster_path')
        if poster_path:
            full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
        else:
            full_path = "https://via.placeholder.com/500x750?text=No+Image"
            
        # Fetch IMDb ID
        imdb_id = data.get('imdb_id')
        if imdb_id:
            imdb_link = f"https://www.imdb.com/title/{imdb_id}/"
        else:
            imdb_link = "https://www.imdb.com/"
            
        return full_path, imdb_link
        
    except:
        return "https://via.placeholder.com/500x750?text=Error", "https://www.imdb.com/"

# ---------------------------------------------------------
# 4. RECOMMENDATION ENGINE
# ---------------------------------------------------------
def get_score(pair):
    return pair[1]

def recommend(movie_name):
    # Find index of selected movie
    movie_index = -1
    for i in range(len(movies)):
        if movies['title'][i].lower() == movie_name.lower():
            movie_index = i
            break
            
    if movie_index == -1:
        return [], [], []

    # Retrieve similarity scores
    scores = similarity[movie_index]
    
    # Create list of (index, score) tuples
    pairs = list(enumerate(scores))
    
    # Sort by similarity score in descending order
    sorted_pairs = sorted(pairs, key=get_score, reverse=True)
    
    # Select top 5 recommendations (exclude self)
    top_5_pairs = sorted_pairs[1:6]
    
    rec_names = []
    rec_posters = []
    rec_links = []
    
    for pair in top_5_pairs:
        index = pair[0]
        movie_id = movies.iloc[index].id
        title = movies.iloc[index].title
        
        poster, link = fetch_details(movie_id)
        
        rec_names.append(title)
        rec_posters.append(poster)
        rec_links.append(link)
        
    return rec_names, rec_posters, rec_links

# ---------------------------------------------------------
# 5. UI INTERFACE
# ---------------------------------------------------------
st.title("🎬 Movie Recommender")
st.write("Browse movies based on plot, genres, and keywords.")

selected_movie = st.selectbox("Pick a movie:", movies['title'].values)

if st.button("Recommend"):
    names, posters, links = recommend(selected_movie)
    
    if names:
        st.header("Top 5 Recommendations:")
        
        cols = st.columns(5)
        
        for i in range(5):
            with cols[i]:
                st.image(posters[i], use_container_width=True)
                st.write(f"**{names[i]}**")
                st.markdown(f"[View on IMDb]({links[i]})")
