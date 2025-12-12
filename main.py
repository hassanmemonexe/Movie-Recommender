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
# 1. LOAD AND CLEAN DATA (Using Basic Loops)
# ---------------------------------------------------------
@st.cache_data
def load_data():
    # Load the raw data
    movies = pd.read_csv("tmdb_5000_movies.csv")
    
    # We need to create a new list of 'tags' for every movie
    # We will loop through the dataframe row by row
    all_tags = []
    
    for i in range(len(movies)):
        # 1. Get the raw strings for this movie
        # 'eval()' turns the string "['Action', 'Adventure']" into a real Python list
        genres_list = eval(movies['genres'][i]) 
        keywords_list = eval(movies['keywords'][i])
        cast_list = eval(movies['cast'][i])
        overview = str(movies['overview'][i])
        
        # 2. Extract just the names
        # (The CSV has them like dictionaries: {'id': 1, 'name': 'Action'})
        
        # Get Genre Names
        my_genres = []
        for item in genres_list:
            # Replace spaces with underscores (Action Movie -> Action_Movie)
            name = item['name'].replace(" ", "_")
            my_genres.append(name)
            
        # Get Keyword Names
        my_keywords = []
        for item in keywords_list:
            name = item['name'].replace(" ", "_")
            my_keywords.append(name)
            
        # Get Top 3 Cast Names
        my_cast = []
        counter = 0
        for item in cast_list:
            if counter < 3: # Only get the first 3
                name = item['name'].replace(" ", "_")
                my_cast.append(name)
                counter = counter + 1
        
        # 3. Join them all into one big string
        # " ".join(['a', 'b']) becomes "a b"
        genre_str = " ".join(my_genres)
        keyword_str = " ".join(my_keywords)
        cast_str = " ".join(my_cast)
        
        full_tag = overview + " " + genre_str + " " + keyword_str + " " + cast_str
        
        # Add to our list
        all_tags.append(full_tag)

    # Add the new column to the dataframe
    movies['tags'] = all_tags
    
    # Simplify the dataframe
    movies = movies[['id', 'title', 'tags']]
    
    # Remove any broken rows
    movies.dropna(inplace=True)
    movies.reset_index(drop=True, inplace=True)
    
    return movies

movies = load_data()

# ---------------------------------------------------------
# 2. MACHINE LEARNING (The "Brain")
# ---------------------------------------------------------
# This part uses a library (sklearn), so we keep it short.
# It's like using a calculator; we don't build the calculator from scratch.
@st.cache_data
def compute_similarity(movies):
    # Turn text into numbers
    tfidf = TfidfVectorizer(stop_words='english')
    matrix = tfidf.fit_transform(movies['tags'])
    
    # Calculate similarity
    similarity = cosine_similarity(matrix)
    return similarity

similarity = compute_similarity(movies)

# ---------------------------------------------------------
# 3. API HELPER
# ---------------------------------------------------------
def fetch_details(movie_id):
    # Prepare the website link
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
    
    try:
        data = requests.get(url).json()
        
        # Get image
        poster_path = data.get('poster_path')
        if poster_path:
            full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
        else:
            full_path = "https://via.placeholder.com/500x750?text=No+Image"
            
        # Get IMDb ID
        imdb_id = data.get('imdb_id')
        if imdb_id:
            imdb_link = f"https://www.imdb.com/title/{imdb_id}/"
        else:
            imdb_link = "https://www.imdb.com/"
            
        return full_path, imdb_link
        
    except:
        return "https://via.placeholder.com/500x750?text=Error", "https://www.imdb.com/"

# ---------------------------------------------------------
# 4. RECOMMENDATION FUNCTION
# ---------------------------------------------------------
# Helper function for sorting
def get_score(pair):
    return pair[1]

def recommend(movie_name):
    # 1. Find the movie index
    # (We use a simple loop instead of boolean indexing if that's confusing)
    movie_index = -1
    for i in range(len(movies)):
        if movies['title'][i].lower() == movie_name.lower():
            movie_index = i
            break
            
    if movie_index == -1:
        return [], [], []

    # 2. Get similarity scores
    scores = similarity[movie_index]
    
    # 3. Pair the score with the movie index: (0, 0.1), (1, 0.5), etc.
    pairs = list(enumerate(scores))
    
    # 4. Sort them. We use the helper function 'get_score' instead of lambda
    sorted_pairs = sorted(pairs, key=get_score, reverse=True)
    
    # 5. Get top 5 (ignoring the first one because it's the same movie)
    top_5_pairs = sorted_pairs[1:6]
    
    rec_names = []
    rec_posters = []
    rec_links = []
    
    for pair in top_5_pairs:
        index = pair[0] # The movie index
        movie_id = movies.iloc[index].id
        title = movies.iloc[index].title
        
        poster, link = fetch_details(movie_id)
        
        rec_names.append(title)
        rec_posters.append(poster)
        rec_links.append(link)
        
    return rec_names, rec_posters, rec_links

# ---------------------------------------------------------
# 5. USER INTERFACE
# ---------------------------------------------------------
st.title("🎬 Simple Movie Recommender")

selected_movie = st.selectbox("Pick a movie:", movies['title'].values)

if st.button("Recommend"):
    names, posters, links = recommend(selected_movie)
    
    if names:
        st.header("Top 5 Recommendations:")
        
        # Create 5 columns
        cols = st.columns(5)
        
        # Loop through the 5 results
        for i in range(5):
            with cols[i]:
                # Native Streamlit Image
                st.image(posters[i], use_container_width=True)
                # Movie Title
                st.write(f"**{names[i]}**")
                # Link Button (Native Streamlit)
                st.markdown(f"[View on IMDb]({links[i]})")
