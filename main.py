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
    all_genres_for_ui = [] 
    
    # Process each movie
    for i in range(len(movies)):
        
        # Parse stringified lists
        try:
            genres_list = eval(movies['genres'][i]) 
            keywords_list = eval(movies['keywords'][i])
        except:
            continue # Skip rows if data is corrupt
        
        # Handle missing overview
        overview = str(movies['overview'][i])
        if overview == "nan":
            overview = ""
        
        # Extract genre names
        tag_genres = []
        ui_genres = []
        
        for item in genres_list:
            name = item['name']
            ui_genres.append(name) # Save "Science Fiction"
            tag_genres.append(name.replace(" ", "_")) # Save "Science_Fiction"
            
        # Extract keyword names
        tag_keywords = []
        for item in keywords_list:
            name = item['name'].replace(" ", "_")
            tag_keywords.append(name)
        
        # Create tag string
        genre_str = " ".join(tag_genres)
        keyword_str = " ".join(tag_keywords)
        full_tag = overview + " " + genre_str + " " + keyword_str
        
        all_tags.append(full_tag)
        all_genres_for_ui.append(ui_genres)

    movies['tags'] = all_tags
    movies['genres_ui'] = all_genres_for_ui 
    
    # Select relevant columns
    movies = movies[['id', 'title', 'tags', 'genres_ui']]
    movies.dropna(inplace=True)
    movies.reset_index(drop=True, inplace=True)
    
    return movies

movies = load_data()

# ---------------------------------------------------------
# 2. FEATURE EXTRACTION & SIMILARITY
# ---------------------------------------------------------
@st.cache_data
def compute_similarity(movies):
    tfidf = TfidfVectorizer(stop_words='english')
    matrix = tfidf.fit_transform(movies['tags'])
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
        poster_path = data.get('poster_path')
        if poster_path:
            full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
        else:
            full_path = "https://via.placeholder.com/500x750?text=No+Image"
            
        imdb_id = data.get('imdb_id')
        imdb_url = f"https://www.imdb.com/title/{imdb_id}/" if imdb_id else "https://www.imdb.com/"
            
        return full_path, imdb_url
    except:
        return "https://via.placeholder.com/500x750?text=Error", "https://www.imdb.com/"

# ---------------------------------------------------------
# 4. RECOMMENDATION ENGINE
# ---------------------------------------------------------
def get_score(pair):
    return pair[1]

def recommend(movie_name):
    # Case-insensitive search
    movie_index = -1
    for i in range(len(movies)):
        if movies['title'][i].lower() == movie_name.lower():
            movie_index = i
            break
            
    if movie_index == -1:
        return [], [], []

    scores = similarity[movie_index]
    pairs = list(enumerate(scores))
    sorted_pairs = sorted(pairs, key=get_score, reverse=True)
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
st.set_page_config(page_title="Movie Recommender", layout="wide")

st.title("🎬 Movie Recommender")
st.write("Browse movies based on genres, and selected movies.")

# --- GENRE FILTER ---
unique_genres = []
for genre_list in movies['genres_ui']:
    for genre in genre_list:
        if genre not in unique_genres:
            unique_genres.append(genre)
unique_genres.sort()

genre_options = ["All Genres"] + unique_genres
selected_genre = st.selectbox("Filter by Genre:", genre_options)

filtered_movie_titles = []
if selected_genre == "All Genres":
    filtered_movie_titles = movies['title'].values
else:
    for i in range(len(movies)):
        if selected_genre in movies['genres_ui'][i]:
            filtered_movie_titles.append(movies['title'][i])

# --- MOVIE SELECTION ---
if len(filtered_movie_titles) > 0:
    selected_movie = st.selectbox("Pick a movie:", filtered_movie_titles)

    if st.button("Recommend"):
        names, posters, links = recommend(selected_movie)
        
        if names:
            st.subheader(f"Because you liked '{selected_movie}':")
            cols = st.columns(5)
            for i in range(5):
                with cols[i]:
                    html_code = f"""
                        <a href="{links[i]}" target="_blank">
                            <img src="{posters[i]}" style="width:100%; border-radius:10px; margin-bottom: 10px;">
                        </a>
                        <div style="text-align:center;"><b>{names[i]}</b></div>
                    """
                    st.markdown(html_code, unsafe_allow_html=True)
else:
    st.warning("No movies found for this genre.")
