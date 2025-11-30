import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------
# 1. Load Dataset
# ---------------------------------------------------------
movies = pd.read_csv("movies.csv")   # make sure file is in same folder
movies = movies[['title', 'overview']]
movies.dropna(inplace=True)

# ---------------------------------------------------------
# 2. Convert Overview Text → TF-IDF Vectors
# ---------------------------------------------------------
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['overview'])

# ---------------------------------------------------------
# 3. Compute Cosine Similarity Matrix
# ---------------------------------------------------------
similarity = cosine_similarity(tfidf_matrix)

# ---------------------------------------------------------
# 4. Recommendation Function
# ---------------------------------------------------------
def recommend(movie_name):
    movie_name = movie_name.strip()

    if movie_name not in movies['title'].values:
        return ["Movie not found in dataset"]

    index = movies[movies['title'] == movie_name].index[0]
    distances = similarity[index]

    # sort all movies based on similarity score
    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]    # skip the first (same movie)

    recommendations = []
    for i in movie_list:
        recommendations.append(movies.iloc[i[0]].title)

    return recommendations

# ---------------------------------------------------------
# 5. Test (optional)
# ---------------------------------------------------------
if __name__ == "__main__":
    print("Example Recommendations for 'Avatar':")
    print(recommend("Avatar"))
