import requests
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import time

# 1. API Configuration
API_KEY = ""
HEADERS = {"accept": "application/json"}

# 2. Enhanced Data Fetching
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_tmdb_data(max_pages=7):
    """Fetch all popular movies from TMDB API"""
    try:
        movie_data = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for page in range(1, max_pages+1):
            status_text.text(f"Fetching page {page}/{max_pages}...")
            progress_bar.progress(page/max_pages)
            
            url = f"https://api.themoviedb.org/3/movie/popular?api_key={API_KEY}&page={page}"
            response = requests.get(url, headers=HEADERS, timeout=10)
            response.raise_for_status()
            movies = response.json().get("results", [])
            
            for movie in movies:
                try:
                    details_url = f"https://api.themoviedb.org/3/movie/{movie['id']}?api_key={API_KEY}"
                    details = requests.get(details_url, headers=HEADERS, timeout=10).json()
                    
                    keywords_url = f"https://api.themoviedb.org/3/movie/{movie['id']}/keywords?api_key={API_KEY}"
                    keywords = requests.get(keywords_url, headers=HEADERS, timeout=10).json().get("keywords", [])
                    
                    movie_data.append({
                        "id": movie["id"],
                        "title": movie.get("title", "Untitled"),
                        "overview": movie.get("overview", ""),
                        "genres": [g["name"] for g in details.get("genres", [])],
                        "keywords": [k["name"] for k in keywords],
                        "poster_url": f"https://image.tmdb.org/t/p/w200{movie['poster_path']}" if movie.get('poster_path') else None,
                        "year": movie.get("release_date", "")[:4] if movie.get("release_date") else "N/A",
                        "popularity": movie.get("popularity", 0)
                    })
                    time.sleep(0.1)
                except Exception as e:
                    continue
        
        progress_bar.empty()
        status_text.empty()
        return pd.DataFrame(movie_data)
    except Exception as e:
        st.error(f"Failed to fetch data: {str(e)}")
        return pd.DataFrame()

# 3. Recommendation Engine
def build_recommender(df):
    """Create content-based recommendation model"""
    if df.empty:
        return None
        
    df["content"] = (
        df["overview"] + " " +
        df["genres"].apply(lambda x: " ".join(x)) + " " +
        df["keywords"].apply(lambda x: " ".join(x))
    )
    
    tfidf = TfidfVectorizer(stop_words="english", min_df=2)
    tfidf_matrix = tfidf.fit_transform(df["content"])
    
    return {
        "model": tfidf,
        "similarity": linear_kernel(tfidf_matrix, tfidf_matrix),
        "movies": df
    }

# 4. Streamlit App with Improved UI
def main():
    st.set_page_config(page_title="TMDB Recommender", layout="wide")
    st.title("🍿 Movie Recommender")
    
    # Load data
    with st.spinner("Fetching latest movies..."):
        movies_df = fetch_tmdb_data()
        model = build_recommender(movies_df)
        st.success(f"Loaded {len(movies_df)} movies")
    
    # Create display titles with years
    movies_df["display_title"] = movies_df["title"] + " (" + movies_df["year"].astype(str) + ")"
    
    # Movie selection with search
    search_query = st.text_input("Search movies")
    
    if search_query:
        filtered_movies = movies_df[movies_df["title"].str.contains(search_query, case=False)]
        selected_display = st.selectbox(
            "Choose a movie",
            filtered_movies["display_title"].tolist(),
            index=0 if not filtered_movies.empty else None
        )
    else:
        selected_display = st.selectbox(
            "Choose a movie",
            movies_df["display_title"].tolist(),
            index=0
        )
    
    # Extract actual title from display title
    selected_movie = selected_display.split(" (")[0]
    
    if st.button("Get Recommendations"):
        if model is None:
            st.warning("No movies loaded")
            return
            
        try:
            # Get recommendations from ALL movies
            idx = model["movies"][model["movies"]["title"] == selected_movie].index[0]
            sim_scores = list(enumerate(model["similarity"][idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]  # Top 5
            
            st.subheader(f"Movies similar to {selected_display}:")
            cols = st.columns(5)
            
            for i, (movie_idx, score) in enumerate(sim_scores):
                movie = model["movies"].iloc[movie_idx]
                with cols[i]:
                    if movie["poster_url"]:
                        st.image(movie["poster_url"], use_container_width=True)
                    st.markdown(f"**{movie['title']}** ({movie['year']})")
                    st.caption(f"Genres: {', '.join(movie['genres'][:2])}")
                    st.progress(float(score), text=f"Match: {score*100:.0f}%")
                    
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
