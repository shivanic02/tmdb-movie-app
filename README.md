# 🎬 TMDB Movie Recommender System

An interactive Streamlit app that recommends movies using content-based filtering powered by TMDB API. It pulls real-time metadata like overviews, genres, and keywords, then generates recommendations using TF-IDF and cosine similarity.

## 🔍 Features

- 🧠 Recommends top 5 similar movies based on description, genres, and keywords
- 🎥 Real-time movie data from the TMDB API
- 🖼️ Clean UI with posters and progress bars for similarity match
- ⚡ Fast and responsive app with `st.cache_data` for smart caching

## 🧠 How It Works

1. **Data Collection**  
   Fetches movie details and metadata from TMDB API (overview, genres, keywords, posters)

2. **Content Vectorization**  
   Combines all metadata into a single text string and transforms it using TF-IDF

3. **Similarity Computation**  
   Uses cosine similarity (`linear_kernel`) to rank similar movies

4. **User Interaction**  
   User selects a movie, and the app displays 5 most similar recommendations

## 🛠️ Tech Stack

- Python  
- Streamlit  
- TMDB API  
- scikit-learn (`TfidfVectorizer`, `linear_kernel`)  
- pandas, requests

## 🚀 Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/tmdb-recommender.git
   cd tmdb-recommender

2. **Install Dependencies**
   pip install -r requirements.txt

3. **Add your TMDB API key**
   Replace the API_KEY = "" in the script with your actual TMDB API key from TMDB.

4. **Run the app**
   streamlit run app.py

## 📌 Notes
All recommendations are based on content similarity (no collaborative filtering or ratings involved).

By default, the app fetches data from the first 7 pages of TMDB’s popular movies endpoint.

⚙️ Assembled with caffeine, Python, and zero chill by Shivani Chauhan 🧠💻
