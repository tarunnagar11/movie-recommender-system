import streamlit as st
import pickle
import pandas as pd

# Set page configuration
st.set_page_config(page_title="🎥 Movie Recommendation System", layout="wide")

# Full dark theme + white radio labels in sidebar
st.markdown("""
<style>
/* Dark theme for body */
html, body, .stApp {
    background-color: #000000 !important;
    color: #ffffff !important;
}

/* Sidebar background and text */
section[data-testid="stSidebar"] {
    background-color: #121212 !important;
    border-right: 2px solid #333333;
    color: white !important;
}

/* Sidebar radio button text */
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stRadio div {
    color: white !important;
    font-weight: 500;
}

/* Optional: white "i" icons inside sidebar (info) */
section[data-testid="stSidebar"] svg {
    color: white !important;
}

/* Header area */
[data-testid="stHeader"] {
    background-color: #000000 !important;
}

/* Dropdown styling */
div[data-baseweb="select"] {
    background-color: #e4e4e4 !important;
    color: #000000 !important;
    border-radius: 10px;
}

/* Slider text */
.css-1emrehy .stSlider > div > div {
    color: white !important;
}

/* Movie cards */
.movie-card {
    padding: 1rem;
    margin: 0.5rem 0;
    background: linear-gradient(145deg, #1a1a1a, #2a2a2a);
    border-radius: 15px;
    box-shadow: 0 8px 20px rgba(255,255,255,0.05);
    font-size: 18px;
}
.movie-title {
    font-weight: bold;
    font-size: 20px;
    color: #ffffff;
}

/* Button styling */
.stButton>button {
    background: linear-gradient(to right, #ff416c, #ff4b2b);
    color: white;
    font-weight: bold;
    border-radius: 12px;
    border: none;
    padding: 0.7em 1.8em;
    font-size: 1.1rem;
    transition: all 0.3s ease-in-out;
}
.stButton>button:hover {
    background: linear-gradient(to right, #ff4b2b, #ff416c);
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

# Load models and data
cb_movies = pickle.load(open('model/movie_list.pkl', 'rb'))
cb_similarity = pickle.load(open('model/similarity.pkl', 'rb'))

cf_movie_list = pickle.load(open('model/movierecommendation_list.pkl', 'rb'))
cf_movies = pickle.load(open('model/movies.pkl', 'rb'))
cf_final_dataset = pickle.load(open('model/final_dataset.pkl', 'rb'))
cf_knn = pickle.load(open('model/knn_model.pkl', 'rb'))
cf_csr_data = pickle.load(open('model/csr_data.pkl', 'rb'))

# Sidebar
st.sidebar.header("⚙️ Settings")
system_choice = st.sidebar.radio("📽️ Select Recommendation System", ["🧠 Content-Based", "🤖 Collaborative Filtering"])

# Main Title and Subtitle
st.title("🎞️ Movie Recommendation System")
st.markdown("✨ _Get smart movie suggestions based on your preferences or what others love!_")

# Content-Based Section
if system_choice == "🧠 Content-Based":
    st.subheader("🧠 Content-Based Recommendations")
    movie_list = cb_movies['title'].values
    selected_movie = st.selectbox("🎬 Choose a Movie", movie_list, key="cb_movie")
    n_recs = st.slider("📈 How many recommendations?", 1, 20, 5, key="cb_slider")

    def content_based_recommend(movie, n):
        index = cb_movies[cb_movies['title'] == movie].index[0]
        distances = sorted(list(enumerate(cb_similarity[index])), reverse=True, key=lambda x: x[1])
        return [cb_movies.iloc[i[0]].title for i in distances[1:n+1]]

    if st.button("🍿 Recommend Movies", key="cb_btn"):
        recommended_titles = content_based_recommend(selected_movie, n_recs)
        st.success(f"📌 Top {n_recs} picks based on **{selected_movie}**")
        for name in recommended_titles:
            st.markdown(f"<div class='movie-card'><span class='movie-title'>🎬 {name}</span></div>", unsafe_allow_html=True)

# Collaborative Filtering Section
else:
    st.subheader("🤖 Collaborative Filtering Recommendations")
    selected_movie_cf = st.selectbox("🎥 Select a Movie You Like", cf_movie_list, key="cf_movie")
    n_recs_cf = st.slider("📈 Number of Recommendations", 1, 20, 5, key="cf_slider")

    def collaborative_filter_recommend(movie_name, num_recs=10):
        movie_match = cf_movies[cf_movies['title'] == movie_name]
        if movie_match.empty:
            return ["❌ Movie not found."]
        movie_id = movie_match.iloc[0]['movieId']

        try:
            idx_in_final = cf_final_dataset[cf_final_dataset['movieId'] == movie_id].index[0]
        except IndexError:
            return ["⚠️ This movie has too few ratings to be recommended."]

        distances, indices = cf_knn.kneighbors(cf_csr_data[idx_in_final], n_neighbors=num_recs + 1)
        recs = []
        for i in sorted(zip(indices.squeeze(), distances.squeeze()), key=lambda x: x[1])[1:]:
            m_id = cf_final_dataset.iloc[i[0]]['movieId']
            title = cf_movies[cf_movies['movieId'] == m_id]['title'].values[0]
            recs.append(f"{title} 🎯 Score: {round(1 - i[1], 2)}")
        return recs

    if st.button("🎥 Recommend Movies", key="cf_btn"):
        recommendations = collaborative_filter_recommend(selected_movie_cf, n_recs_cf)
        st.success(f"📌 Top {n_recs_cf} picks based on **{selected_movie_cf}**")
        for idx, movie in enumerate(recommendations, 1):
            st.markdown(f"<div class='movie-card'><span class='movie-title'>#{idx} 🎬 {movie}</span></div>", unsafe_allow_html=True)
