
import html
import re
from pathlib import Path

import pandas as pd
import streamlit as st

# 1. PAGE CONFIGURATION

st.set_page_config(
    page_title="CineMatch | Discover Movies",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

# 2. CINEMATIC DESIGN

st.markdown("""
<style>
.stApp {
    background: #090b10;
    color: #f4f4f5;
}

.block-container {
    max-width: 1450px;
    padding-top: 1.5rem;
}

[data-testid="stSidebar"] {
    background: #14161e;
}

.brand {
    color: #e50914;
    font-size: 2.3rem;
    font-weight: 900;
    letter-spacing: 2px;
}

.hero {
    background: linear-gradient(115deg, #29213e, #131a29, #101014);
    border-radius: 18px;
    padding: 50px 38px;
    margin: 20px 0 30px;
    border: 1px solid #333344;
}

.hero-title {
    font-size: 3.2rem;
    font-weight: 900;
    color: white;
    margin-bottom: 12px;
}

.hero-subtitle {
    color: #c9cad5;
    font-size: 1.1rem;
    max-width: 650px;
}

.movie-card {
    background: linear-gradient(150deg, #292337, #171d2b);
    border: 1px solid #39364b;
    border-radius: 14px;
    padding: 20px;
    min-height: 195px;
    margin-bottom: 12px;
}

.movie-symbol {
    font-size: 2.8rem;
    margin-bottom: 15px;
}

.movie-title {
    font-weight: 700;
    color: white;
    font-size: 1.05rem;
    min-height: 52px;
}

.movie-rating {
    color: #ffcb73;
    font-weight: 700;
}

.movie-year {
    color: #a7adba;
    font-size: 0.85rem;
}
.movie-row {
    display: flex;
    gap: 16px;
    overflow-x: auto;
    padding: 12px 4px 22px;
    scroll-snap-type: x mandatory;
}

.movie-row::-webkit-scrollbar {
    height: 7px;
}

.movie-row::-webkit-scrollbar-thumb {
    background: #52576a;
    border-radius: 10px;
}

.movie-row .movie-card {
    flex: 0 0 220px;
    scroll-snap-align: start;
    transition: transform 0.2s ease;
}

.movie-row .movie-card:hover {
    transform: translateY(-6px);
    border-color: #e50914;
}
.section-heading {
    color: white;
    font-size: 1.5rem;
    font-weight: 750;
    margin-top: 24px;
    margin-bottom: 16px;
}

[data-testid="stMetric"] {
    background: #171b27;
    padding: 15px;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# 3. LOAD MODEL DATA

@st.cache_data
def load_data():

    predictions = pd.read_pickle(
        BASE_DIR / "models" / "predicted_ratings.pkl"
    )

    ratings = pd.read_csv(
        BASE_DIR / "data" / "u.data",
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"]
    )

    movies = pd.read_csv(
        BASE_DIR / "data" / "u.item",
        sep="|",
        encoding="latin-1",
        header=None,
        usecols=[0, 1],
        names=["movie_id", "title"]
    )

    predictions.index = predictions.index.astype(int)
    predictions.columns = predictions.columns.astype(int)

    return predictions, ratings, movies


predictions, ratings, movies = load_data()

movie_titles = movies.set_index("movie_id")["title"]

# 4. RECOMMENDATION FUNCTION

def recommend_movies(user_id, top_n=12):

    rated_movies = ratings.loc[
        ratings["user_id"] == user_id,
        "movie_id"
    ]

    user_predictions = predictions[user_id]

    unseen_movies = user_predictions.drop(
        index=rated_movies,
        errors="ignore"
    ).dropna()

    top_movies = unseen_movies.nlargest(top_n)

    result = pd.DataFrame({
        "movie_id": top_movies.index,
        "raw_score": top_movies.values
    })

    result["predicted_rating"] = (
        result["raw_score"].clip(1, 5)
    )

    result["title"] = (
        result["movie_id"]
        .map(movie_titles)
        .fillna("Unknown movie")
    )

    return result.reset_index(drop=True)

# 5. MOVIE DISPLAY HELPERS

def get_year(title):

    match = re.search(r"\((\d{4})\)", title)

    return match.group(1) if match else "Classic"


def movie_card(movie, rank):

    title = html.escape(str(movie["title"]))
    year = get_year(str(movie["title"]))
    score = float(movie["predicted_rating"])

    st.markdown(f"""
    <div class="movie-card">
        <div class="movie-symbol">🎞️</div>
        <div class="movie-title">{title}</div>
        <div class="movie-year">{year}</div>
        <div class="movie-rating">
            ⭐ {score:.2f} / 5
        </div>
        <div class="movie-year">
            Recommended #{rank}
        </div>
    </div>
    """, unsafe_allow_html=True)

# MOVIE DETAILS POPUP

@st.dialog("🎬 Movie Details")
def show_movie_details(movie, rank):

    title = str(movie["title"])
    year = get_year(title)
    rating = float(movie["predicted_rating"])

    st.markdown(f"## {title}")

    st.write(f"📅 Release Year: {year}")

    st.write(f"⭐ Predicted Rating: {rating:.2f} / 5")

    st.write(f"🏆 Recommendation Rank: #{rank}")

    st.divider()

    st.info(
        "This movie was recommended by our "
        "TensorFlow collaborative filtering model, "
        "based on patterns in user ratings."
    )

    if st.button("Close", use_container_width=True):
        st.rerun()

# 6. SIDEBAR PROFILE

with st.sidebar:

    st.markdown(
        '<div class="brand">CINEMATCH</div>',
        unsafe_allow_html=True
    )

    st.write("Your personalized movie universe")

    st.divider()

    user_id = st.selectbox(
        "Choose your profile",
        options=sorted(predictions.columns.tolist()),
        format_func=lambda user: f"👤 User {user}"
    )

    top_n = st.slider(
        "Recommendations",
        min_value=6,
        max_value=24,
        value=12,
        step=3
    )

    st.divider()

    st.caption(
        "Recommendations powered by TensorFlow "
        "collaborative filtering."
    )

# 7. PERSONALIZED HOMEPAGE

st.markdown(
    '<div class="brand" style="padding-top:40px;">CINEMATCH</div>',
    unsafe_allow_html=True
)

st.write(f"Welcome back, User {user_id} 👋")

recommendations = recommend_movies(user_id, top_n)

if recommendations.empty:
    st.info("No unseen movies available for this profile.")
    st.stop()

featured = recommendations.iloc[0]

featured_title = html.escape(str(featured["title"]))
featured_score = float(featured["predicted_rating"])

st.markdown(f"""
<div class="hero">
    <p style="color:#ff8b91;font-weight:700;">
        ✨ YOUR #1 PICK
    </p>
    <div class="hero-title">
        {featured_title}
    </div>
    <p class="hero-subtitle">
        A movie selected for your profile using
        your viewing preferences and our
        TensorFlow recommendation engine.
    </p>
    <h3 style="color:#ffd27a;">
        ⭐ {featured_score:.2f} / 5
    </h3>
</div>
""", unsafe_allow_html=True)

# 8. CLICKABLE MOVIE RECOMMENDATIONS

st.markdown(
    '<div class="section-heading">🍿 Recommended For You</div>',
    unsafe_allow_html=True
)

movie_row = st.container(
    horizontal=True,
    wrap=False,
    gap="small"
)

with movie_row:

    for index, movie in recommendations.iterrows():

        with st.container(width=230, border=False):

            movie_card(movie, index + 1)

            if st.button(
                "View Details",
                key=f"movie_{user_id}_{index}",
                use_container_width=True
            ):

                show_movie_details(
                    movie,
                    index + 1
                )

# 9. BECAUSE YOU LIKED

st.markdown(
    '<div class="section-heading">❤️ Movies You Rated Highly</div>',
    unsafe_allow_html=True
)

user_history = ratings[
    ratings["user_id"] == user_id
].copy()

liked_movies = (
    user_history[user_history["rating"] >= 4]
    .sort_values("rating", ascending=False)
    .head(4)
)

if liked_movies.empty:

    st.info("No highly rated movies found.")

else:

    columns = st.columns(4)

    for position, (_, movie) in enumerate(liked_movies.iterrows()):

        with columns[position]:

            title = html.escape(
                str(movie_titles.get(
                    movie["movie_id"],
                    "Unknown movie"
                ))
            )

            st.markdown(f"""
            <div class="movie-card">
                <div class="movie-symbol">❤️</div>
                <div class="movie-title">{title}</div>
                <div class="movie-rating">
                    Your rating: ⭐ {movie["rating"]:.0f}/5
                </div>
            </div>
            """, unsafe_allow_html=True)

# 10. MODEL INFORMATION

st.divider()

st.markdown(
    '<div class="section-heading">🧠 Powered by Machine Learning</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Users", "943")
col2.metric("Movies", "1,682")
col3.metric("Test MAE", "0.8788")
col4.metric("Test RMSE", "1.1151")

st.caption(
    "MovieLens 100K | TensorFlow | "
    "Collaborative Filtering | Matrix Factorization"
)
