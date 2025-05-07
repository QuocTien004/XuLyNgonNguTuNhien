import streamlit as st
import pandas as pd
import sys
import os

# Thêm thư mục gốc vào PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.controllers.movie_controller import MovieController
from app.controllers.recommendation_controller import RecommendationController

st.set_page_config(page_title="Movie Recommendation System", layout="wide")

# Khởi tạo controllers
movie_controller = MovieController()
recommendation_controller = RecommendationController()

# Sidebar
st.sidebar.title("Movie Recommendation System")
page = st.sidebar.selectbox("Choose a page", ["Home", "Movies", "Recommendations"])

if page == "Home":
    st.title("Welcome to Movie Recommendation System")
    st.write("Discover new movies based on your preferences!")

elif page == "Movies":
    st.title("Movies")
    
    # Search movies
    search_query = st.text_input("Search movies")
    if search_query:
        movies = movie_controller.search_movies(search_query)
        for movie in movies:
            year_str = f" ({movie['year']})" if movie['year'] > 0 else ""
            with st.expander(f"{movie['title']}{year_str}"):
                # Display only summary and rating
                st.write(f"**Rating:** {movie['vote_average']}/10")
                
                # Rating and review (optional)
                rating = st.slider("Rate this movie", 1, 10, 5, key=f"rating_{movie['id']}")
                review = st.text_area("Write a review", key=f"review_{movie['id']}")
                if st.button("Submit Review", key=f"submit_{movie['id']}"):
                    movie_controller.add_review(movie['id'], rating, review)
                    st.success("Review submitted successfully!")

elif page == "Recommendations":
    st.title("Movie Recommendations")
    
    # Select recommendation type
    rec_type = st.selectbox(
        "Choose recommendation type",
        ["Collaborative Filtering", "Content-based Filtering", "Hybrid Filtering"]
    )
    
    if rec_type == "Collaborative Filtering":
        recommendations = recommendation_controller.get_collaborative_recommendations()
    elif rec_type == "Content-based Filtering":
        recommendations = recommendation_controller.get_content_based_recommendations()
    else:
        recommendations = recommendation_controller.get_hybrid_recommendations()
    
    # Display recommendations with summary and rating only
    for movie in recommendations:
        year_str = f" ({movie['year']})" if movie['year'] > 0 else ""
        with st.expander(f"{movie['title']}{year_str}"):
            st.write(f"**Rating:** {movie['vote_average']}/10")
            
            # Rating and review (optional)
            rating = st.slider("Rate this movie", 1, 10, 5, key=f"rating_{movie['id']}")
            review = st.text_area("Write a review", key=f"review_{movie['id']}")
            if st.button("Submit Review", key=f"submit_{movie['id']}"):
                movie_controller.add_review(movie['id'], rating, review)
                st.success("Review submitted successfully!")
