import streamlit as st
import nltk
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# NLP modules
from views.aug import augmentation_view
from views.col import scraper_view
from views.pre import preprocessor_view
from views.rep import representation_view
from views.clas import classification_view

# Movie system controllers
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from app.controllers.movie_controller import MovieController
from app.controllers.recommendation_controller import RecommendationController

# Setup
st.set_page_config(page_title="Ứng dụng NLP & Movie Recommender", layout="wide")
nltk.download('averaged_perceptron_tagger_eng')

# Sidebar
st.sidebar.title("Chọn Ứng Dụng")
app_choice = st.sidebar.selectbox("Chọn ứng dụng", ["NLP Tool", "Movie Recommendation System"])

# ========================== NLP TOOL ==========================
if app_choice == "NLP Tool":
    st.sidebar.subheader("Tác vụ NLP")
    nlp_choice = st.sidebar.radio("Chọn một tác vụ:", 
                                  ["Thu thập dữ liệu", "Tăng cường dữ liệu", 
                                   "Tiền xử lý", "Biểu diễn dữ liệu", 
                                   "Phân loại văn bản"])
    
    if nlp_choice == "Thu thập dữ liệu":
        scraper_view()
    elif nlp_choice == "Tăng cường dữ liệu":
        augmentation_view()
    elif nlp_choice == "Tiền xử lý":
        preprocessor_view()
    elif nlp_choice == "Biểu diễn dữ liệu":
        representation_view()
    elif nlp_choice == "Phân loại văn bản":
        classification_view()

# ====================== MOVIE RECOMMENDER =====================
elif app_choice == "Movie Recommendation System":
    # Init controllers
    movie_controller = MovieController()
    recommendation_controller = RecommendationController()

    st.sidebar.subheader("Tính năng")
    movie_page = st.sidebar.selectbox("Chọn trang", ["Home", "Movies", "Recommendations"])

    if movie_page == "Home":
        st.title("Welcome to Movie Recommendation System")
        st.write("Discover new movies based on your preferences!")

    elif movie_page == "Movies":
        st.title("Movies")
        search_query = st.text_input("Search movies")
        if search_query:
            movies = movie_controller.search_movies(search_query)
            for movie in movies:
                year_str = f" ({movie['year']})" if movie['year'] > 0 else ""
                with st.expander(f"{movie['title']}{year_str}"):
                    st.write(f"**Rating:** {movie['vote_average']}/10")
                    rating = st.slider("Rate this movie", 1, 10, 5, key=f"rating_{movie['id']}")
                    review = st.text_area("Write a review", key=f"review_{movie['id']}")
                    if st.button("Submit Review", key=f"submit_{movie['id']}"):
                        movie_controller.add_review(movie['id'], rating, review)
                        st.success("Review submitted successfully!")

    elif movie_page == "Recommendations":
        st.title("Movie Recommendations")
        rec_type = st.selectbox("Choose recommendation type",
                                ["Collaborative Filtering", "Content-based Filtering", "Hybrid Filtering"])
        
        if rec_type == "Collaborative Filtering":
            recommendations = recommendation_controller.get_collaborative_recommendations()
        elif rec_type == "Content-based Filtering":
            recommendations = recommendation_controller.get_content_based_recommendations()
        else:
            recommendations = recommendation_controller.get_hybrid_recommendations()
        
        for movie in recommendations:
            year_str = f" ({movie['year']})" if movie['year'] > 0 else ""
            with st.expander(f"{movie['title']}{year_str}"):
                st.write(f"**Rating:** {movie['vote_average']}/10")
                rating = st.slider("Rate this movie", 1, 10, 5, key=f"rating_{movie['id']}")
                review = st.text_area("Write a review", key=f"review_{movie['id']}")
                if st.button("Submit Review", key=f"submit_{movie['id']}"):
                    movie_controller.add_review(movie['id'], rating, review)
                    st.success("Review submitted successfully!")
