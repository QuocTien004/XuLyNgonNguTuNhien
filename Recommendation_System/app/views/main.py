import streamlit as st
import pandas as pd
import sys
import os

# Thêm thư mục gốc vào PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.controllers.movie_controller import MovieController
from app.controllers.recommendation_controller import RecommendationController

st.set_page_config(page_title="Hệ thống Gợi ý Phim", layout="wide")

# Khởi tạo controllers
movie_controller = MovieController()
recommendation_controller = RecommendationController()

# Sidebar - Thanh chọn nút
st.sidebar.title("Chọn trang")
page = st.sidebar.radio("Chọn chức năng", ["Trang chủ", "Danh sách phim", "Gợi ý phim"])

# Trang chủ
if page == "Trang chủ":
    st.title("Hệ thống Gợi ý Phim")
    st.markdown("Chào mừng bạn đến với hệ thống gợi ý phim!")

# Danh sách phim
if page == "Danh sách phim":
    st.title("Danh sách phim")
    st.markdown("Tìm kiếm và xem thông tin phim.")

    search_query = st.text_input("🔍 Nhập tên phim cần tìm")
    user_id = st.text_input("Nhập ID người dùng")

    if search_query:
        movies = movie_controller.search_movies(search_query)
        if not movies:
            st.warning("Không tìm thấy phim với tên này. Vui lòng thử lại.")
        for movie in movies:
            year_str = f" ({movie['year']})" if movie['year'] > 0 else ""
            st.markdown("---")
            with st.container():
                cols = st.columns([2, 1])

                with cols[0]:
                    st.markdown(f"### {movie['title']}{year_str}")
                    st.markdown(f"**Tóm tắt:** {movie['overview']}")

                with cols[1]:
                    with st.expander("Đánh giá"):
                        st.markdown(f"**Đánh giá hiện tại:** {movie['vote_average']}/10")
                        rating = st.slider("Đánh giá phim", 1, 10, 5, key=f"rating_{movie['id']}")
                        
                        # Thêm phần nhập đánh giá chi tiết
                        review = st.text_area("Nhập đánh giá chi tiết của bạn (Tùy chọn)", "")
                        
                        if st.button("Gửi", key=f"submit_{movie['id']}"):
                            user_id_clean = str(user_id).strip()
                            if not user_id_clean:
                                st.warning("Vui lòng nhập ID người dùng trước khi gửi đánh giá.")
                            elif not rating:
                                st.warning("Vui lòng chọn đánh giá phim.")
                            else:
                                movie_controller.add_review(movie['id'], rating, review, user_id_clean)
                                st.success("Cảm ơn bạn đã gửi đánh giá!")
                                # Thêm phần hiển thị các đánh giá hiện tại
                                reviews = movie_controller.get_movie_reviews(movie['id'])
                                if reviews:
                                    st.markdown("### Đánh giá hiện tại:")
                                    for rev in reviews:
                                        st.markdown(f"**User {rev['userId']}**: {rev['rating']}/10 - {rev['review']}")
                                else:
                                    st.markdown("Chưa có đánh giá nào.")