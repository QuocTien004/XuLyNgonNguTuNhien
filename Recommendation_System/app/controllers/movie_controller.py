from app.models.movie_model import MovieModel
import pandas as pd

class MovieController:
    def __init__(self):
        self.movie_model = MovieModel()
    
    def search_movies(self, query):
        """
        Tìm kiếm phim dựa trên query
        """
        return self.movie_model.search_movies(query)
    
    def get_movie_by_id(self, movie_id):
        """
        Lấy thông tin phim theo ID
        """
        return self.movie_model.get_movie_by_id(movie_id)
    
    def add_review(self, movie_id, rating, review, user_id='user_1'):
        """
        Thêm đánh giá và bình luận cho phim
        """
        if not review:  # Kiểm tra nếu review là rỗng
            review = "Không có bình luận."  # Mặc định là chuỗi này nếu không có review
        new_review = pd.DataFrame({
            'movieId': [movie_id],
            'userId': [user_id],
            'rating': [rating],
            'review': [review]
        })
        self.movie_model.reviews_df = pd.concat([self.movie_model.reviews_df, new_review], ignore_index=True)
        self.movie_model.reviews_df.to_csv('app/data/reviews.csv', index=False)
        return True
    
    def get_movie_reviews(self, movie_id):
        """
        Lấy danh sách đánh giá và bình luận của phim
        """
        return self.movie_model.get_movie_reviews(movie_id)
