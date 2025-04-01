import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datasets import load_dataset  
import time
from sklearn.utils import shuffle 

class TextClassifier:
    def __init__(self, dataset_name, model_type):
        self.dataset_name = dataset_name
        self.model_type = model_type
        self.vectorizer = TfidfVectorizer(max_features=5000)  
        self.model = None

        self.text_column = None
        self.label_column = None

        self._load_dataset()
    
    def _load_dataset(self):
        """Tải dataset từ Hugging Face"""
        dataset_mapping = {
            "IMDb Review": "imdb",
            "Yelp Review": "yelp_review_full",
            "Amazon Review": "amazon_polarity",
            "TREC": "trec",
            "Yahoo! Answer": "yahoo_answers_topics",
            "AG's News": "ag_news",
            "Sogou News": "sogou_news",
            "DBPedia": "dbpedia_14"
        }
        
        if self.dataset_name not in dataset_mapping:
            raise ValueError("Dataset không hợp lệ!")

        dataset = load_dataset(dataset_mapping[self.dataset_name], trust_remote_code=True)

        sample = dataset["train"][0]
        for col in sample.keys():
            if isinstance(sample[col], str):  
                self.text_column = col
                break
        
        for col in sample.keys():
            unique_values = set(dataset["train"][col])
            if len(unique_values) < 20 and isinstance(list(unique_values)[0], (int, str)):  
                self.label_column = col
                break

        if not self.text_column or not self.label_column:
            raise ValueError("Không xác định được cột văn bản hoặc nhãn!")

        # Lấy dữ liệu
        self.train_texts, self.train_labels = dataset["train"][self.text_column], dataset["train"][self.label_column]
        self.test_texts, self.test_labels = dataset["test"][self.text_column], dataset["test"][self.label_column]

    def train_model(self, progress_bar):
        """Huấn luyện mô hình"""
        start_time = time.time()

        # Vector hóa dữ liệu
        X_train = self.vectorizer.fit_transform(self.train_texts)
        X_test = self.vectorizer.transform(self.test_texts)

        y_train = np.array(self.train_labels)
        y_test = np.array(self.test_labels)

        # Chọn mô hình 
        model_mapping = {
            "Naive Bayes": MultinomialNB(),
            "Logistic Regression": LogisticRegression(max_iter=200),
            "SVM": SVC(kernel='linear'),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
            "Decision Tree": DecisionTreeClassifier()
        }

        if self.model_type not in model_mapping:
            raise ValueError("Thuật toán không hợp lệ!")

        self.model = model_mapping[self.model_type]

        # Huấn luyện mô hình trên toàn bộ dữ liệu
        self.model.fit(X_train, y_train)

        train_time = time.time() - start_time  

        # Đánh giá mô hình
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        return accuracy, train_time

    def simulate_training_step(self, progress_bar, train_time):
        """Cập nhật progress bar theo thời gian train thực tế"""
        step_duration = train_time / 100 
        for i in range(1, 101):
            time.sleep(step_duration)
            progress_bar.progress(i / 100.0) 

    def predict(self, text):
        """Dự đoán phân loại văn bản"""
        if self.model is None:
            raise ValueError("Mô hình chưa được train! Vui lòng huấn luyện trước khi dự đoán.")
    
        X_text = self.vectorizer.transform([text])
        prediction = self.model.predict(X_text)

        if len(prediction) == 0:
            return "Không thể dự đoán"  

        label = prediction[0]  # Lấy nhãn dự đoán

        # Kiểm tra nhãn
        if self.dataset_name in ["IMDb Review", "Yelp Review", "Amazon Review"]:
            return "Tích cực" if label == 1 else "Tiêu cực"
    
        return f"Dự đoán nhãn: {label}"
