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
        """Khởi tạo bộ phân loại văn bản"""
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

        num_batches = 10 
        batch_size = X_train.shape[0] // num_batches

        X_train, y_train = shuffle(X_train, y_train, random_state=42)

        # Huấn luyện mô hình 
        for i in range(num_batches):
            batch_start = i * batch_size
            batch_end = (i + 1) * batch_size if (i + 1) * batch_size < X_train.shape[0] else X_train.shape[0]
            
            self.model.partial_fit(X_train[batch_start:batch_end], y_train[batch_start:batch_end], classes=np.unique(y_train))  
            
            if progress_bar:
                progress_bar.progress((i + 1) / num_batches)  
            time.sleep(0.2) 

        train_time = time.time() - start_time  

        # độ chính xác
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
        X_text = self.vectorizer.transform([text])
        prediction = self.model.predict(X_text)[0]
        return prediction