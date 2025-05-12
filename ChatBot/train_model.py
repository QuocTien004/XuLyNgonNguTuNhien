import json
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

nltk.download('punkt')

# Đọc dữ liệu từ intents.json
with open("intents.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Tạo danh sách các câu hỏi (patterns) và nhãn (tags)
patterns = []
tags = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])

# Biến đổi các câu hỏi thành các vector số học
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(patterns)

# Huấn luyện mô hình Naive Bayes
model = MultinomialNB()
model.fit(X, tags)

# Lưu mô hình và vectorizer để sử dụng sau
with open('chatbot_model.pkl', 'wb') as f:
    pickle.dump((vectorizer, model, data), f)

print("Mô hình đã được huấn luyện và lưu lại.")
