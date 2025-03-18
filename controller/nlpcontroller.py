from model.processing import TextProcessing
from model.representation import TextRepresentation
from model.augmentation import TextAugmentation
from model.data_collection import Data
from model.text_classification import NaiveBayes
from model.text_classification import MaximizingLikelihood
from model.text_classification import SVM
import numpy as np

class NLPController:

    #----------Cào dữ liệu----------
    @staticmethod
    def scrape_data(url):
        return Data.web_scraping(url)

    #----------Tăng cường dữ liệu----------
    @staticmethod
    def augment_text(text, methods):
        return TextAugmentation.augment_text(text, methods)

    #----------Tiền xử lý dữ liệu----------
    @staticmethod
    def process_text(text, options):
        result = text
        if options.get("Tách câu"):
            result = TextProcessing.sentence_tokenization(result)
        if options.get("Xóa từ dừng"):
            result = TextProcessing.remove_stopwords(result)
        if options.get("Xóa dấu câu"):
            result = TextProcessing.remove_punctuation(result)
        if options.get("Stemming - Lemmatization"):
            result = TextProcessing.stemming(result)
        if options.get("Nhận diện thực thể (NER)"):
            result = TextProcessing.named_entity_recognition(result)
        if options.get("Sửa lỗi chính tả"):
            result = TextProcessing.correct_spelling(result)
        if options.get("Sửa viết tắt"):
            result = TextProcessing.fix_contractions(result)
        return result

    #----------Biểu diễn dữ liệu----------
    @staticmethod
    def represent_text(text, method):
        if method == "One-Hot Encoding":
            return TextRepresentation.one_hot_encoding(text)
        elif method == "Bag of Words":
            return TextRepresentation.bag_of_words(text)
        elif method == "TF-IDF":
            return TextRepresentation.tfidf_representation(text)
        elif method == "GloVe":
            return TextRepresentation.glove_representation(text)
        elif method == "Bag of N-Grams":
            return TextRepresentation.bag_of_ngrams(text)
        return None
    
    #----------Phân loại dữ liệu----------
    #Naive Bayes
    @staticmethod
    def train_naive_bayes(train_data, labels):
        return NaiveBayes.train(train_data, labels)

    @staticmethod
    def predict_naive_bayes(model, features):
        features_array = np.array([features])
        predicted_class = model.predict(features_array)[0]
        predicted_proba = model.predict_proba(features_array)
        return predicted_class, predicted_proba

    #Maximizing Likehood
    @staticmethod
    def train_maximizing_likelihood(train_data, labels):
        return MaximizingLikelihood.train(train_data, labels)
    
    @staticmethod
    def predict_maximizing_likelihood(model, features):
        features_array = np.array([features])
        predicted_class = model.predict(features_array)[0]
        predicted_proba = model.predict_proba(features_array)
        return predicted_class, predicted_proba
    
    #SVM
    @staticmethod
    def train_SVM(train_data, labels):
        return SVM.train(train_data, labels)
    
    @staticmethod
    def predict_SVM(model, features):
        features_array = np.array([features])
        predicted_class = model.predict([features])[0] 
        predicted_proba = model.predict_proba([features])[0] 
        return predicted_class, predicted_proba
    
    