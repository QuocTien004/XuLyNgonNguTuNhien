import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import gensim.downloader as api
from gensim.models import FastText

glove_vectors = api.load("glove-wiki-gigaword-100")

class TextRepresentation:
    @staticmethod
    def one_hot_encoding(text):
        words = text.split()
        words_array = np.array(words).reshape(-1, 1)
        onehot_encoder = OneHotEncoder(sparse_output=False)
        onehot_encoded = onehot_encoder.fit_transform(words_array)

        feature_names = onehot_encoder.categories_[0]

        return pd.DataFrame(onehot_encoded, index=words, columns=feature_names)

    @staticmethod
    def bag_of_words(text):  
        words = text.split("\n")
        vectorizer = CountVectorizer()
        x = vectorizer.fit_transform(words)
        feature_names = vectorizer.get_feature_names_out()
        return pd.DataFrame(x.toarray(), columns=feature_names)
    
    @staticmethod
    def tfidf_representation(text):
        words = text.split("\n")
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(words)
        df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
        return df
    
    @staticmethod
    def glove_representation(text):
        words = text.split()
        word_vectors = []

        for word in words:
            try:
                word_vectors.append(glove_vectors[word])
            except KeyError:
                word_vectors.append(np.zeros(glove_vectors.vector_size))

        word_vectors = np.array(word_vectors)
        return pd.DataFrame(word_vectors, index=words, columns=[f"{i}" for i in range(word_vectors.shape[1])])
    
    @staticmethod
    def bag_of_ngrams(text, ngram_range=(1, 3)):
        vectorizer = CountVectorizer(ngram_range=ngram_range)
        processed_docs = text.strip().split("\n") 
        ngram_matrix = vectorizer.fit_transform(processed_docs)
        feature_names = vectorizer.get_feature_names_out()
        return pd.DataFrame(ngram_matrix.toarray(), columns=feature_names)

    @staticmethod
    def fasttext_representation(text):
        sentences = [word.split() for word in text.split("\n")]
        model = FastText(sentences, vector_size=100, window=5, min_count=1)
        model.train(sentences, total_examples=len(sentences), epochs=10)
    
        unique_words = list(set(text.split()))
    
        word_vectors = {word: model.wv[word] for word in unique_words if word in model.wv}
    
        return pd.DataFrame(word_vectors).T