from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
from gensim.models import Word2Vec, FastText
import gensim.downloader as api
from scipy.sparse import csr_matrix
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel, GPT2Tokenizer, GPT2Model
import torch


class DataRepresentation:
    def __init__(self, method="count"):
        self.method = method
        self.vectorizer = None
        self.word2vec_model = None
        self.fasttext_model = None
        self.glove_model = None

    def fit_transform(self, texts):
        if not texts:
            return "Không có dữ liệu!"

        method_map = {
            "count": self._count_vectorizer,
            "onehot": self._onehot_encoding,
            "bagofngram": self._bag_of_ngram,
            "tfidf": self._tfidf_vectorizer,
            "word2vec": self._word2vec_embedding,
            "glove": self._glove_embedding,
            "fasttext": self._fasttext_embedding,
            "chatgpt": self._chatgpt_embedding,
            "bert": self._bert_embedding
        }

        if self.method in method_map:
            return method_map[self.method](texts)
        else:
            return "Phương thức không hợp lệ!"

    def _count_vectorizer(self, texts):
        """Bag of Words sử dụng CountVectorizer."""
        self.vectorizer = CountVectorizer(binary=False)
        transformed = self.vectorizer.fit_transform(texts)
        feature_names = self.vectorizer.get_feature_names_out()
        return self._to_dataframe(transformed, feature_names)

    def _onehot_encoding(self, texts):
        """One-hot encoding chính xác"""
        words = [text.split() for text in texts]
        words_flatten = [[word] for sentence in words for word in sentence]

        encoder = OneHotEncoder(sparse_output=False)
        transformed = encoder.fit_transform(words_flatten)

        feature_names = encoder.get_feature_names_out()
        return self._to_dataframe(transformed, feature_names)

    def _bag_of_ngram(self, texts):
        """Bag of N-grams"""
        self.vectorizer = CountVectorizer(ngram_range=(1, 2))
        transformed = self.vectorizer.fit_transform(texts)
        feature_names = self.vectorizer.get_feature_names_out()
        return self._to_dataframe(transformed, feature_names)

    def _tfidf_vectorizer(self, texts):
        """TF-IDF"""
        self.vectorizer = TfidfVectorizer()
        transformed = self.vectorizer.fit_transform(texts)
        feature_names = self.vectorizer.get_feature_names_out()
        return self._to_dataframe(transformed, feature_names)

    def _word2vec_embedding(self, texts):
        """Word2Vec"""
        tokenized_texts = [text.split() for text in texts]
        self.word2vec_model = Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=1, workers=4)
        embeddings = [np.mean([self.word2vec_model.wv[word] for word in text if word in self.word2vec_model.wv]
                              or [np.zeros(100)], axis=0) for text in tokenized_texts]
        feature_names = [f"feat_{i}" for i in range(len(embeddings[0]))]
        return self._to_dataframe(np.array(embeddings), feature_names)

    def _glove_embedding(self, texts):
        """GloVe"""
        if self.glove_model is None:
            self.glove_model = api.load("glove-wiki-gigaword-100")

        embeddings = []
        for text in texts:
            word_vectors = [self.glove_model.get_vector(word) for word in text.split() if word in self.glove_model]
            if word_vectors:
                embeddings.append(np.mean(word_vectors, axis=0))
            else:
                embeddings.append(np.zeros(100))  

        feature_names = [f"feat_{i}" for i in range(len(embeddings[0]))]
        return self._to_dataframe(np.array(embeddings), feature_names)

    def _fasttext_embedding(self, texts):
        """FastText"""
        tokenized_texts = [text.split() for text in texts]
        self.fasttext_model = FastText(sentences=tokenized_texts, vector_size=100, window=5, min_count=1, workers=4)
        embeddings = [np.mean([self.fasttext_model.wv[word] for word in text if word in self.fasttext_model.wv]
                              or [np.zeros(100)], axis=0) for text in tokenized_texts]
        feature_names = [f"feat_{i}" for i in range(len(embeddings[0]))]
        return self._to_dataframe(np.array(embeddings), feature_names)

    def _bert_embedding(self, texts):
        """BERT"""
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased")
        return self._transform_transformer(texts, tokenizer, model)

    def _chatgpt_embedding(self, texts):
        """GPT-2"""
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2Model.from_pretrained("gpt2")
        return self._transform_transformer(texts, tokenizer, model)

    def _transform_transformer(self, texts, tokenizer, model):
        """Chuyển đổi văn bản sang embeddings từ Transformer models (BERT, RoBERTa)."""
        tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**tokens)

        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        feature_names = [f"feat_{i}" for i in range(embeddings.shape[1])]
        return self._to_dataframe(embeddings, feature_names)

    def _to_dataframe(self, transformed, feature_names):
        """Chuyển đổi dữ liệu thành DataFrame"""
        if isinstance(transformed, np.ndarray):
            df = pd.DataFrame(transformed, columns=feature_names)
        elif isinstance(transformed, csr_matrix):
            df = pd.DataFrame(transformed.toarray(), columns=feature_names)
        else:
            raise ValueError("Dữ liệu không hợp lệ!")

        return df
