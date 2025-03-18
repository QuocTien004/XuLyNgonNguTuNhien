from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import numpy as np

class NaiveBayes:
    @staticmethod
    def train(train_data, labels):
        model = MultinomialNB()
        model.fit(train_data, labels)
        return model

