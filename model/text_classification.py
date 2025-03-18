from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np

class NaiveBayes:
    @staticmethod
    def train(train_data, labels):
        model = MultinomialNB()
        model.fit(train_data, labels)
        return model

class MaximizingLikelihood:
    @staticmethod
    def train(train_data, labels):
        model = LogisticRegression()
        model.fit(train_data, labels)
        return model

class SVM:
    @staticmethod
    def train(train_data, labels):
        model = SVC(probability=True)
        model.fit(train_data, labels)
        return model
