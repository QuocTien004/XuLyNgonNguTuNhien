import spacy
import nltk
import contractions
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.stem import PorterStemmer, SnowballStemmer
from spellchecker import SpellChecker
from spacy import displacy

nltk.download('punkt')

class TextPreprocessor:
    def __init__(self, text):
        self.text = text
        self.nlp = spacy.load("en_core_web_sm")  
        self.spell = SpellChecker() 
        self.doc = self.nlp(self.text)  
        self.stemmer = PorterStemmer()
        self.snowball = SnowballStemmer(language="english")

    def sentence_tokenization(self):
        """Tách câu"""
        return "\n".join(f"{i+1}: {sent.text}" for i, sent in enumerate(self.doc.sents))

    def word_tokenization(self):
        """Tách từ"""
        return "\n".join(f"{i+1}: {token.text}" for i, token in enumerate(self.doc))


    def remove_stopwords_punctuation(self):
        """Xóa stop words và dấu câu, hiển thị đẹp hơn"""
        words = [token.text for token in self.doc if not token.is_stop and not token.is_punct]
        return "\n".join(f"{i+1}: {word}" for i, word in enumerate(words))


    def to_lowercase(self):
        """Viết thường tất cả từ"""
        return self.text.lower()

    def stemming(self, method="snowball"):
        """Áp dụng stemming (Porter hoặc Snowball), hiển thị có số thứ tự"""
        stem_func = self.snowball.stem
        return "\n".join(f"{i+1}: {token.text} → {stem_func(token.text)}" for i, token in enumerate(self.doc))


    def lemmatization(self):
        """Đưa từ về dạng gốc, hiển thị có số thứ tự"""
        return "\n".join(f"{i+1}: {token.text} → {token.lemma_}" for i, token in enumerate(self.doc))


    def pos_tagging(self):
        """Xác định từ loại (POS tagging), hiển thị danh sách"""
        return ", ".join(f"{token.text} → {token.pos_}" for token in self.doc)


    def expand_contractions(self):
        """Sửa lại các từ viết tắt"""
        return contractions.fix(self.text)

    def correct_spelling(self):
        """Sửa lỗi chính tả"""
        words = self.text.split()
        return " ".join([self.spell.correction(word) if self.spell.correction(word) else word for word in words])

    def named_entity_recognition(self):
        """Nhận diện thực thể (NER), hiển thị có số thứ tự"""
        return "\n".join(f"{i+1}: {entity.text} → {entity.label_}" for i, entity in enumerate(self.doc.ents))

    
    def summary(self):
        """Tóm tắt tất cả các bước xử lý"""
        print("**Sentence Tokenization:**", self.sentence_tokenization(), "\n")
        print("**Word Tokenization:**", self.word_tokenization(), "\n")
        print("**Remove Stopwords & Punctuation:**", self.remove_stopwords_punctuation(), "\n")
        print("**Lowercase Text:**", self.to_lowercase(), "\n")
        print("**Stemming (Porter):**", self.stemming("porter"), "\n")
        print("**Lemmatization:**", self.lemmatization(), "\n")
        print("**POS Tagging:**", self.pos_tagging(), "\n")
        print("**Expand Contractions:**", self.expand_contractions(), "\n")
        print("**Correct Spelling:**", self.correct_spelling(), "\n")
        print("**Named Entities:**", self.named_entity_recognition(), "\n")