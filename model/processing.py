import spacy
import string
from spellchecker import SpellChecker
import contractions

nlp = spacy.load("en_core_web_sm")

class TextProcessing:
    @staticmethod
    def sentence_tokenization(text):
        doc = nlp(text)
        return "\n".join([sent.text for sent in doc.sents])

    @staticmethod
    def remove_stopwords(text):
        doc = nlp(text)
        return " ".join([token.text for token in doc if not token.is_stop and token.text not in string.punctuation])

    @staticmethod
    def remove_punctuation(text):
        doc = nlp(text)
        return " ".join([token.text for token in doc if token.text not in string.punctuation])

    @staticmethod
    def stemming(text):
        doc = nlp(text)
        return " ".join([token.lemma_ for token in doc])

    @staticmethod
    def named_entity_recognition(text):
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return "\n".join([f"{ent[0]} ({ent[1]})" for ent in entities]) if entities else "Không tìm thấy thực thể nào."

    @staticmethod
    def correct_spelling(text):
        spell = SpellChecker()
        words = text.split()
        return " ".join([spell.correction(word) for word in words])

    @staticmethod
    def fix_contractions(text):
        return contractions.fix(text)