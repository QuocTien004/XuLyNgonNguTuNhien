from model.processing import TextProcessing
from model.representation import TextRepresentation
from model.augmentation import TextAugmentation

class NLPController:
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

    @staticmethod
    def augment_text(text, methods):
        return TextAugmentation.augment_text(text, methods)