from view.aug import augmentation_view
from view.col import scraper_view
from view.pre import preprocessor_view
from view.rep import representation_view
from view.clas import classification_view

import nltk
nltk.download('averaged_perceptron_tagger_eng')

def main():
    scraper_view()
    augmentation_view()
    preprocessor_view()
    representation_view()
    classification_view()

if __name__ == "__main__":
    main()