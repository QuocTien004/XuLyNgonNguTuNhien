from view.home import Menu
from view.aug import augmentation_view
from view.col import scraper_view
from view.pre import preprocessor_view
from view.rep import representation_view
from view.clas import classification_view

import nltk
nltk.download('averaged_perceptron_tagger_eng')

def main():
    # choice = Menu()

    augmentation_view()
    scraper_view()
    preprocessor_view()
    representation_view()
    classification_view()

    # if choice == "Tăng cường dữ liệu":
    #     return augmentation_view()

    # elif choice == "Thu thập dữ liệu":
    #     return scraper_view()

    # elif choice == "Tiền xử lý dữ liệu":
    #     return preprocessor_view()

    # elif choice == "Biểu diễn dữ liệu":
    #     return representation_view()
    
    # elif choice == "Phân loại dữ liệu":
    #     return classification_view()

if __name__ == "__main__":
    main()